//! Differential training of the twin network, hand-rolled end to end.
//!
//! The loss mixes a value term and a differential term (Huge & Savine):
//! `L = α·mean(ŷ − y)² + β·mean_j λ_j²(ĝ_j − q_j)²`, on standardized inputs and
//! labels, with per-input RMS weights `λ_j = 1/√mean(q_j²)`, `α = 1/(1+n)`,
//! `β = 1−α`. Training it needs `∂L/∂θ` where the predicted differential `ĝ`
//! *itself* depends on `θ` — i.e. a backprop **through the twin (input-gradient)
//! pass**. That collapses to one ordinary reverse sweep over the doubled network,
//! using the softplus second derivative `σ''`; it is implemented analytically in
//! [`accumulate_grad`] and gated by a finite-difference gradient check in the
//! tests. Optimisation is Adam with a one-cycle learning-rate schedule.

use crate::activation::{softplus_prime, softplus_second};
use crate::data::{BsSpec, generate_european};
use crate::mlp::{Mlp, matvec, matvec_t, outer};
use oxis_core::{OxisError, path_seed};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// Hyperparameters for training a differential-ML surrogate.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// The contract + market to learn the pricing function of.
    pub spec: BsSpec,
    /// Number of simulated training samples.
    pub n_samples: usize,
    /// Hidden-layer widths (a linear scalar output is appended).
    pub hidden: Vec<usize>,
    /// Training epochs.
    pub epochs: usize,
    /// Log-normal spread of input spots (multiple of `σ√τ`).
    pub spread: f64,
    /// RNG seed — fixes data, initialisation, and shuffling.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            spec: BsSpec {
                spot: 100.0,
                strike: 100.0,
                rate: 0.05,
                vol: 0.2,
                maturity: 1.0,
                option_type: oxis_core::OptionType::Call,
            },
            n_samples: 8192,
            hidden: vec![32, 32],
            epochs: 100,
            spread: 2.0,
            seed: 1,
        }
    }
}

/// A trained surrogate: the network plus the standardisation needed to map raw
/// spots in and raw prices/deltas out.
#[derive(Debug, Clone)]
pub struct TrainedModel {
    mlp: Mlp,
    x_mean: Vec<f64>,
    x_std: Vec<f64>,
    y_mean: f64,
    y_std: f64,
    /// Final training loss (on standardized data).
    pub final_loss: f64,
    /// Epochs run.
    pub epochs: usize,
    /// Training samples used.
    pub n_samples: usize,
}

impl TrainedModel {
    /// Predicted price and delta at `spot`.
    pub fn price_and_delta(&self, spot: f64) -> (f64, f64) {
        let xn = (spot - self.x_mean[0]) / self.x_std[0];
        let (y_norm, grad) = self.mlp.predict_with_grad(&[xn]);
        let price = self.y_mean + self.y_std * y_norm;
        let delta = grad[0] * self.y_std / self.x_std[0];
        (price, delta)
    }

    /// Predicted price at `spot`.
    pub fn price(&self, spot: f64) -> f64 {
        self.price_and_delta(spot).0
    }
}

/// Train a differential-ML surrogate for `cfg.spec`.
///
/// # Errors
/// [`OxisError::InvalidInput`] from data generation or for an empty hidden spec.
pub fn train_differential(cfg: &TrainConfig) -> Result<TrainedModel, OxisError> {
    if cfg.hidden.is_empty() {
        return Err(OxisError::invalid_input("hidden layers must be non-empty"));
    }
    if cfg.epochs == 0 {
        return Err(OxisError::invalid_input("epochs must be >= 1"));
    }
    let data = generate_european(&cfg.spec, cfg.n_samples, cfg.spread, cfg.seed)?;

    // Standardize inputs (single feature: spot) and labels.
    let m = data.len();
    let x: Vec<f64> = data.iter().map(|d| d.x).collect();
    let y: Vec<f64> = data.iter().map(|d| d.y).collect();
    let q: Vec<f64> = data.iter().map(|d| d.dydx).collect();
    let (x_mean, x_std) = mean_std(&x);
    let (y_mean, y_std) = mean_std(&y);

    let xn: Vec<f64> = x.iter().map(|&v| (v - x_mean) / x_std).collect();
    let yn: Vec<f64> = y.iter().map(|&v| (v - y_mean) / y_std).collect();
    // Differentials transform by the chain rule: d(y_norm)/d(x_norm) = q·x_std/y_std.
    let qn: Vec<f64> = q.iter().map(|&v| v * x_std / y_std).collect();

    // Differential weighting (per input; n = 1 here).
    let n_in = 1usize;
    let q_rms = (qn.iter().map(|&v| v * v).sum::<f64>() / m as f64).sqrt();
    let lambda = if q_rms > 0.0 { 1.0 / q_rms } else { 1.0 };
    let alpha = 1.0 / (1.0 + n_in as f64);
    let beta = 1.0 - alpha;

    // Initialise the network and Adam state.
    let mut rng = SmallRng::seed_from_u64(path_seed(cfg.seed, usize::MAX));
    let mut mlp = Mlp::new(1, &cfg.hidden, &mut rng);
    let mut adam = Adam::new(&mlp);

    let batch_size = m.min(256.max(m / 16)).max(1);
    let batches = m.div_ceil(batch_size);
    let total_steps = (cfg.epochs * batches).max(1);

    let mut idx: Vec<usize> = (0..m).collect();
    let mut shuffler = SmallRng::seed_from_u64(path_seed(cfg.seed, 0xABCD));
    let mut step = 0usize;
    let mut final_loss = 0.0;

    for _epoch in 0..cfg.epochs {
        idx.shuffle(&mut shuffler);
        for batch in idx.chunks(batch_size) {
            let lr = one_cycle_lr(step as f64 / total_steps as f64);
            let (loss, grad) = batch_grad(&mlp, &xn, &yn, &qn, batch, alpha, beta, lambda);
            adam.step(&mut mlp, &grad, lr);
            final_loss = loss;
            step += 1;
        }
    }

    Ok(TrainedModel {
        mlp,
        x_mean: vec![x_mean],
        x_std: vec![x_std],
        y_mean,
        y_std,
        final_loss,
        epochs: cfg.epochs,
        n_samples: m,
    })
}

/// Mean and (population) standard deviation; std floored at a tiny value so the
/// standardisation never divides by zero.
fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt().max(1e-12))
}

/// Piecewise-linear one-cycle learning-rate schedule over the training fraction
/// `p ∈ [0, 1]` (Adam on standardized data).
fn one_cycle_lr(p: f64) -> f64 {
    const KNOTS: [(f64, f64); 5] = [
        (0.0, 1.0e-8),
        (0.2, 0.1),
        (0.6, 0.01),
        (0.9, 1.0e-6),
        (1.0, 1.0e-8),
    ];
    let p = p.clamp(0.0, 1.0);
    for w in KNOTS.windows(2) {
        let (p0, l0) = w[0];
        let (p1, l1) = w[1];
        if p <= p1 {
            let t = if p1 > p0 { (p - p0) / (p1 - p0) } else { 0.0 };
            return l0 + t * (l1 - l0);
        }
    }
    KNOTS[KNOTS.len() - 1].1
}

/// A parameter-shaped gradient buffer (mirrors the layers' `(w, b)`).
struct Grad {
    gw: Vec<Vec<Vec<f64>>>,
    gb: Vec<Vec<f64>>,
}

impl Grad {
    fn zeros_like(mlp: &Mlp) -> Self {
        let gw = mlp
            .layers
            .iter()
            .map(|l| l.w.iter().map(|r| vec![0.0; r.len()]).collect())
            .collect();
        let gb = mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        Self { gw, gb }
    }
}

/// Mean loss and gradient over a minibatch.
#[allow(clippy::too_many_arguments)]
fn batch_grad(
    mlp: &Mlp,
    xn: &[f64],
    yn: &[f64],
    qn: &[f64],
    batch: &[usize],
    alpha: f64,
    beta: f64,
    lambda: f64,
) -> (f64, Grad) {
    let mut grad = Grad::zeros_like(mlp);
    let mut loss = 0.0;
    for &i in batch {
        loss += accumulate_grad(mlp, xn[i], yn[i], qn[i], alpha, beta, lambda, &mut grad);
    }
    let scale = 1.0 / batch.len() as f64;
    for layer in grad.gw.iter_mut() {
        for row in layer.iter_mut() {
            for g in row.iter_mut() {
                *g *= scale;
            }
        }
    }
    for layer in grad.gb.iter_mut() {
        for g in layer.iter_mut() {
            *g *= scale;
        }
    }
    (loss * scale, grad)
}

/// Add one sample's contribution to `grad` and return its (unscaled) loss.
///
/// This is the doubled-network reverse sweep: it backpropagates through both the
/// value output and the twin (input-gradient) output, so the differential term of
/// the loss is differentiated w.r.t. the weights. `σ''` enters via the twin pass.
#[allow(clippy::too_many_arguments)]
fn accumulate_grad(
    mlp: &Mlp,
    x: f64,
    y_label: f64,
    q_label: f64,
    alpha: f64,
    beta: f64,
    lambda: f64,
    grad: &mut Grad,
) -> f64 {
    let l = mlp.layers.len();
    let fwd = mlp.forward(&[x]);
    let twin = mlp.twin(&fwd);
    let y_hat = mlp.value(&fwd);
    let g_hat = twin.grad[0];

    // Loss + output adjoints (n = 1 input).
    let val_res = y_hat - y_label;
    let diff_res = g_hat - q_label;
    let loss = alpha * val_res * val_res + beta * lambda * lambda * diff_res * diff_res;
    let gy = 2.0 * alpha * val_res; // ∂L/∂ŷ
    let gg = vec![2.0 * beta * lambda * lambda * diff_res]; // ∂L/∂ĝ

    // Adjoint accumulators over pre-activations and over the twin adjoints `u`.
    let mut d_a: Vec<Vec<f64>> = (0..l).map(|k| vec![0.0; fwd.a[k].len()]).collect();
    let mut d_u: Vec<Vec<f64>> = (0..l).map(|k| vec![0.0; twin.delta[k].len()]).collect();

    // --- Reverse the twin (input-gradient) pass ---
    // g = W₀ᵀ·u₀  →  seeds d_u[0] and ∂L/∂W₀.
    let du0 = matvec(&mlp.layers[0].w, &gg); // W₀ · d_g
    for (acc, v) in d_u[0].iter_mut().zip(du0) {
        *acc += v;
    }
    accum_outer(&mut grad.gw[0], &twin.delta[0], &gg);

    for k in 0..l - 1 {
        // u[k] = σ'(a[k]) ⊙ tprime[k];  tprime[k] = W_{k+1}ᵀ·u[k+1].
        let sp: Vec<f64> = fwd.a[k].iter().map(|&a| softplus_prime(a)).collect();
        let spp: Vec<f64> = fwd.a[k].iter().map(|&a| softplus_second(a)).collect();
        let d_t: Vec<f64> = sp.iter().zip(d_u[k].iter()).map(|(&s, &d)| s * d).collect();
        // d_a[k] += σ''(a[k]) ⊙ tprime[k] ⊙ d_u[k].
        for j in 0..d_a[k].len() {
            d_a[k][j] += spp[j] * twin.tprime[k][j] * d_u[k][j];
        }
        // d_u[k+1] += W_{k+1}·d_t;   ∂L/∂W_{k+1} += u[k+1] ⊗ d_t.
        let du_next = matvec(&mlp.layers[k + 1].w, &d_t);
        for (acc, v) in d_u[k + 1].iter_mut().zip(du_next) {
            *acc += v;
        }
        accum_outer(&mut grad.gw[k + 1], &twin.delta[k + 1], &d_t);
    }

    // --- Reverse the forward (value) pass ---
    for (acc, v) in d_a[l - 1].iter_mut().zip(std::iter::once(gy)) {
        *acc += v; // linear output: ∂L/∂a[L-1] += gy
    }
    for k in (0..l).rev() {
        accum_outer(&mut grad.gw[k], &d_a[k], &fwd.z[k]);
        for (gb, &da) in grad.gb[k].iter_mut().zip(d_a[k].iter()) {
            *gb += da;
        }
        if k > 0 {
            let d_z = matvec_t(&mlp.layers[k].w, &d_a[k]);
            // z[k] = σ(a[k-1]) for a hidden layer.
            for j in 0..d_a[k - 1].len() {
                d_a[k - 1][j] += softplus_prime(fwd.a[k - 1][j]) * d_z[j];
            }
        }
    }

    loss
}

/// `acc += a ⊗ b` (outer product accumulation).
fn accum_outer(acc: &mut [Vec<f64>], a: &[f64], b: &[f64]) {
    let add = outer(a, b);
    for (row, arow) in acc.iter_mut().zip(add) {
        for (g, v) in row.iter_mut().zip(arow) {
            *g += v;
        }
    }
}

/// Adam optimiser state, mirroring the parameter shapes.
struct Adam {
    mw: Vec<Vec<Vec<f64>>>,
    vw: Vec<Vec<Vec<f64>>>,
    mb: Vec<Vec<f64>>,
    vb: Vec<Vec<f64>>,
    t: i32,
}

impl Adam {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;

    fn new(mlp: &Mlp) -> Self {
        let g = Grad::zeros_like(mlp);
        let g2 = Grad::zeros_like(mlp);
        Self {
            mw: g.gw,
            vw: g2.gw,
            mb: g.gb,
            vb: g2.gb,
            t: 0,
        }
    }

    fn step(&mut self, mlp: &mut Mlp, grad: &Grad, lr: f64) {
        self.t += 1;
        let bc1 = 1.0 - Self::B1.powi(self.t);
        let bc2 = 1.0 - Self::B2.powi(self.t);
        for k in 0..mlp.layers.len() {
            adam_update_matrix(
                &mut mlp.layers[k].w,
                &grad.gw[k],
                &mut self.mw[k],
                &mut self.vw[k],
                lr,
                bc1,
                bc2,
            );
            adam_update_vector(
                &mut mlp.layers[k].b,
                &grad.gb[k],
                &mut self.mb[k],
                &mut self.vb[k],
                lr,
                bc1,
                bc2,
            );
        }
    }
}

fn adam_update_matrix(
    w: &mut [Vec<f64>],
    g: &[Vec<f64>],
    m: &mut [Vec<f64>],
    v: &mut [Vec<f64>],
    lr: f64,
    bc1: f64,
    bc2: f64,
) {
    for i in 0..w.len() {
        adam_update_vector(&mut w[i], &g[i], &mut m[i], &mut v[i], lr, bc1, bc2);
    }
}

fn adam_update_vector(
    w: &mut [f64],
    g: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    lr: f64,
    bc1: f64,
    bc2: f64,
) {
    for j in 0..w.len() {
        m[j] = Adam::B1 * m[j] + (1.0 - Adam::B1) * g[j];
        v[j] = Adam::B2 * v[j] + (1.0 - Adam::B2) * g[j] * g[j];
        let mhat = m[j] / bc1;
        let vhat = v[j] / bc2;
        w[j] -= lr * mhat / (vhat.sqrt() + Adam::EPS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxis_core::OptionType;
    use rand::rngs::SmallRng;

    fn spec() -> BsSpec {
        BsSpec {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            vol: 0.2,
            maturity: 1.0,
            option_type: OptionType::Call,
        }
    }

    /// The analytic doubled-network gradient must match central finite differences
    /// of the combined loss — the safety net for the hardest piece of M8.
    #[test]
    fn gradient_check() {
        let mut rng = SmallRng::seed_from_u64(path_seed(3, 0));
        let mlp = Mlp::new(1, &[5, 4], &mut rng);
        let (x, y, q) = (0.6_f64, 0.3_f64, 0.7_f64);
        let (alpha, beta, lambda) = (0.5, 0.5, 1.3);

        // Analytic gradient for the single sample.
        let mut grad = Grad::zeros_like(&mlp);
        accumulate_grad(&mlp, x, y, q, alpha, beta, lambda, &mut grad);

        // Loss as a function of one perturbed parameter.
        let loss_at = |m: &Mlp| -> f64 {
            let fwd = m.forward(&[x]);
            let twin = m.twin(&fwd);
            let vr = m.value(&fwd) - y;
            let dr = twin.grad[0] - q;
            alpha * vr * vr + beta * lambda * lambda * dr * dr
        };

        let h = 1e-6;
        let mut worst = 0.0_f64;
        for k in 0..mlp.layers.len() {
            for i in 0..mlp.layers[k].w.len() {
                for j in 0..mlp.layers[k].w[i].len() {
                    let mut up = mlp.clone();
                    let mut dn = mlp.clone();
                    up.layers[k].w[i][j] += h;
                    dn.layers[k].w[i][j] -= h;
                    let fd = (loss_at(&up) - loss_at(&dn)) / (2.0 * h);
                    worst = worst.max((fd - grad.gw[k][i][j]).abs());
                }
            }
            for i in 0..mlp.layers[k].b.len() {
                let mut up = mlp.clone();
                let mut dn = mlp.clone();
                up.layers[k].b[i] += h;
                dn.layers[k].b[i] -= h;
                let fd = (loss_at(&up) - loss_at(&dn)) / (2.0 * h);
                worst = worst.max((fd - grad.gb[k][i]).abs());
            }
        }
        assert!(worst < 1e-5, "worst grad mismatch {worst:.3e}");
    }

    #[test]
    #[ignore = "calibration: run with --release --ignored --nocapture to size bands"]
    fn calibrate_grid_accuracy() {
        use crate::result::black_scholes_price_delta;
        let cfg = TrainConfig {
            spec: spec(),
            n_samples: 4096,
            hidden: vec![30, 30],
            epochs: 60,
            spread: 2.0,
            seed: 1,
        };
        let model = train_differential(&cfg).unwrap();
        let mut pmax = 0.0_f64;
        let mut dmax = 0.0_f64;
        let mut psse = 0.0_f64;
        let mut dsse = 0.0_f64;
        let mut count = 0;
        let mut s = 80.0;
        while s <= 120.0 + 1e-9 {
            let mut sp = cfg.spec;
            sp.spot = s;
            let (bp, bd) = black_scholes_price_delta(&sp).unwrap();
            let (mp, md) = model.price_and_delta(s);
            pmax = pmax.max((mp - bp).abs());
            dmax = dmax.max((md - bd).abs());
            psse += (mp - bp).powi(2);
            dsse += (md - bd).powi(2);
            count += 1;
            s += 2.5;
        }
        let prmse = (psse / count as f64).sqrt();
        let drmse = (dsse / count as f64).sqrt();
        eprintln!(
            "grid[80..120] price: max={pmax:.4} rmse={prmse:.4} | delta: max={dmax:.4} rmse={drmse:.4} | final_loss={:.4}",
            model.final_loss
        );
    }

    #[test]
    fn one_cycle_schedule_bounds() {
        assert!((one_cycle_lr(0.0) - 1e-8).abs() < 1e-12);
        assert!((one_cycle_lr(0.2) - 0.1).abs() < 1e-12);
        assert!(one_cycle_lr(0.4) > 0.01 && one_cycle_lr(0.4) < 0.1);
        assert!((one_cycle_lr(1.0) - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn trains_and_prices_near_bs() {
        // A modest config should land the ATM price within a sensible band.
        let cfg = TrainConfig {
            spec: spec(),
            n_samples: 4096,
            hidden: vec![24, 24],
            epochs: 60,
            spread: 2.0,
            seed: 5,
        };
        let model = train_differential(&cfg).unwrap();
        let (price, delta) = model.price_and_delta(100.0);
        // BS ATM call ≈ 10.45, delta ≈ 0.637.
        assert!((price - 10.45).abs() < 1.0, "price {price}");
        assert!((delta - 0.637).abs() < 0.1, "delta {delta}");
    }
}
