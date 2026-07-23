//! Shared optimisation primitives for the hand-rolled networks: a
//! parameter-shaped gradient buffer, the Adam optimiser, a one-cycle
//! learning-rate schedule, z-score standardisation, and a plain reverse-mode
//! value backprop.
//!
//! These were factored out of the differential-ML trainer ([`crate::ml::train`]) so
//! the American engines ([`crate::ml::deep_lsm`], [`crate::ml::dos`]) can reuse exactly
//! the same optimiser and value backprop. The doubled-network *twin* backprop
//! (which differentiates the input-gradient output) stays in `train.rs`; what
//! lives here is the machinery common to every objective.

use crate::ml::activation::softplus_prime;
use crate::ml::mlp::{Forward, Mlp, matvec_t, outer};

/// Mean and (population) standard deviation; std floored at a tiny value so the
/// standardisation never divides by zero.
pub(crate) fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt().max(1e-12))
}

/// Piecewise-linear one-cycle learning-rate schedule over the training fraction
/// `p ∈ [0, 1]` (Adam on standardized data).
pub(crate) fn one_cycle_lr(p: f64) -> f64 {
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
pub(crate) struct Grad {
    pub(crate) gw: Vec<Vec<Vec<f64>>>,
    pub(crate) gb: Vec<Vec<f64>>,
}

impl Grad {
    pub(crate) fn zeros_like(mlp: &Mlp) -> Self {
        let gw = mlp
            .layers
            .iter()
            .map(|l| l.w.iter().map(|r| vec![0.0; r.len()]).collect())
            .collect();
        let gb = mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        Self { gw, gb }
    }
}

/// `acc += a ⊗ b` (outer product accumulation).
pub(crate) fn accum_outer(acc: &mut [Vec<f64>], a: &[f64], b: &[f64]) {
    let add = outer(a, b);
    for (row, arow) in acc.iter_mut().zip(add) {
        for (g, v) in row.iter_mut().zip(arow) {
            *g += v;
        }
    }
}

/// Reverse-mode backprop of a scalar output adjoint `dy = ∂L/∂y` through the
/// linear-output softplus MLP, accumulating `∂L/∂θ` into `grad`.
///
/// This is the ordinary forward/value reverse sweep (no twin pass): it seeds the
/// output pre-activation with `dy` and walks the layers backward. It is exactly
/// what the differential trainer inlines for its value term, exposed so the
/// American engines can train on plain scalar objectives (MSE continuation,
/// expected-payoff stopping).
pub(crate) fn backward_value(mlp: &Mlp, fwd: &Forward, dy: f64, grad: &mut Grad) {
    let l = mlp.layers.len();
    let mut d_a: Vec<Vec<f64>> = (0..l).map(|k| vec![0.0; fwd.a[k].len()]).collect();
    // Linear scalar output: ∂L/∂a[L-1] = dy.
    d_a[l - 1][0] += dy;
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
}

/// Adam optimiser state, mirroring the parameter shapes.
pub(crate) struct Adam {
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

    pub(crate) fn new(mlp: &Mlp) -> Self {
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

    pub(crate) fn step(&mut self, mlp: &mut Mlp, grad: &Grad, lr: f64) {
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
