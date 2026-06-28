//! A hand-rolled feed-forward network and its **twin** (value + input-gradient).
//!
//! The net `g_θ: Rⁿ → R` is a stack of affine layers with **softplus** hidden
//! activations and a **linear** output. There is no ML framework here: the forward
//! pass and the input-gradient pass are plain linear algebra over `Vec<f64>`,
//! seeded deterministically from [`oxis_core::path_seed`] so a `(seed)` fixes the
//! whole model.
//!
//! The *twin* pass computes `∂y/∂x` by ordinary backpropagation through the same
//! shared weights — this is Huge & Savine's twin network, and it is also what the
//! pricing surface needs (the option's delta). Training that prediction (in
//! [`crate::train`]) backpropagates once more through this pass; the caches stored
//! here (`a`, `z`, `delta`, `tprime`) are exactly what that second pass consumes.

use crate::activation::{softplus, softplus_prime};
use rand::Rng;
use rand::rngs::SmallRng;
use rand_distr::StandardNormal;

/// One affine layer `z ↦ W·z + b`. `w[i][j]` is the weight from input `j` to
/// output `i`; `b[i]` the bias of output `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Weights, shape `[out][in]`.
    pub w: Vec<Vec<f64>>,
    /// Biases, length `out`.
    pub b: Vec<f64>,
}

/// A feed-forward network: `L` layers, softplus hidden, linear scalar output.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    /// Dimension of the input vector.
    pub input_dim: usize,
    /// The affine layers, in order; the last one is the linear output layer.
    pub layers: Vec<Layer>,
}

/// Forward-pass cache: `z[0] = x`, `z[k+1]` the output of layer `k`, and `a[k]`
/// the pre-activation of layer `k`.
pub struct Forward {
    /// Layer activations, `z[0..=L]` (`z[0]` is the input).
    pub z: Vec<Vec<f64>>,
    /// Pre-activations, `a[0..L]` (one per layer).
    pub a: Vec<Vec<f64>>,
}

/// Twin-pass cache for the input-gradient `∂y/∂x`.
pub struct Twin {
    /// Adjoints `delta[k] = ∂y/∂a[k]` (one per layer).
    pub delta: Vec<Vec<f64>>,
    /// `tprime[k] = Wᵀ_{k+1}·delta[k+1] = ∂y/∂z[k+1]`, for `k = 0..L-1`.
    pub tprime: Vec<Vec<f64>>,
    /// The input-gradient `g = ∂y/∂x`.
    pub grad: Vec<f64>,
}

impl Mlp {
    /// Build a net with the given hidden widths (the output layer of width 1 is
    /// appended automatically), initialising weights `~ N(0, 1/√fan_in)` and biases
    /// to zero from a deterministic RNG.
    pub fn new(input_dim: usize, hidden: &[usize], rng: &mut SmallRng) -> Self {
        let mut dims = Vec::with_capacity(hidden.len() + 2);
        dims.push(input_dim);
        dims.extend_from_slice(hidden);
        dims.push(1); // scalar output

        let mut layers = Vec::with_capacity(dims.len() - 1);
        for k in 0..dims.len() - 1 {
            let (fan_in, fan_out) = (dims[k], dims[k + 1]);
            let scale = (1.0 / fan_in as f64).sqrt();
            let w = (0..fan_out)
                .map(|_| {
                    (0..fan_in)
                        .map(|_| scale * rng.sample::<f64, _>(StandardNormal))
                        .collect()
                })
                .collect();
            layers.push(Layer {
                w,
                b: vec![0.0; fan_out],
            });
        }
        Self { input_dim, layers }
    }

    /// Number of layers (hidden + output).
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Run the forward pass, caching activations for the twin / training passes.
    pub fn forward(&self, x: &[f64]) -> Forward {
        let l = self.layers.len();
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(l + 1);
        let mut a: Vec<Vec<f64>> = Vec::with_capacity(l);
        z.push(x.to_vec());
        for (k, layer) in self.layers.iter().enumerate() {
            let ak = affine(&layer.w, &layer.b, &z[k]);
            // Hidden layers use softplus; the last (output) layer is linear.
            let zk1 = if k + 1 == l {
                ak.clone()
            } else {
                ak.iter().map(|&v| softplus(v)).collect()
            };
            a.push(ak);
            z.push(zk1);
        }
        Forward { z, a }
    }

    /// The scalar value `y = g_θ(x)` from a forward cache.
    pub fn value(&self, fwd: &Forward) -> f64 {
        fwd.z[self.layers.len()][0]
    }

    /// The twin pass: compute `∂y/∂x` by backpropagating a unit seed through the
    /// shared weights. Returns the caches the training gradient needs.
    pub fn twin(&self, fwd: &Forward) -> Twin {
        let l = self.layers.len();
        let mut delta: Vec<Vec<f64>> = vec![Vec::new(); l];
        let mut tprime: Vec<Vec<f64>> = vec![Vec::new(); l];

        // Output layer is linear: ∂y/∂a[L-1] = 1.
        delta[l - 1] = vec![1.0];
        // Propagate down through the hidden layers.
        for k in (0..l - 1).rev() {
            let tp = matvec_t(&self.layers[k + 1].w, &delta[k + 1]); // ∂y/∂z[k+1]
            let dk: Vec<f64> = fwd.a[k]
                .iter()
                .zip(tp.iter())
                .map(|(&ak, &t)| softplus_prime(ak) * t)
                .collect();
            tprime[k] = tp;
            delta[k] = dk;
        }
        // Input-gradient: ∂y/∂x = W₀ᵀ·delta[0].
        let grad = matvec_t(&self.layers[0].w, &delta[0]);
        Twin {
            delta,
            tprime,
            grad,
        }
    }

    /// Convenience: value and input-gradient together.
    pub fn predict_with_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let fwd = self.forward(x);
        let twin = self.twin(&fwd);
        (self.value(&fwd), twin.grad)
    }
}

/// `W·v + b` (affine map), `W` shaped `[out][in]`.
pub(crate) fn affine(w: &[Vec<f64>], b: &[f64], v: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            bi + row
                .iter()
                .zip(v.iter())
                .map(|(&wij, &vj)| wij * vj)
                .sum::<f64>()
        })
        .collect()
}

/// `W·v` for `W` shaped `[out][in]`, returning a length-`out` vector.
pub(crate) fn matvec(w: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&wij, &vj)| wij * vj).sum())
        .collect()
}

/// `Wᵀ·v` for `W` shaped `[out][in]`, returning a length-`in` vector.
pub(crate) fn matvec_t(w: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let in_dim = w.first().map_or(0, |r| r.len());
    let mut out = vec![0.0; in_dim];
    for (row, &vi) in w.iter().zip(v.iter()) {
        for (o, &wij) in out.iter_mut().zip(row.iter()) {
            *o += wij * vi;
        }
    }
    out
}

/// Outer product `a ⊗ b`, shape `[a.len()][b.len()]`.
pub(crate) fn outer(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|&ai| b.iter().map(|&bj| ai * bj).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxis_core::path_seed;
    use rand::SeedableRng;

    fn net() -> Mlp {
        let mut rng = SmallRng::seed_from_u64(path_seed(7, 0));
        Mlp::new(1, &[6, 6], &mut rng)
    }

    #[test]
    fn shapes_are_consistent() {
        let m = net();
        assert_eq!(m.depth(), 3); // 2 hidden + output
        let fwd = m.forward(&[1.2]);
        assert_eq!(fwd.z.len(), 4);
        assert_eq!(fwd.z[0], vec![1.2]);
        assert_eq!(m.value(&fwd), fwd.z[3][0]);
    }

    #[test]
    fn twin_grad_matches_finite_difference() {
        // The twin pass must equal a numerical derivative of the forward value.
        let m = net();
        let h = 1e-6;
        for &x in &[-2.0, -0.3, 0.5, 1.7, 3.0] {
            let (_y, g) = m.predict_with_grad(&[x]);
            let fd = (m.value(&m.forward(&[x + h])) - m.value(&m.forward(&[x - h]))) / (2.0 * h);
            assert!((g[0] - fd).abs() < 1e-6, "x={x}: twin {} vs fd {fd}", g[0]);
        }
    }

    #[test]
    fn deterministic_init() {
        let mut r1 = SmallRng::seed_from_u64(path_seed(11, 0));
        let mut r2 = SmallRng::seed_from_u64(path_seed(11, 0));
        let a = Mlp::new(2, &[4], &mut r1);
        let b = Mlp::new(2, &[4], &mut r2);
        assert_eq!(a, b);
    }
}
