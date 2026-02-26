use crate::math::bijective;
use ndarray::{Array1, Array2};

/// Generates a deterministic weight matrix for the Implicit Diffusion Filter
/// purely out of the 64-bit address space, consuming 0 explicit memory.
pub fn generate_implicit_weights(
    in_features: usize,
    out_features: usize,
    seed: u64,
    scale: f64,
) -> Array2<f64> {
    let mut weights = Array2::<f64>::zeros((out_features, in_features));

    for r in 0..out_features {
        for c in 0..in_features {
            // Replicate the Feistel coordinate XOR coordinate logic
            // coords = seed ^ row ^ col
            let coords = seed ^ (r as u64) ^ (c as u64);

            // Map via the Generative Memory v_mask
            let hash = bijective::vl_mask(coords, seed);

            // Normalize to [-1.0, 1.0]
            // We cast directly keeping the maximum bound in mind to avoid precision destruction
            let mut val = (hash as f64 / (u64::MAX as f64)) * 2.0 - 1.0;
            val *= scale;

            weights[[r, c]] = val;
        }
    }

    weights
}

/// A pure Rust Implicit Linear Layer (Holographic Filter).
pub struct ImplicitLinear {
    in_features: usize,
    out_features: usize,
    seed: u64,
    scale: f64,
}

impl ImplicitLinear {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let scale = if in_features > 0 {
            1.0 / (in_features as f64).sqrt()
        } else {
            1.0
        };

        Self {
            in_features,
            out_features,
            seed,
            scale,
        }
    }

    /// Forward pass of the Holographic filter: executes matmul deterministically.
    /// Expects a 1D vector `x` of length `in_features`.
    /// Returns a 1D vector of length `out_features`.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let weights =
            generate_implicit_weights(self.in_features, self.out_features, self.seed, self.scale);
        // Compute x * W^T. Since `weights` is (out_features, in_features),
        // we can just dot `weights` with `x` mathematically yielding the equivalent to PyTorch's `linear`.
        weights.dot(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_linear_forward() {
        let layer = ImplicitLinear::new(128, 64, 42);

        // Input: 128-dimensional vector of ones
        let x = Array1::<f64>::ones(128);

        let out = layer.forward(&x);

        assert_eq!(out.len(), 64);

        // Check output is deterministic across repeated identical initializations
        let layer2 = ImplicitLinear::new(128, 64, 42);
        let out2 = layer2.forward(&x);

        for i in 0..64 {
            assert!((out[i] - out2[i]).abs() < 1e-10);
        }
    }
}
