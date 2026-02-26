//! Semantic Compiling Engine (Inverse Projection)
//!
//! Exposes an `AnchorNavigator` that utilizes Singular Value Decomposition (SVD)
//! via the `faer` Linear Algebra library to calculate morphological bounds that
//! map the $2^{64}$ manifold onto external physical datasets.
//!
//! Mathematics:
//! $ A_{i, j} \approx K_{i} \times W^{-1} \times R_{j} $

use ndarray::Array2;

/// Analyzes an arbitrary physical matrix and computes the $W^{-1}$ pseudo-inverse.
/// This allows us to "compress" external matrices into purely structural topology bounds.
pub struct AnchorNavigator {
    pub k_mat: Array2<f64>,
    pub r_mat: Array2<f64>,
    pub w_inv: Array2<f64>,
}

impl AnchorNavigator {
    /// Maps a target matrix of size (m x n) using an anchor subset size $S$.
    /// The matrix is parameterized by a basis `a` and projection `b`.
    pub fn new(a: &Array2<f64>, b: &Array2<f64>, s: usize) -> Self {
        let (m, k) = a.dim();
        let n = b.dim().1;
        let s = s.min(m).min(n);

        // 1. Selector logic (simplified for proof of concept):
        // In full production, this chooses maximal norms. Here we take the first $S$ rows/cols.
        let i_subset: Vec<usize> = (0..s).collect();
        let j_subset: Vec<usize> = (0..s).collect();

        // 2. Build K = A @ B[:, J]
        eprintln!(
            "Building K[m={}, s={}] | a_dim={:?}, b_dim={:?}, k={}",
            m,
            s,
            a.dim(),
            b.dim(),
            k
        );
        let mut k_mat = Array2::<f64>::zeros((m, s));
        for (j_out, &j_in) in j_subset.iter().enumerate() {
            for row in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[[row, l]] * b[[l, j_in]];
                }
                k_mat[[row, j_out]] = sum;
            }
        }

        // 3. Build R = A[I, :] @ B
        let mut r_mat = Array2::<f64>::zeros((s, n));
        for (i_out, &i_in) in i_subset.iter().enumerate() {
            for col in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[[i_in, l]] * b[[l, col]];
                }
                r_mat[[i_out, col]] = sum;
            }
        }

        // 4. Build W = R[:, J]
        let mut w_mat = Array2::<f64>::zeros((s, s));
        for (j_out, &j_in) in j_subset.iter().enumerate() {
            for row in 0..s {
                w_mat[[row, j_out]] = r_mat[[row, j_in]];
            }
        }

        // 5. Compute Pseudo-inverse of W via SVD using faer
        let mut w_faer = faer::Mat::<f64>::zeros(s, s);
        for row in 0..s {
            for col in 0..s {
                w_faer[(row, col)] = w_mat[[row, col]];
            }
        }

        let svd = w_faer.svd().unwrap();
        let s_faer = svd.S().column_vector();
        let u = svd.U();
        let v = svd.V();

        let mut s_inv = faer::Mat::<f64>::zeros(s, s);
        let s_len = s_faer.nrows();
        for i in 0..s {
            let val = if i < s_len { s_faer[i] } else { 0.0 };
            if val > 1e-12 {
                s_inv[(i, i)] = 1.0 / val;
            }
        }

        // W^{-1} = V \Sigma^{-1} U^T
        let w_inv_faer = v * s_inv * u.transpose();

        let mut w_inv = Array2::<f64>::zeros((s, s));
        for r in 0..s {
            for c in 0..s {
                w_inv[[r, c]] = w_inv_faer[(r, c)];
            }
        }

        Self {
            k_mat,
            r_mat,
            w_inv,
        }
    }

    /// Retrieve the synthesized value exactly at index $(i, j)$ using the inverse map
    #[inline(always)]
    pub fn navigate(&self, i: usize, j: usize) -> f64 {
        let s = self.w_inv.dim().0;
        let mut tmp = vec![0.0f64; s];
        for col in 0..s {
            let mut sum = 0.0;
            for l in 0..s {
                sum += self.k_mat[[i, l]] * self.w_inv[[l, col]];
            }
            tmp[col] = sum;
        }

        let mut final_sum = 0.0;
        for l in 0..s {
            final_sum += tmp[l] * self.r_mat[[l, j]];
        }
        final_sum
    }
}
