use std::f64::consts::PI;
use rustfft::{FftPlanner, num_complex::Complex};
use nalgebra::{DMatrix, DVector};

/// Represents the continuous mathematical regression found by the Spectral Sonar.
#[derive(Debug, Clone)]
pub struct KoopmanState {
    pub omega: f64,
    pub beta: [f64; 5],
}

/// Discovers the macro-structure of high-dimensional data directly from its
/// 1D space-filling Hilbert topological path, natively in Rust.
pub fn koopman_sonar(signal_1d: &[f64]) -> KoopmanState {
    let len = signal_1d.len();
    if len == 0 {
        return KoopmanState { omega: 0.0, beta: [0.0; 5] };
    }

    // Step 1: O(N log N) Fast Fourier Transform to locate dominant angular frequency
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = signal_1d.iter()
        .map(|&val| Complex { re: val, im: 0.0 })
        .collect();
    
    fft.process(&mut buffer);

    // Find the peak magnitude, ignoring the DC offset at index 0
    let mut max_mag = 0.0;
    let mut peak_idx = 1;
    
    // We only need to check up to len/2 for real signals, but scanning all is fine
    let search_len = len / 2;
    for i in 1..=search_len {
        let mag = buffer[i].norm();
        if mag > max_mag {
            max_mag = mag;
            peak_idx = i;
        }
    }

    // Scale frequency inside the [-1, 1] viewport
    let sensed_freq = peak_idx as f64;
    let omega = sensed_freq * PI;

    // Step 2: O(N) Projection into 5D Koopman Manifold
    // β_0 * x^2 + β_1 * x + β_2 * sin(ω * x) + β_3 * cos(ω * x) + β_4
    let mut x_mat = DMatrix::<f64>::zeros(len, 5);
    let mut y_vec = DVector::<f64>::zeros(len);

    for i in 0..len {
        // Map [0, len-1] -> [-1, 1]
        let pt = -1.0 + 2.0 * (i as f64) / ((len - 1) as f64).max(1.0);
        
        y_vec[i] = signal_1d[i];
        
        x_mat[(i, 0)] = pt * pt;
        x_mat[(i, 1)] = pt;
        x_mat[(i, 2)] = (omega * pt).sin();
        x_mat[(i, 3)] = (omega * pt).cos();
        x_mat[(i, 4)] = 1.0;
    }

    // Step 3: Least Squares Solver for Beta extraction
    // Uses SVD (Singular Value Decomposition) to solve overdetermined systems cleanly
    let svd = x_mat.svd(true, true);
    let beta_vec = svd.solve(&y_vec, 1e-10).unwrap_or_else(|_| DVector::zeros(5));

    KoopmanState {
        omega,
        beta: [beta_vec[0], beta_vec[1], beta_vec[2], beta_vec[3], beta_vec[4]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_koopman_sonar() {
        // Create a simple sine wave + noise test vector
        let length = 1024;
        let mut signal = vec![0.0; length];
        let target_freq = 5.0; // Expected peak_idx = 5
        let target_omega = target_freq * PI;
        
        for i in 0..length {
            let pt = -1.0 + 2.0 * (i as f64) / ((length - 1) as f64);
            // beta_2 = 1.0, beta_4 = 0.5
            signal[i] = (target_omega * pt).sin() + 0.5; 
        }

        let state = koopman_sonar(&signal);
        
        // Assert we caught the frequency
        assert!((state.omega - target_omega).abs() < 1e-4);
        
        // Assert beta_2 is near 1.0, and beta_4 is near 0.5
        assert!((state.beta[2] - 1.0).abs() < 1e-4, "Failed to regress Beta_2 (sin)");
        assert!((state.beta[4] - 0.5).abs() < 1e-4, "Failed to regress Beta_4 (bias)");
        
        // Assert others are 0
        assert!(state.beta[0].abs() < 1e-4);
        assert!(state.beta[1].abs() < 1e-4);
        assert!(state.beta[3].abs() < 1e-4);
    }
}
