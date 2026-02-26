// gielis_kernel.cu
// Generative Memory: Universal Semantic Autoencoder
// Computes the Gielis Superformula Lattice Lock over tens of thousands of Ghost
// Topologies simultaneously.

extern "C" {

// CUDA Kernel: Lattice Lock MSE Minimizer
// Computes the Gielis ghost projection for a specific (M, N1, N2, N3)
// combination against a target normalized signal, computing the Mean Squared
// Error.
//
// Launch configuration:
// Grid: (num_topologies + 255) / 256
// Block: 256
//
// target_signal: float array of length L (already standardized)
// phi_arr: float array of length L (already computed angles)
// m_grid, n1_grid, n2_grid, n3_grid: float arrays of parameters for each
// topology mse_out: output float array of length num_topologies to store the
// result L: length of the signal num_topologies: total number of combinations
// to check

__global__ void gielis_lattice_lock(const float *target_signal,
                                    const float *phi_arr, const float *m_grid,
                                    const float *n1_grid, const float *n2_grid,
                                    const float *n3_grid, float *mse_out, int L,
                                    int num_topologies) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Check bounds
  if (idx >= num_topologies) {
    return;
  }

  // Load topology parameters to registers
  float c_m = m_grid[idx];
  float c_n1 = n1_grid[idx];
  float c_n2 = n2_grid[idx];
  float c_n3 = n3_grid[idx];

  float mse = 0.0f;
  float sum_s = 0.0f;
  float sum_sq_s = 0.0f;

  // We must normalize the ghost shape first to compare topology and not
  // amplitude. However, allocating L floats per thread is impossible in
  // registers, so we compute the mean/std in one pass, then compute MSE in a
  // second pass.

  // Pass 1: Compute Mean and Variance
  for (int t = 0; t < L; ++t) {
    float p = phi_arr[t];

    float term1 = powf(fabsf(cosf(c_m * p / 4.0f)), c_n2);
    float term2 = powf(fabsf(sinf(c_m * p / 4.0f)), c_n3);

    // Handle n1 = 0 edge case to prevent NaN explosions
    float inv_n1 = (c_n1 < 1e-6f) ? 1e6f : (1.0f / c_n1);
    float R = powf(term1 + term2, -inv_n1);

    // Note: supershape_engine uses R * sin(phi) as the ghost shape projection
    float S = R * sinf(p);

    sum_s += S;
    sum_sq_s += S * S;
  }

  float mean_s = sum_s / (float)L;
  // Variance = E[X^2] - E[X]^2
  float var_s = (sum_sq_s / (float)L) - (mean_s * mean_s);
  float std_s = sqrtf(fmaxf(var_s, 0.0f)) + 1e-6f;

  // Pass 2: Compute MSE against the target normalized view
  for (int t = 0; t < L; ++t) {
    float p = phi_arr[t];

    float term1 = powf(fabsf(cosf(c_m * p / 4.0f)), c_n2);
    float term2 = powf(fabsf(sinf(c_m * p / 4.0f)), c_n3);

    float inv_n1 = (c_n1 < 1e-6f) ? 1e6f : (1.0f / c_n1);
    float R = powf(term1 + term2, -inv_n1);

    float S = R * sinf(p);

    // Normalize S
    float S_norm = (S - mean_s) / std_s;

    // Compute Squared Error
    float diff = S_norm - target_signal[t];
    mse += diff * diff;
  }

  // Write out mean squared error
  mse_out[idx] = mse / (float)L;
}

} // extern "C"
