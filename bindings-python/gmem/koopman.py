# gmem/koopman.py
import numpy as np

class KoopmanSpectralSonar:
    """
    Phase 8H: Semantic Autoencoder Koopman Spectral Sonar Regression.
    Discover the macro-structure of high-dimensional data directly from its
    1D Space-Filling Hilbert topology path.
    """
    
    @staticmethod
    def scan(signal_1d: np.ndarray) -> tuple:
        """
        Executes a rigorous spectral scan.
        Takes any length signal and extracts the single dominant angular frequency (omega)
        and the 5D continuous polynomial regression tensor (beta).
        """
        L = signal_1d.shape[0]
        
        # Standardize the physical viewport to strictly [-1, 1] 
        # This keeps the Generative Memory geometry mathematically bounded.
        pts = np.linspace(-1, 1, L, dtype=np.float64)
        
        # O(N log N) Fast Fourier Transform to locate the primary geometric echo
        yf = np.fft.rfft(signal_1d)
        
        # Find index of the peak magnitude (ignoring DC offset at idx 0)
        dominant_idx = np.argmax(np.abs(yf[1:])) + 1
        
        # We scale the frequency so it fits inside the mathematical viewport defined by [-1, 1]
        sensed_freq = float(dominant_idx) / 1.0 
        omega = sensed_freq * np.pi
        
        # O(N) Projection of the Data into the Spectral Koopman Manifold
        # β_0 * x^2 + β_1 * x + β_2 * sin(ω * x) + β_3 * cos(ω * x) + β_4
        X = np.stack([
            pts**2, 
            pts, 
            np.sin(omega * pts), 
            np.cos(omega * pts), 
            np.ones_like(pts)
        ], axis=1)
        
        # O(N^3) Least Squares Solution to calculate Beta
        try:
            # lstsq returns [solution, residuals, rank, singular_values]
            beta, _, _, _ = np.linalg.lstsq(X, signal_1d, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback for perfectly silent or chaotic topologies
            beta = np.zeros(5, dtype=np.float64)
            
        return omega, beta
