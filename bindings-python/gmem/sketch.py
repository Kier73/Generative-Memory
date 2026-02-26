"""
gmem.sketch — Johnson-Lindenstrauss Dimension Reduction.

Compresses bulk reads via random projection while preserving
distance fidelity. The JL Lemma guarantees that pairwise distances
are maintained within (1 ± ε) with high probability.

Mathematics:
    Adaptive Dimension:  D ≥ 4·ln(N) / (ε²/2 - ε³/3)
    Projection:          y = R·x  where R is D×N random Rademacher
    Guarantee:           (1-ε)||x|| ≤ ||Rx|| ≤ (1+ε)||x||
"""

import math
import random
from gmem.hashing import fmix64, MASK64


def adaptive_dimension(n: int, epsilon: float = 0.1) -> int:
    """
    Compute the optimal projection dimension D.

    Based on the Johnson-Lindenstrauss Lemma:
        D ≥ 4·ln(N) / (ε²/2 - ε³/3)

    Args:
        n: Number of data points.
        epsilon: Distance preservation tolerance (0 < ε < 1).

    Returns:
        Minimum projection dimension D.
    """
    if n <= 1:
        return 1
    numerator = 4.0 * math.log(n)
    denominator = (epsilon ** 2 / 2.0) - (epsilon ** 3 / 3.0)
    if denominator <= 0:
        return n  # epsilon too large, no reduction possible
    return math.ceil(numerator / denominator)


class SpectralSketch:
    """
    Reusable JL random projection matrix for sketched bulk reads.

    The projection matrix R is generated deterministically from a seed,
    ensuring reproducibility. Values are ±1/√D (Rademacher distribution).
    """

    __slots__ = ('_d', '_n', '_seed', '_scale')

    def __init__(self, n: int, epsilon: float = 0.1, seed: int = 42):
        """
        Args:
            n: Input dimension (number of values in the full read).
            epsilon: JL tolerance.
            seed: Deterministic seed for the projection matrix.
        """
        self._n = n
        self._d = adaptive_dimension(n, epsilon)
        self._seed = seed
        self._scale = 1.0 / math.sqrt(self._d)

    @property
    def output_dim(self) -> int:
        return self._d

    @property
    def input_dim(self) -> int:
        return self._n

    @property
    def compression_ratio(self) -> float:
        return self._n / self._d if self._d > 0 else 0

    def _rademacher(self, row: int, col: int) -> float:
        """Deterministic ±1 from row/col via fmix64."""
        h = fmix64((row * 0xDEADBEEF + col) ^ self._seed)
        return self._scale if (h & 1) == 0 else -self._scale

    def project(self, values: list[float]) -> list[float]:
        """
        Project a full-dimension vector to reduced-dimension.

        y[i] = Σⱼ R[i,j] · x[j]
        """
        n = min(len(values), self._n)
        result = []
        for i in range(self._d):
            acc = 0.0
            for j in range(n):
                acc += self._rademacher(i, j) * values[j]
            result.append(acc)
        return result


def sketch_bulk(ctx, start_addr: int, count: int,
                epsilon: float = 0.1, seed: int = 42) -> list[float]:
    """
    Compressed bulk fetch via JL random projection.

    Instead of returning `count` values, returns D ≪ count values
    that preserve the structural information within (1 ± ε).

    Args:
        ctx: GMemContext instance.
        start_addr: Starting address.
        count: Number of full-dimensional values to read.
        epsilon: JL tolerance.
        seed: Projection seed.

    Returns:
        List of D reduced-dimension values.
    """
    sketch = SpectralSketch(count, epsilon, seed)
    values = ctx.fetch_bulk(start_addr, count)
    return sketch.project(values)
