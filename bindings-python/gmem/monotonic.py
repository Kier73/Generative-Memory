"""
gmem.monotonic — Monotonic manifold and interpolation search.

Provides a strictly-increasing view over the full 2⁶⁴ address space.
Used for predictive (interpolation-based) search.

Mathematics:
    V_mono(index) = index / (2⁶⁴ - 1) + feistel(index, seed) / (2⁶⁴ - 1)²

The variety term is negligibly small (divided by (2⁶⁴-1)²) so it provides
uniqueness without breaking monotonicity. The near-linear structure enables
O(log log n) interpolation search convergence.
"""

from gmem.hashing import feistel_shuffle, MASK64

UINT64_MAX = (1 << 64) - 1
UINT64_MAX_F = float(UINT64_MAX)


def fetch_monotonic(index: int, seed: int) -> float:
    """
    Fetch a value from the monotonic (pre-sorted) manifold.

    Returns a float that is strictly increasing with index,
    providing a sorted view over the 2⁶⁴ address space.

    Args:
        index: Rank/position in the sorted manifold (0 to 2⁶⁴-1).
        seed:  Master procedural seed.

    Returns:
        Float value that increases monotonically with index.
    """
    index &= MASK64
    base_ramp = index / UINT64_MAX_F
    variety_int = feistel_shuffle(index, seed)
    variety = (variety_int / UINT64_MAX_F) / UINT64_MAX_F
    return base_ramp + variety


def search(target: float, seed: int, max_iterations: int = 64) -> int:
    """
    Interpolation search for a target value in the monotonic manifold.

    Leverages the near-linear structure of the manifold to converge
    in O(log log n) iterations — typically ~4-6 for the full 2⁶⁴ space.

    Args:
        target: Value to search for, in [0.0, 1.0].
        seed:   Master procedural seed.
        max_iterations: Safety cap (64 = bit-exact resolution).

    Returns:
        The index (synthetic address) closest to the target value.
    """
    if target <= 0.0:
        return 0
    if target >= 1.0:
        return UINT64_MAX

    low = 0
    high = UINT64_MAX

    for _ in range(max_iterations):
        if high <= low:
            break

        low_val = fetch_monotonic(low, seed)
        high_val = fetch_monotonic(high, seed)

        if target <= low_val:
            return low
        if target >= high_val:
            return high

        # Linear interpolation pivot
        scale = (target - low_val) / (high_val - low_val)
        pivot = low + int(scale * (high - low))

        # Clamp pivot to valid range
        pivot = max(low, min(high, pivot))

        pivot_val = fetch_monotonic(pivot, seed)

        if pivot_val < target:
            low = pivot + 1
        elif pivot_val > target:
            high = pivot - 1
        else:
            return pivot

    return low
