"""
gmem.trinity — Trinity Synergy: Hilbert Curve, CRT, and Inductive Sort.

Provides 2D spatial addressing via Hilbert curves and 128-bit resolution
via Chinese Remainder Theorem (CRT) reconstruction using Goldilocks primes.

Key Components:
    1. Hilbert Curve (2D → 1D) — locality-preserving spatial mapping
    2. Inductive Resolve — monotonic sorted value at a 2D coordinate
    3. Trinity Solve — 128-bit residue reconstruction via CRT

Goldilocks Primes (3 industrial-grade primes):
    M₁ = 0xFFFFFFFF00000001  (Goldilocks prime)
    M₂ = 0xFFFFFFFF7FFFFFFF  (Safe prime)
    M₃ = 0xFFFFFFFFFFFFFFC5  (Ultra prime)
"""

from gmem.hashing import djb2, MASK64

# Goldilocks primes
MOD_GOLDILOCKS = 0xFFFFFFFF00000001
MOD_SAFE = 0xFFFFFFFF7FFFFFFF
MOD_ULTRA = 0xFFFFFFFFFFFFFFC5
TRINITY_MODULI = (MOD_GOLDILOCKS, MOD_SAFE, MOD_ULTRA)


def hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    """
    Map 2D coordinates (x, y) to a 1D Hilbert curve index.

    The Hilbert curve provides excellent spatial locality —
    nearby 2D points map to nearby 1D indices.

    Args:
        n: Grid size (must be a power of 2).
        x: X coordinate.
        y: Y coordinate.

    Returns:
        1D Hilbert index d.
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)

        # Rotate/Flip
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return d


def _mod_inverse(a: int, m: int) -> int:
    """
    Extended Euclidean Algorithm for modular inverse.

    Returns x such that (a * x) ≡ 1 (mod m).
    Uses Python's arbitrary precision integers (no __int128 needed).
    """
    m0 = m
    x0, x1 = 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        a, m = m, a % m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1


def inductive_resolve_sorted(ctx, x: int, y: int, n: int) -> float:
    """
    Resolve a 2D coordinate to a monotonically sorted float in [0, 1].

    Uses the Hilbert curve for locality-preserving 2D→1D mapping,
    then applies a base ramp with seed-based variety.

    Args:
        ctx: GMemContext (needs .seed attribute).
        x:   X coordinate in the grid.
        y:   Y coordinate in the grid.
        n:   Grid size (power of 2).

    Returns:
        Float in [0, 1] that increases monotonically along the Hilbert curve.
    """
    d = hilbert_xy_to_d(n, x, y)
    total_size = float(n) * float(n)
    base_ramp = d / total_size

    # Subtle variety based on seed (maintains strict monotonicity)
    variety_seed = (d ^ ctx.seed) & MASK64
    variety_seed = (variety_seed * 0x517cc1b727220a95) & MASK64
    variety_seed ^= (variety_seed >> 31)
    variety = (variety_seed / 18446744073709551615.0) / total_size

    return float(base_ramp + variety)


def trinity_solve_rns(ctx, intention: str, law: str,
                      x: int, y: int, event_sig: int) -> int:
    """
    Solve a synthetic manifold coordinate using Trinity Synergy.

    Combines intention, law, and event signatures with the context seed,
    generates residues under 3 Goldilocks primes, and reconstructs
    via the Chinese Remainder Theorem.

    The result is a 128-bit (or larger) integer — Python handles this
    natively with arbitrary-precision integers.

    Args:
        ctx:       GMemContext (needs .seed attribute).
        intention: Intention string (hashed via DJB2).
        law:       Law string (hashed via DJB2).
        x:         X coordinate.
        y:         Y coordinate.
        event_sig: Event signature (64-bit integer).

    Returns:
        Bit-exact residue reconstruction (arbitrary precision int).
    """
    i_sig = djb2(intention)
    l_sig = djb2(law)
    result_sig = (i_sig ^ l_sig ^ event_sig ^ ctx.seed) & MASK64

    moduli = TRINITY_MODULI

    # 1. Generate residues
    residues = []
    for m in moduli:
        mix = result_sig
        mix ^= (x ^ (y << 13)) & MASK64
        mix = (mix * m) & MASK64
        residues.append(mix % m)

    # 2. Chinese Remainder Theorem reconstruction
    m_prod = moduli[0] * moduli[1] * moduli[2]
    total = 0

    for i in range(3):
        mi_partial = m_prod // moduli[i]
        yi = _mod_inverse(mi_partial % moduli[i], moduli[i])
        total = (total + residues[i] * mi_partial * yi) % m_prod

    return total
