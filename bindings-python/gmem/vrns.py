"""
gmem.vrns — Virtual Residue Number System (vRNS) synthesis engine.

UPGRADED: 16-prime RNS with precomputed CRT and multi-channel fmix64.

Two synthesis modes:
    1. Legacy Torus Projection (8 small primes, backward compatible)
    2. Multi-Channel fmix64 (16 large primes, higher entropy)

Mathematics:
    Legacy:  V = frac(Σ rᵢ/pᵢ)
    Enhanced: V = (⊕ᵢ fmix64((rᵢ+idx)%pᵢ | pᵢ<<16 | i<<32)) / (2⁶⁴-1)
    CRT:     X = [Σ rᵢ · Mᵢ · yᵢ] mod M

16-prime pool near 2¹⁶ → ~310 bits dynamic range.
"""

from gmem.hashing import MASK64, fmix64

# ── Legacy 8 small-prime moduli (backward compatible) ──────────

MODULI_LEGACY = (251, 257, 263, 269, 271, 277, 281, 283)
_RECIP_LEGACY = tuple(1.0 / m for m in MODULI_LEGACY)

# ── Extended 16-prime pool near 2¹⁶ ───────────────────────────

MODULI_16 = (
    65447, 65449, 65479, 65497, 65519, 65521, 65437, 65423,
    65419, 65413, 65407, 65393, 65381, 65371, 65357, 65353,
)

# Default: use extended primes
MODULI = MODULI_16

# ── Precomputed CRT Weights ───────────────────────────────────

def _mod_inverse(a: int, m: int) -> int:
    """Extended Euclidean Algorithm for modular inverse."""
    m0 = m
    x0, x1 = 0, 1
    if m == 1:
        return 0
    a = a % m
    while a > 1:
        q = a // m
        a, m = m, a % m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1


class CRTEngine:
    """
    Precomputed Chinese Remainder Theorem engine.

    Computes Mᵢ = M/pᵢ and yᵢ = Mᵢ⁻¹ mod pᵢ once at init time,
    then reconstruction is O(N) multiply-accumulate with no inverse.
    """
    __slots__ = ('primes', 'M', '_mi', '_yi', '_n')

    def __init__(self, primes: tuple[int, ...] = MODULI_16):
        self.primes = primes
        self._n = len(primes)
        self.M = 1
        for p in primes:
            self.M *= p
        # Precompute weights
        self._mi = []
        self._yi = []
        for p in primes:
            mi = self.M // p
            self._mi.append(mi)
            self._yi.append(_mod_inverse(mi, p))

    def reconstruct(self, residues: tuple[int, ...] | list[int]) -> int:
        """
        CRT reconstruction: residues → original integer (mod M).

        X = [Σ rᵢ · Mᵢ · yᵢ] mod M

        Uses precomputed weights → no inverse computation at call time.
        """
        total = 0
        for r, mi, yi in zip(residues, self._mi, self._yi):
            total += r * mi * yi
        return total % self.M

    def decompose(self, n: int) -> tuple[int, ...]:
        """Project an integer into residue space."""
        return tuple(n % p for p in self.primes)

    def roundtrip(self, n: int) -> bool:
        """Verify decompose → reconstruct identity."""
        return self.reconstruct(self.decompose(n % self.M)) == (n % self.M)


# Global precomputed CRT engine
_CRT_16 = CRTEngine(MODULI_16)
_CRT_LEGACY = CRTEngine(MODULI_LEGACY)


# ── Legacy Synthesis (backward compatible) ─────────────────────

def vrns_project(addr: int, seed: int) -> tuple[int, ...]:
    """Project a 64-bit address into residue channels (legacy 8-prime)."""
    x = (addr ^ seed) & MASK64
    return tuple(x % m for m in MODULI_LEGACY)


def vrns_to_float(residues: tuple[int, ...] | list[int]) -> float:
    """Torus Projection: 8 residues → normalized float in [0, 1)."""
    acc = sum(r * rec for r, rec in zip(residues, _RECIP_LEGACY))
    return acc - int(acc)


def synthesize(addr: int, seed: int) -> float:
    """One-shot legacy synthesis: address + seed → float in [0, 1)."""
    return vrns_to_float(vrns_project(addr, seed))


# ── Enhanced 16-Prime Synthesis ────────────────────────────────

def vrns_project_16(addr: int, seed: int) -> tuple[int, ...]:
    """Project into 16 residue channels (extended prime set)."""
    x = (addr ^ seed) & MASK64
    return tuple(x % m for m in MODULI_16)


def synthesize_multichannel(addr: int, seed: int) -> float:
    """
    Multi-channel fmix64 synthesis (enhanced).

    Each of 16 residue channels is independently mixed through
    MurmurMix64, then XOR-folded into a single hash. This prevents
    the correlated cancellations possible with simple torus summation.

    h = ⊕ᵢ fmix64((rᵢ + idx) % pᵢ | pᵢ << 16 | i << 32)
    V = h / (2⁶⁴ - 1)
    """
    x = (addr ^ seed) & MASK64
    h = 0
    for i, p in enumerate(MODULI_16):
        residue = x % p
        channel = (residue | (p << 16) | (i << 32)) & MASK64
        h ^= fmix64(channel)
    return (h & MASK64) / 18446744073709551615.0


def crt_reconstruct(residues: tuple[int, ...] | list[int],
                    extended: bool = True) -> int:
    """
    CRT reconstruction — residues → exact integer.

    Args:
        residues: Tuple of residues.
        extended: True for 16-prime, False for legacy 8-prime.
    """
    engine = _CRT_16 if extended else _CRT_LEGACY
    return engine.reconstruct(residues)
