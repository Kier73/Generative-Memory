"""
gmem.hashing — All hash and mixing functions used by Generative Memory.

Primitives:
    fnv1a_u64       — FNV-1a hash of a 64-bit integer (overlay addressing)
    fnv1a_string    — FNV-1a hash of a UTF-8 string (filter paths)
    djb2            — DJB2 string hash (Trinity signatures)
    feistel_shuffle — 4-round Feistel network (deterministic permutation)
    crc32           — CRC32 with polynomial 0xEDB88320 (integrity checks)
    xi_mix          — MurmurHash3-style mixer (sector→manifold mapping)
"""

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
MASK32 = 0xFFFF_FFFF

# FNV-1a constants (64-bit)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def fnv1a_u64(addr: int) -> int:
    """FNV-1a hash of a single 64-bit integer."""
    h = FNV_OFFSET
    h = ((h ^ (addr & MASK64)) * FNV_PRIME) & MASK64
    return h


def fnv1a_string(s: str, salt: int = 0) -> int:
    """FNV-1a hash of a UTF-8 string with optional salt."""
    h = (FNV_OFFSET ^ (salt & MASK64)) & MASK64
    for ch in s:
        h = ((h ^ ord(ch)) * FNV_PRIME) & MASK64
    return h


def djb2(s: str) -> int:
    """DJB2 hash for string signatures (Trinity intention/law hashing)."""
    h = 5381
    for ch in s:
        h = (((h << 5) + h) + ord(ch)) & MASK64
    return h


def feistel_shuffle(val: int, seed: int) -> int:
    """
    4-round Feistel network on a 64-bit value.

    Splits val into two 32-bit halves (L, R) and applies:
        R' = L XOR (R * φ + seed_low32)  mod 2³²
        L' = R
    where φ = 0x9E3779B9 (golden ratio constant).
    """
    l = (val >> 32) & MASK32
    r = val & MASK32
    s = seed & MASK32
    phi = 0x9E3779B9

    for _ in range(4):
        temp = r
        r = (l ^ ((r * phi + s) & MASK32)) & MASK32
        l = temp

    return ((l << 32) | r) & MASK64


def crc32(data: bytes) -> int:
    """CRC32 with polynomial 0xEDB88320 (reflected), matching the C implementation."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return (crc ^ 0xFFFFFFFF) & MASK32


def xi_mix(sector_id: int, seed: int) -> int:
    """
    MurmurHash3-style mixer (Xi mapping).

    Maps a sector ID into the synthetic manifold address space.
    Used by the block filter for sector→manifold projection.

    ξ(sector_id, seed):
        d = sector_id XOR seed
        d *= 0xBF58476D1CE4E5B9
        d ^= d >> 33
        d *= 0x94D049BB133111EB
        d ^= d >> 33
    """
    d = (sector_id ^ seed) & MASK64
    d = (d * 0xBF58476D1CE4E5B9) & MASK64
    d ^= (d >> 33)
    d = (d * 0x94D049BB133111EB) & MASK64
    d ^= (d >> 33)
    return d


# ── MurmurMix64 Finalizer ──────────────────────────────────────

def fmix64(h: int) -> int:
    """
    MurmurHash3 64-bit finalizer (avalanche mixer).

    Collapses a high-entropy manifold into a single well-distributed
    64-bit value. Used for multi-channel RNS resolution and HDC seed
    generation.
    """
    h &= MASK64
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & MASK64
    h ^= h >> 33
    return h


# ── Invertible Variety Hash ────────────────────────────────────

# Spectral anchor constants (golden ratio fractional part)
C_MAGIC = 0x9E3779B97F4A7C15
C2 = 0xBF58476D1CE4E5B9
C3 = 0x94D049BB133111EB

# Modular multiplicative inverses (C × C⁻¹ ≡ 1 mod 2⁶⁴)
INV_C2 = 0x96DE1B173F119089
INV_C3 = 0x319642B2D24D8EC3


def vl_mask(addr: int, seed: int) -> int:
    """
    Invertible Feistel-based variety generator.

    Maps (addr, seed) → 64-bit hash with full avalanche effect.
    Unlike FNV-1a, this is a BIJECTION — every output maps to
    exactly one input, recoverable via vl_inverse_mask().

    z = (addr + seed + φ)
    z ← (z ⊕ (z >> 30)) × C₂   mod 2⁶⁴
    z ← (z ⊕ (z >> 27)) × C₃   mod 2⁶⁴
    z ← z ⊕ (z >> 31)
    """
    z = (addr + seed + C_MAGIC) & MASK64
    z = ((z ^ (z >> 30)) * C2) & MASK64
    z = ((z ^ (z >> 27)) * C3) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _invert_shift_xor(val: int, k: int) -> int:
    """Invert the z ^ (z >> k) transformation."""
    res = val
    v = val
    while True:
        v >>= k
        if v == 0:
            break
        res ^= v
    return res & MASK64


def vl_inverse_mask(z: int, seed: int) -> int:
    """
    Exact inverse of vl_mask: recover the input address from the hash output.

    Given h = vl_mask(addr, seed), then vl_inverse_mask(h, seed) == addr.
    This enables O(1) reverse address lookup — "spectral resonance collapse".
    """
    # Undo z ^ (z >> 31)
    z = _invert_shift_xor(z, 31)
    # Undo multiplication by C3
    z = (z * INV_C3) & MASK64
    # Undo z ^ (z >> 27)
    z = _invert_shift_xor(z, 27)
    # Undo multiplication by C2
    z = (z * INV_C2) & MASK64
    # Undo z ^ (z >> 30)
    z = _invert_shift_xor(z, 30)
    # Undo addition
    return (z - seed - C_MAGIC) & MASK64
