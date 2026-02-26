"""
gmem.ntt — NTT Goldilocks Field Synthesis.

Algebraic law composition and recovery in the Goldilocks prime field.
Enables context multiplication (two contexts compose into a product)
and law recovery (extract one parent from a product + the other parent).

Goldilocks Prime:
    P = 2⁶⁴ - 2³² + 1 = 0xFFFFFFFF00000001

Key Properties:
    - Fits in a 64-bit register
    - Efficient Montgomery reduction
    - Supports NTT-based polynomial multiplication
    - Full multiplicative group of order P-1

Equations:
    Composition:   law_C = (seed_A × seed_B) mod P
    Resolution:    V(x,y) = (seed × x + y) mod P
    Recovery:      seed_B = product × seed_A⁻¹ mod P
"""

from gmem.hashing import MASK64, C_MAGIC

P_GOLDILOCKS = 0xFFFFFFFF00000001


def multiply_mod(a: int, b: int) -> int:
    """Multiply two values in the Goldilocks field."""
    return (a * b) % P_GOLDILOCKS


def add_mod(a: int, b: int) -> int:
    """Add two values in the Goldilocks field."""
    return (a + b) % P_GOLDILOCKS


def sub_mod(a: int, b: int) -> int:
    """Subtract in the Goldilocks field."""
    return (a - b) % P_GOLDILOCKS


def pow_mod(base: int, exp: int, mod: int = P_GOLDILOCKS) -> int:
    """Fast modular exponentiation."""
    return pow(base, exp, mod)


def mod_inverse(a: int, m: int = P_GOLDILOCKS) -> int:
    """
    Modular inverse: a⁻¹ mod m.

    Uses Extended Euclidean Algorithm (identical to gmem_trinity.c).
    """
    a = a % m
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


def synthesize_product_law(seed_a: int, seed_b: int) -> int:
    """
    Compose two context seeds into a product law.

    law_C = (seed_A × seed_B) mod P

    This IS the algebraic matmul of seeds in the Goldilocks field.
    """
    return multiply_mod(seed_a, seed_b)


def inverse_product_law(product_seed: int, parent_a_seed: int) -> int:
    """
    Recover parent_b from a product and one known parent.

    seed_B = product × seed_A⁻¹ mod P

    This is algebraic decompression — given C = A⊗B and A, recover B.
    """
    inv_a = mod_inverse(parent_a_seed)
    return multiply_mod(product_seed, inv_a)


def resolve_value_at(seed: int, x: int, y: int) -> int:
    """
    Resolve a deterministic value at coordinate (x, y).

    V(x, y) = (seed × x + y) mod P
    """
    term1 = multiply_mod(seed, x)
    return add_mod(term1, y)


def resolve_from_law_256(law_seed: list[int], input_variety: list[int],
                         output_index: int, seed: int) -> int:
    """
    Resolve a bit-exact output from a 256-bit (4×64) law seed.

    Core "Instantaneous Materialization" operation.
    """
    ls_mix = law_seed[0] ^ law_seed[1] ^ law_seed[2] ^ law_seed[3]
    iv_mix = input_variety[0] ^ input_variety[1] ^ input_variety[2] ^ input_variety[3]

    combined = synthesize_product_law(ls_mix, iv_mix)
    addr = ((combined << 16) | (output_index & 0xFFFF)) & MASK64

    return (addr * C_MAGIC + seed) & MASK64


def verify_law_roundtrip(seed_a: int, seed_b: int) -> bool:
    """
    Self-test: verify inverse_product_law(a×b, a) == b.

    Returns True if the algebraic roundtrip is exact.
    """
    product = synthesize_product_law(seed_a, seed_b)
    recovered = inverse_product_law(product, seed_a)
    return recovered == (seed_b % P_GOLDILOCKS)
