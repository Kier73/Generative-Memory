"""
gmem.bridge — Projective Page Mapping (Phi/Psi).

Maps linear system addresses to synthetic manifold addresses while
preserving page locality. This is the compatibility layer between
traditional memory layouts and the generative manifold.

Mathematics:
    Φ(linear_addr, seed):
        page_index = linear_addr // PAGE_SIZE
        offset     = linear_addr  % PAGE_SIZE
        base = murmur_mix(page_index XOR seed)
        return (base & ~(PAGE_SIZE-1)) | offset

    Ψ_read(ctx, linear_addr)  = ctx.fetch(Φ(linear_addr, seed))
    Ψ_write(ctx, linear_addr, v) = ctx.write(Φ(linear_addr, seed), v)

The page alignment ensures that addresses within the same 4 KB page
map to contiguous manifold addresses, preserving spatial locality.
"""

from gmem.hashing import MASK64

PAGE_SIZE = 4096
PAGE_MASK = ~(PAGE_SIZE - 1) & MASK64


def phi(linear_addr: int, seed: int = 0x1337BEEF) -> int:
    """
    Projective Page Mapping: linear address → manifold address.

    Preserves intra-page offset continuity while scattering
    pages across the manifold via a hash mixer.

    Args:
        linear_addr: System linear address.
        seed: Mapping seed (default 0x1337BEEF matches C impl).

    Returns:
        Manifold address.
    """
    linear_addr &= MASK64
    page_index = linear_addr // PAGE_SIZE
    offset = linear_addr % PAGE_SIZE

    # MurmurHash3-style mixer on the page index
    base = (page_index ^ seed) & MASK64
    base = (base * 0xBF58476D1CE4E5B9) & MASK64
    base ^= (base >> 33)

    # Recombine: page-aligned base + original offset
    return ((base & PAGE_MASK) | offset) & MASK64


def psi_read(ctx, linear_addr: int, seed: int = 0x1337BEEF) -> float:
    """
    Coherence Read: fetch a value from the manifold via linear address.

    Args:
        ctx: GMemContext instance.
        linear_addr: System linear address.
        seed: Mapping seed.

    Returns:
        The synthesized or overlaid value.
    """
    m_addr = phi(linear_addr, seed)
    return ctx.fetch(m_addr)


def psi_write(ctx, linear_addr: int, value: float, seed: int = 0x1337BEEF):
    """
    Coherence Write: write a value to the manifold via linear address.

    Args:
        ctx: GMemContext instance.
        linear_addr: System linear address.
        value: Value to write.
        seed: Mapping seed.
    """
    m_addr = phi(linear_addr, seed)
    ctx.write(m_addr, value)
