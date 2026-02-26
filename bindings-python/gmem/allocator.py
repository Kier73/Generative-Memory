"""
gmem.allocator — g_malloc / g_free style sparse allocator.

Wraps GMemContext to provide a familiar malloc-like interface.
Each allocation gets its own dedicated context (seeded by size),
providing a fully independent virtual address space.

Usage:
    buf = g_malloc(1024 * 1024)       # "allocate" 1 MB
    val = g_get_f32(buf, 42)          # read index 42
    g_free(buf)                       # release

The "allocation" is virtual — no physical memory is consumed for
unwritten indices. Only overlay writes consume real RAM.
"""

from gmem.hashing import MASK64


class _GMemAlloc:
    """Internal allocation handle wrapping a dedicated context."""
    __slots__ = ('ctx', 'size', 'magic')

    MAGIC = 0x474D454D  # "GMEM"

    def __init__(self, ctx, size: int):
        self.ctx = ctx
        self.size = size
        self.magic = self.MAGIC


# Global registry of live allocations
_allocations: dict[int, _GMemAlloc] = {}


def g_malloc(size: int):
    """
    Allocate a sparse virtual buffer of `size` bytes.

    Returns a handle (integer ID) to the allocation.
    The buffer is backed by a dedicated GMemContext.
    """
    from gmem.core import GMemContext

    seed = (size ^ 0xFEEDFACE) & MASK64
    ctx = GMemContext(seed)

    alloc = _GMemAlloc(ctx, size)
    handle = id(alloc)
    _allocations[handle] = alloc
    return handle


def g_free(handle: int):
    """Free a sparse virtual buffer by handle."""
    if handle in _allocations:
        del _allocations[handle]


def g_get_f32(handle: int, index: int) -> float:
    """
    Fetch a float from a g_malloc'd buffer by index.

    Args:
        handle: Handle returned by g_malloc.
        index:  Element index in the virtual buffer.

    Returns:
        Synthesized or overlaid float value.
    """
    alloc = _allocations.get(handle)
    if alloc is None or alloc.magic != _GMemAlloc.MAGIC:
        return 0.0
    return alloc.ctx.fetch(index)


def g_set_f32(handle: int, index: int, value: float):
    """
    Write a float to a g_malloc'd buffer by index.

    Args:
        handle: Handle returned by g_malloc.
        index:  Element index in the virtual buffer.
        value:  Float value to write.
    """
    alloc = _allocations.get(handle)
    if alloc is None or alloc.magic != _GMemAlloc.MAGIC:
        return
    alloc.ctx.write(index, value)


def g_get_ctx(handle: int):
    """Get the raw GMemContext for an allocation (advanced use)."""
    alloc = _allocations.get(handle)
    if alloc and alloc.magic == _GMemAlloc.MAGIC:
        return alloc.ctx
    return None
