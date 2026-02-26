"""
gmem.zmask — Hierarchical Z-Mask (dirty page tracking).

Two-level bitmap that tracks which regions of the virtual address space
have been written to. This allows bulk operations to skip overlay lookups
for clean (never-written) regions.

Level 1 — Macro Mask:     1 bit per 1 TB chunk (128 bytes covers 1 PB)
Level 2 — Detail Mask:    1 bit per 4 KB page within a 1 TB chunk
                          (lazily allocated per TB chunk)

Constants (matching C implementation):
    PAGE_SIZE      = 1024 floats = 4 KB
    TB_SIZE        = 1 TB / sizeof(float) = 2⁳⁸ addresses
    PAGES_PER_TB   = TB_SIZE / PAGE_SIZE  = 2⁲⁸ pages
    MACRO_PAGES    = 1024 (covers 1 Petabyte)
"""

PAGE_SIZE = 1024                          # 4 KB in floats
TB_SIZE = (1024 * 1024 * 1024 * 1024) // 4  # 1 TB / sizeof(float)
PAGES_PER_TB = TB_SIZE // PAGE_SIZE
MACRO_PAGES = 1024                        # 1024 TB chunks = 1 PB


class ZMask:
    """
    Hierarchical dirty-page tracker.

    Uses Python sets for sparse representation instead of raw bitmaps,
    which is more memory-efficient when only a small fraction of pages
    are dirty (the common case).
    """

    __slots__ = ('_macro_dirty', '_detail_dirty')

    def __init__(self):
        self._macro_dirty: set[int] = set()          # Set of dirty TB indices
        self._detail_dirty: dict[int, set[int]] = {}  # TB → set of dirty page indices

    def mark_dirty(self, addr: int):
        """Mark the page containing `addr` as dirty."""
        tb_idx = addr // TB_SIZE
        if tb_idx >= MACRO_PAGES:
            return

        self._macro_dirty.add(tb_idx)

        if tb_idx not in self._detail_dirty:
            self._detail_dirty[tb_idx] = set()

        page_in_tb = (addr % TB_SIZE) // PAGE_SIZE
        self._detail_dirty[tb_idx].add(page_in_tb)

    def is_dirty(self, addr: int) -> bool:
        """Check if the page containing `addr` has been written to."""
        tb_idx = addr // TB_SIZE
        if tb_idx >= MACRO_PAGES:
            return True  # Out of tracked range — assume dirty

        if tb_idx not in self._macro_dirty:
            return False  # Entire TB is clean

        detail = self._detail_dirty.get(tb_idx)
        if detail is None:
            return False

        page_in_tb = (addr % TB_SIZE) // PAGE_SIZE
        return page_in_tb in detail

    def is_range_clean(self, start_addr: int, count: int) -> bool:
        """Check if an entire address range is clean (no writes)."""
        tb_start = start_addr // TB_SIZE
        tb_end = (start_addr + count) // TB_SIZE

        for tb in range(tb_start, min(tb_end + 1, MACRO_PAGES)):
            if tb in self._macro_dirty:
                return False
        return True

    def clear(self):
        """Reset all dirty tracking."""
        self._macro_dirty.clear()
        self._detail_dirty.clear()
