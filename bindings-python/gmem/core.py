"""
gmem.core — GMemContext: the central class of Generative Memory.

This is the Python equivalent of the C `gmem_context` struct and its
associated API. A GMemContext provides:

    - A 2⁶⁴ address space of deterministic synthetic values
    - O(1) fetch and write with sparse physical overlay
    - Monotonic manifold with interpolation search
    - Mirroring (shadow contexts) and morphing (derived views)
    - Hierarchical Z-Mask for dirty page tracking
    - AOF persistence and overlay snapshot save/load
    - Virtual filesystem archetypes

The Fundamental Invariant:
    ∀ addr ∈ [0, 2⁶⁴):
        fetch(addr) = overlay[addr]           if addr ∈ overlay
                      morph(source.fetch(addr)) if morph_source attached
                      mirror.fetch(addr)        if mirror attached
                      synthesize(addr, seed)    otherwise

    where synthesize is pure, deterministic, and O(1)
"""

import threading

from gmem.hashing import MASK64
from gmem.vrns import synthesize
from gmem.overlay import Overlay
from gmem.zmask import ZMask
from gmem.monotonic import fetch_monotonic, search as monotonic_search
from gmem.morph import MorphState, MorphMode
from gmem.archetype import ArchetypeManager, Archetype
from gmem.persistence import AOFLog, save_overlay, load_overlay


class GMemContext:
    """
    Generative Memory Context.

    Provides a virtual address space of 2⁶⁴ cells (~16 Exabytes)
    with O(1) deterministic synthesis and sparse overlay writes.
    """

    __slots__ = (
        '_seed', '_overlay', '_zmask', '_lock',
        '_mirror_source', '_mirror_mode',
        '_morph', '_archetype_mgr', '_aof',
        '_high_precision',
    )

    def __init__(self, seed: int):
        """
        Create a new Generative Memory context.

        Args:
            seed: 64-bit master seed for deterministic synthesis.
        """
        self._seed = seed & MASK64
        self._overlay = Overlay()
        self._zmask = ZMask()
        self._lock = threading.Lock()

        # Mirroring
        self._mirror_source = None      # Source GMemContext or None
        self._mirror_mode = 0

        # Morphing
        self._morph = MorphState()

        # Archetype
        self._archetype_mgr = ArchetypeManager()

        # Persistence
        self._aof = AOFLog()
        self._high_precision = False

    # ── Properties ──────────────────────────────────────────────

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value & MASK64

    @property
    def overlay_count(self) -> int:
        return self._overlay.count

    # ── Core Fetch ──────────────────────────────────────────────

    def fetch(self, addr: int) -> float:
        """
        Fetch a value from the synthetic address space.

        Resolution order:
            1. Overlay (user writes)
            2. Morph source (derived transform)
            3. Mirror source (shadow)
            4. vRNS synthesis (procedural generation)

        This operation is O(1) — synthesized on demand.

        Args:
            addr: 64-bit virtual address.

        Returns:
            Float value (typically in [0, 1] for synthesized values).
        """
        addr &= MASK64

        # 1. Check overlay
        found, val = self._overlay.lookup(addr)
        if found:
            return val

        # 2. Morphing (derived manifold)
        if self._morph.active:
            src_val = self._morph.source.fetch(addr)
            return self._morph.apply(src_val)

        # 3. Mirroring (shadow context)
        if self._mirror_source is not None:
            return self._mirror_source.fetch(addr)

        # 4. Procedural synthesis via vRNS
        return synthesize(addr, self._seed)

    # ── Core Write ──────────────────────────────────────────────

    def write(self, addr: int, value: float):
        """
        Write a value to the synthetic address space.

        The value is stored in the sparse overlay and the
        corresponding Z-Mask page is marked dirty.

        Args:
            addr:  64-bit virtual address.
            value: Float value to store.
        """
        addr &= MASK64
        self._write_internal(addr, value)

        # AOF logging
        if self._aof.active:
            self._aof.log_write(addr, value)

    def _write_internal(self, addr: int, value: float):
        """Internal write (overlay + zmask, no AOF)."""
        self._overlay.insert(addr, float(value))
        self._zmask.mark_dirty(addr)

    # ── Monotonic Manifold ──────────────────────────────────────

    def fetch_monotonic(self, index: int) -> float:
        """
        Fetch from the monotonic (pre-sorted) manifold view.

        Returns a strictly increasing value for increasing indices.

        Args:
            index: Rank in the sorted manifold (0 to 2⁶⁴-1).
        """
        return fetch_monotonic(index, self._seed)

    def search(self, target: float) -> int:
        """
        Search for the address closest to a target value.

        Uses interpolation search on the monotonic manifold.
        Converges in O(log log n) iterations.

        Args:
            target: Value to search for, in [0.0, 1.0].

        Returns:
            The index closest to the target.
        """
        return monotonic_search(target, self._seed)

    # ── Bulk Fetch ──────────────────────────────────────────────

    def fetch_bulk(self, start_addr: int, count: int) -> list[float]:
        """
        Fetch a contiguous range of values.

        Optimized path: skips overlay lookup for clean pages
        (via Z-Mask) and uses direct synthesis.

        Args:
            start_addr: Starting address.
            count:      Number of floats to fetch.

        Returns:
            List of float values.
        """
        result = []
        i = 0
        while i < count:
            addr = (start_addr + i) & MASK64

            # Check if this page is clean (no writes in it)
            if not self._zmask.is_dirty(addr) and not self._morph.active and self._mirror_source is None:
                # Fast path: direct synthesis (no overlay check needed)
                result.append(synthesize(addr, self._seed))
            else:
                # Slow path: full fetch (checks overlay, morph, mirror)
                result.append(self.fetch(addr))
            i += 1

        return result

    # ── Mirroring ───────────────────────────────────────────────

    def mirror_attach(self, source: 'GMemContext', mode: int = 0):
        """
        Attach a source context for mirroring (shadow context).

        All reads on this context will delegate to the source.

        Args:
            source: The source context to mirror.
            mode:   Mirror mode (0 = identity).
        """
        if source is self:
            raise ValueError("Cannot mirror a context to itself")
        self._mirror_source = source
        self._mirror_mode = mode

    def mirror_detach(self):
        """Detach mirroring."""
        self._mirror_source = None
        self._mirror_mode = 0

    # ── Variety Morphing ────────────────────────────────────────

    def morph_attach(self, source: 'GMemContext', mode: int = 0,
                     a: float = 1.0, b: float = 0.0):
        """
        Attach a source context for variety morphing.

        This context's values become a real-time transform of the source.

        Args:
            source: Source context.
            mode:   MorphMode (0=identity, 1=linear, 2=add, 3=mul).
            a:      Scale factor (for LINEAR and MUL).
            b:      Offset (for LINEAR and ADD).
        """
        if source is self:
            raise ValueError("Cannot morph a context to itself")
        self._morph.attach(source, MorphMode(mode), a, b)

    def morph_detach(self):
        """Detach morphing."""
        self._morph.detach()

    # ── Archetypes ──────────────────────────────────────────────

    def set_archetype(self, archetype: int):
        """Set the semantic archetype (0=RAW, 1=FAT)."""
        self._archetype_mgr.archetype = Archetype(archetype)

    @property
    def archetype(self) -> ArchetypeManager:
        return self._archetype_mgr

    # ── Persistence ─────────────────────────────────────────────

    def save_overlay(self, path: str) -> int:
        """Save overlay state to a snapshot file. Returns 0 on success."""
        return save_overlay(self, path)

    def load_overlay(self, path: str) -> int:
        """Load overlay state from a snapshot file. Returns 0 on success."""
        return load_overlay(self, path)

    def persistence_attach(self, path: str, high_precision: bool = False) -> int:
        """
        Attach an AOF (Append-Only File) for persistent delta logging.

        Replays existing log on attach, then logs all future writes.

        Args:
            path: Path to .gvm_delta file.
            high_precision: If True, uses float64 (8 bytes) for writes.
                            Note: Replay logic depends on this flag matching
                            the file's original precision.

        Returns:
            0 on success.
        """
        self._high_precision = high_precision
        return self._aof.attach(self, path, high_precision=high_precision)

    def persistence_detach(self):
        """Detach AOF logging."""
        self._aof.detach()

    # ── Dunder ──────────────────────────────────────────────────

    def __repr__(self):
        return (f"GMemContext(seed=0x{self._seed:X}, "
                f"overlay={self._overlay.count} entries)")

    def __getitem__(self, addr: int) -> float:
        """Allow ctx[addr] syntax for fetch."""
        return self.fetch(addr)

    def __setitem__(self, addr: int, value: float):
        """Allow ctx[addr] = value syntax for write."""
        self.write(addr, value)

    def __len__(self) -> int:
        """Number of overlay (written) entries."""
        return self._overlay.count

    def __del__(self):
        """Cleanup on garbage collection."""
        self._aof.detach()
