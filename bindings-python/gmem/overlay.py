"""
gmem.overlay — Sparse physical overlay (write-through cache).

In the C implementation this is an open-addressing hash map with linear probing.
In Python, we use a native dict for the fast path while preserving the same
semantics: only written addresses consume memory.

The overlay sits *in front* of the procedural synthesis engine:
    fetch(addr) = overlay[addr]  if addr in overlay
                  synthesize(addr, seed)  otherwise

This class also tracks a quota limit (max entries) for multi-tenant isolation.
"""

import threading


class Overlay:
    """
    Sparse overlay backed by a Python dict.

    Thread-safe via a reentrant lock. Supports optional quota enforcement.
    """

    __slots__ = ('_data', '_lock', '_limit')

    def __init__(self, limit: int = 0):
        """
        Args:
            limit: Maximum overlay entries (0 = unlimited).
        """
        self._data: dict[int, float] = {}
        self._lock = threading.Lock()
        self._limit = limit

    # --- Core Operations ---

    def insert(self, addr: int, value: float) -> bool:
        """
        Insert or update a value in the overlay.

        Returns True if the write succeeded, False if quota was exceeded
        and the address was not an existing entry.
        """
        with self._lock:
            if self._limit > 0 and len(self._data) >= self._limit:
                if addr not in self._data:
                    return False  # Quota exceeded
            self._data[addr] = value
            return True

    def lookup(self, addr: int) -> tuple[bool, float]:
        """
        Look up an address in the overlay.

        Returns (found, value). If not found, value is 0.0.
        """
        with self._lock:
            if addr in self._data:
                return True, self._data[addr]
            return False, 0.0

    def __contains__(self, addr: int) -> bool:
        return addr in self._data

    def __len__(self) -> int:
        return len(self._data)

    # --- Bulk Access ---

    def items(self):
        """Iterate over (addr, value) pairs (snapshot under lock)."""
        with self._lock:
            return list(self._data.items())

    def clear(self):
        """Remove all overlay entries."""
        with self._lock:
            self._data.clear()

    # --- Properties ---

    @property
    def count(self) -> int:
        return len(self._data)

    @property
    def limit(self) -> int:
        return self._limit

    @limit.setter
    def limit(self, value: int):
        self._limit = max(0, value)

    # --- Merge (for loading) ---

    def merge(self, entries: dict[int, float]):
        """Bulk-merge entries into the overlay (used during load)."""
        with self._lock:
            self._data.update(entries)
