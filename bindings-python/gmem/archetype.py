"""
gmem.archetype — Virtual filesystem archetypes (semantic overlays).

Archetypes define how the synthetic manifold presents itself structurally.
The FAT archetype makes the manifold appear as a filesystem with directories
and files, with dynamic file creation backed by the overlay.

Archetype Modes:
    RAW (0) — Raw manifold access (no structuring)
    FAT (1) — Virtual FAT-like filesystem structure
"""

from enum import IntEnum
from gmem.hashing import fnv1a_string, MASK64


class Archetype(IntEnum):
    """Semantic archetype modes."""
    RAW = 0
    FAT = 1


class VirtEntry:
    """A virtual filesystem entry (file or directory)."""
    __slots__ = ('name', 'offset', 'size', 'is_dir')

    def __init__(self, name: str = "", offset: int = 0,
                 size: int = 0, is_dir: bool = False):
        self.name = name
        self.offset = offset
        self.size = size
        self.is_dir = is_dir

    def __repr__(self):
        kind = "DIR" if self.is_dir else "FILE"
        return f"VirtEntry({kind}: {self.name!r}, offset=0x{self.offset:X}, size={self.size})"


class DynFile:
    """A dynamically created file in the virtual filesystem."""
    __slots__ = ('path', 'offset', 'size', 'is_dir')

    def __init__(self, path: str, is_dir: bool = False):
        self.path = path
        self.is_dir = is_dir
        # Assign virtual offset in "High Memory" area
        self.offset = 0xF000000000000000 | (fnv1a_string(path) & 0xFFFFFFFFFF)
        self.size = 0


class ArchetypeManager:
    """
    Manages the virtual filesystem archetype for a context.

    Provides static procedural entries (FAT layout) and
    dynamic overlay entries created at runtime.
    """

    __slots__ = ('_archetype', '_dynamic_files')

    def __init__(self, archetype: Archetype = Archetype.RAW):
        self._archetype = archetype
        self._dynamic_files: list[DynFile] = []

    @property
    def archetype(self) -> Archetype:
        return self._archetype

    @archetype.setter
    def archetype(self, value: Archetype):
        self._archetype = Archetype(value)

    def create_file(self, path: str, is_dir: bool = False) -> int:
        """
        Create a dynamic virtual file or directory.

        Returns 0 on success.
        """
        # Check if already exists
        for f in self._dynamic_files:
            if f.path == path:
                return 0  # Already exists
        self._dynamic_files.append(DynFile(path, is_dir))
        return 0

    def get_entries(self, path: str, max_entries: int = 100) -> list[VirtEntry]:
        """
        List virtual entries at a given path.

        Combines static procedural entries with dynamic overlay entries.
        """
        entries: list[VirtEntry] = []

        if self._archetype == Archetype.RAW:
            return entries

        # Static procedural entries (FAT mode)
        if self._archetype == Archetype.FAT:
            if path == "/":
                entries.append(VirtEntry("README.txt", 0, 1024, False))
                entries.append(VirtEntry("data_volume.bin", 1024 * 1024,
                                         1024 * 1024 * 1024 * 1024, False))
                entries.append(VirtEntry("assets", 2048, 0, True))
            elif path == "/assets":
                for i in range(5):
                    entries.append(VirtEntry(
                        f"asset_{i:03d}.raw",
                        (i + 1) * 1024 * 1024 * 1024,
                        64 * 1024 * 1024,
                        False
                    ))

        # Dynamic entries (overlay)
        for df in self._dynamic_files:
            if len(entries) >= max_entries:
                break
            # Check if this file's parent matches the requested path
            parent = _get_parent(df.path)
            if parent == path:
                basename = _get_basename(df.path)
                entries.append(VirtEntry(basename, df.offset, df.size, df.is_dir))

        return entries[:max_entries]


def _get_basename(path: str) -> str:
    """Get the filename from a path."""
    idx = path.rfind('/')
    return path[idx + 1:] if idx >= 0 else path


def _get_parent(path: str) -> str:
    """Get the parent directory from a path."""
    idx = path.rfind('/')
    if idx > 0:
        return path[:idx]
    elif idx == 0:
        return "/"
    return "."
