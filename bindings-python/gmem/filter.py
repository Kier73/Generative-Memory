"""
gmem.filter — JSON and Block filters for structured manifold access.

Filters provide domain-specific access patterns over the synthetic manifold:

    JSON Filter:  Maps semantic paths (e.g. "users[0].email") to manifold
                  addresses via FNV-1a string hashing with a schema salt.

    Block Filter: Maps sector IDs to contiguous manifold regions via
                  Xi mixing, enabling block-device-style access.
"""

from gmem.hashing import fnv1a_string, xi_mix


class JSONFilter:
    """
    Structured JSON-path access to the synthetic manifold.

    Maps semantic paths like "users[0].profile.email" to unique
    manifold addresses via FNV-1a hashing with a schema-specific salt.
    """

    __slots__ = ('schema_name', '_salt')

    def __init__(self, schema_name: str = ""):
        self.schema_name = schema_name
        # Salt derived from schema name (matches C: 0xDEADC0DE * 31 + chars)
        self._salt = 0xDEADC0DE
        for ch in schema_name:
            self._salt = ((self._salt * 31) + ord(ch)) & 0xFFFFFFFFFFFFFFFF

    def resolve_path(self, path: str) -> int:
        """
        Map a semantic path to a manifold address.

        Args:
            path: JSON-style path (e.g. "users[0].profile.email").

        Returns:
            64-bit manifold address.
        """
        return fnv1a_string(path, salt=self._salt)

    def get_val(self, ctx, path: str) -> float:
        """Fetch a synthesized value by semantic path."""
        addr = self.resolve_path(path)
        return ctx.fetch(addr)

    def set_val(self, ctx, path: str, value: float):
        """Write a value by semantic path."""
        addr = self.resolve_path(path)
        ctx.write(addr, value)


class BlockFilter:
    """
    Block-device-style access to the synthetic manifold.

    Maps sector IDs to contiguous manifold regions, allowing
    the manifold to be used as a virtual block device.
    """

    __slots__ = ('sector_size',)

    def __init__(self, sector_size: int = 4096):
        self.sector_size = sector_size if sector_size > 0 else 4096

    def read_block(self, ctx, sector_id: int) -> list[float]:
        """
        Read a full sector of synthetic data from the manifold.

        Args:
            ctx:       GMemContext instance.
            sector_id: Block/sector identifier.

        Returns:
            List of floats, length = sector_size // 4.
        """
        base_d = xi_mix(sector_id, ctx.seed)
        float_count = self.sector_size // 4  # sizeof(float) = 4
        return [ctx.fetch(base_d + i) for i in range(float_count)]
