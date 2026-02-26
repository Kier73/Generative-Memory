"""
gmem.manager — Multi-Tenant Hypervisor.

Manages multiple GMemContext instances indexed by seed or path.
Provides automatic tenant provisioning, quota enforcement, and
lifecycle management.

This is the Python equivalent of the C gmem_manager that acts
as a hypervisor for many independent generative address spaces.
"""

import threading


class GMemManager:
    """
    Multi-tenant manager (hypervisor) for Generative Memory contexts.

    Manages the lifecycle of multiple isolated contexts, each identified
    by a unique seed. Supports quota-limited overlays for tenant isolation.
    """

    def __init__(self):
        self._tenants: dict[int, dict] = {}  # seed → {ctx, path}
        self._lock = threading.Lock()
        self._capacity = 16

    def get_by_seed(self, seed: int, quota: int = 0):
        """
        Get or create a tenant context by seed.

        If the seed already exists, returns the existing context.
        Otherwise creates a new one.

        Args:
            seed:  64-bit generative seed.
            quota: Max overlay entries (0 = unlimited).

        Returns:
            GMemContext instance.
        """
        from gmem.core import GMemContext

        with self._lock:
            if seed in self._tenants:
                return self._tenants[seed]['ctx']

            ctx = GMemContext(seed)
            if quota > 0:
                ctx._overlay.limit = quota

            self._tenants[seed] = {
                'ctx': ctx,
                'path': f"/seed_0x{seed:X}.raw"
            }
            return ctx

    def get_by_path(self, path: str):
        """
        Look up a tenant context by its path.

        Args:
            path: Tenant path string.

        Returns:
            GMemContext or None if not found.
        """
        with self._lock:
            for tenant in self._tenants.values():
                if tenant['path'] == path:
                    return tenant['ctx']
            return None

    def list_tenants(self) -> list[str]:
        """Return a list of all active tenant paths."""
        with self._lock:
            return [t['path'] for t in self._tenants.values()]

    def remove_tenant(self, seed: int) -> bool:
        """Remove a tenant by seed. Returns True if removed."""
        with self._lock:
            if seed in self._tenants:
                del self._tenants[seed]
                return True
            return False

    def shutdown(self):
        """Destroy all tenant contexts."""
        with self._lock:
            self._tenants.clear()

    @property
    def tenant_count(self) -> int:
        return len(self._tenants)

    def __len__(self) -> int:
        return len(self._tenants)

    def __contains__(self, seed: int) -> bool:
        return seed in self._tenants
