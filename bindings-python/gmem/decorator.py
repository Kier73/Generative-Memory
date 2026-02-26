"""
gmem.decorator — Decorators for seamless virtual memory integration.

Apply to any Python function to transparently redirect data operations
through Generative Memory. This lets external programs use the 2⁶⁴
virtual address space without changing their internal logic.

Decorators:
    @gmem_context       — Inject a GMemContext as a keyword argument
    @gmem_virtual_array — Replace large list/array args with virtual arrays
    @gmem_cached        — Cache function results in the manifold by input hash
    @gmem_workspace     — Provide a scoped virtual workspace that auto-cleans

Usage:
    from gmem.decorator import gmem_context, gmem_virtual_array

    @gmem_context(seed=42)
    def my_function(data, *, gmem=None):
        gmem[0] = data[0]           # Write to virtual memory
        return gmem[1000000000]     # Read from 1B offset — zero RAM

    @gmem_virtual_array(seed=99)
    def process(arr):
        # 'arr' is now a VirtualArray backed by generative memory
        print(arr[10**15])          # Access petabyte offset
"""

import functools
import hashlib
import struct
from typing import Any, Callable

from gmem.core import GMemContext
from gmem.hashing import fnv1a_string, MASK64


# ─────────────────────────────────────────────────────────────────
# Virtual Array — a list-like interface backed by GMemContext
# ─────────────────────────────────────────────────────────────────

class VirtualArray:
    """
    A list-like object backed by Generative Memory.

    Supports indexing, slicing, iteration (bounded), and len().
    Reads are O(1) synthesized; writes are sparse overlay entries.

    This is the primary bridge between Python code that expects
    lists/arrays and the generative memory system.

    Usage:
        va = VirtualArray(seed=12345, size=10**18)
        va[0]                      # synthesized
        va[42] = 3.14              # sparse write
        va[100:110]                # slice → list of 10 values
        list(va.iter_range(0, 5))  # bounded iteration
    """

    __slots__ = ('ctx', 'size', '_name')

    def __init__(self, seed: int = 0, size: int = 2**64,
                 ctx: GMemContext | None = None, name: str = ""):
        """
        Args:
            seed: Master seed (ignored if ctx is provided).
            size: Logical size of the array (for len() and bounds info).
            ctx:  Existing GMemContext to wrap (overrides seed).
            name: Optional label for repr.
        """
        self.ctx = ctx if ctx is not None else GMemContext(seed)
        self.size = min(size, 2**64)
        self._name = name

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.size)
            return [self.ctx.fetch(i) for i in range(start, stop, step)]
        return self.ctx.fetch(int(key))

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.size)
            indices = range(start, stop, step)
            for i, v in zip(indices, value):
                self.ctx.write(i, float(v))
        else:
            self.ctx.write(int(key), float(value))

    def __len__(self):
        return self.size

    def __repr__(self):
        label = f" '{self._name}'" if self._name else ""
        return (f"VirtualArray{label}(seed=0x{self.ctx.seed:X}, "
                f"size={self.size:,}, writes={self.ctx.overlay_count})")

    def __contains__(self, value: float) -> bool:
        """Search for a value using interpolation search (approximate)."""
        idx = self.ctx.search(value)
        return abs(self.ctx.fetch_monotonic(idx) - value) < 1e-6

    def iter_range(self, start: int, stop: int, step: int = 1):
        """Iterate over a bounded range (yields synthesized values)."""
        for i in range(start, stop, step):
            yield self.ctx.fetch(i)

    def bulk_read(self, start: int, count: int) -> list[float]:
        """Read a contiguous block efficiently."""
        return self.ctx.fetch_bulk(start, count)

    def search(self, target: float) -> int:
        """Find the address closest to a target value."""
        return self.ctx.search(target)

    @property
    def writes(self) -> int:
        """Number of physical writes (overlay entries)."""
        return self.ctx.overlay_count


# ─────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────

def gmem_context(seed: int = 0, arg_name: str = "gmem"):
    """
    Decorator: inject a GMemContext as a keyword argument.

    The decorated function receives a ready-to-use GMemContext
    via the specified keyword argument.

    Usage:
        @gmem_context(seed=42)
        def my_func(x, y, *, gmem=None):
            gmem[0] = x + y
            return gmem[0]

        @gmem_context(seed=42, arg_name="vm")
        def other_func(data, *, vm=None):
            return vm.fetch(0)
    """
    def decorator(func: Callable) -> Callable:
        ctx = GMemContext(seed)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault(arg_name, ctx)
            return func(*args, **kwargs)

        # Expose the context for external access
        wrapper.gmem_ctx = ctx
        return wrapper
    return decorator


def gmem_virtual_array(seed: int = 0, size: int = 2**64,
                       arg_index: int = 0, arg_name: str | None = None):
    """
    Decorator: replace a function argument with a VirtualArray.

    The specified argument (by index or name) is replaced with a
    VirtualArray backed by generative memory. The original value
    is discarded.

    Usage:
        @gmem_virtual_array(seed=99, arg_index=0)
        def process(arr):
            # 'arr' is now a VirtualArray
            print(arr[0])               # synthesized
            arr[42] = 3.14              # sparse write
            return arr[0:10]            # slice

        @gmem_virtual_array(seed=99, arg_name="data")
        def analyze(data=None):
            return data.bulk_read(0, 100)
    """
    def decorator(func: Callable) -> Callable:
        va = VirtualArray(seed=seed, size=size)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if arg_name and arg_name in kwargs:
                kwargs[arg_name] = va
            elif arg_name:
                kwargs[arg_name] = va
            else:
                args = list(args)
                if arg_index < len(args):
                    args[arg_index] = va
                else:
                    args.append(va)
                args = tuple(args)
            return func(*args, **kwargs)

        wrapper.virtual_array = va
        return wrapper
    return decorator


def gmem_cached(seed: int = 0, namespace: str = ""):
    """
    Decorator: cache function results in the generative manifold.

    Uses a hash of the function arguments to derive an address,
    then stores the result in the overlay. Subsequent calls with
    the same arguments return the cached result.

    Works best for functions returning a single float. For complex
    return types, the result is hashed into an address for lookup.

    Usage:
        @gmem_cached(seed=42, namespace="math_ops")
        def expensive_calc(x: float, y: float) -> float:
            import time; time.sleep(1)  # simulate work
            return x * y + x / (y + 1)

        result1 = expensive_calc(3.14, 2.71)  # slow (computes + caches)
        result2 = expensive_calc(3.14, 2.71)  # instant (from manifold)
    """
    def decorator(func: Callable) -> Callable:
        ctx = GMemContext(seed)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build a cache key from function name + args
            key_str = f"{namespace}:{func.__name__}:{args}:{sorted(kwargs.items())}"
            addr = fnv1a_string(key_str) & MASK64

            # Check cache
            found, val = ctx._overlay.lookup(addr)
            if found:
                return val

            # Compute and cache
            result = func(*args, **kwargs)
            if isinstance(result, (int, float)):
                ctx.write(addr, float(result))
            return result

        wrapper.gmem_ctx = ctx
        wrapper.cache_clear = lambda: ctx._overlay.clear()
        return wrapper
    return decorator


def gmem_workspace(seed: int = 0, size: int = 2**64):
    """
    Decorator: provide a scoped virtual workspace.

    The decorated function receives a fresh VirtualArray as its
    first argument. This workspace is persistent across calls
    (same seed = same data).

    Usage:
        @gmem_workspace(seed=42, size=10**9)
        def simulation(ws):
            ws[0] = 1.0                 # initial condition
            for i in range(1, 1000):
                ws[i] = ws[i-1] * 0.99  # iterate
            return ws[999]
    """
    def decorator(func: Callable) -> Callable:
        ws = VirtualArray(seed=seed, size=size, name=func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(ws, *args, **kwargs)

        wrapper.workspace = ws
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────
# Convenience: Global virtual memory pool
# ─────────────────────────────────────────────────────────────────

class GMemPool:
    """
    A named pool of VirtualArrays for organizing virtual memory.

    Usage:
        pool = GMemPool(base_seed=100)
        weights = pool.get("model_weights", size=10**9)
        gradients = pool.get("gradients", size=10**9)

        weights[0] = 0.5
        gradients[0] = 0.01
    """

    def __init__(self, base_seed: int = 0):
        self._base_seed = base_seed
        self._arrays: dict[str, VirtualArray] = {}

    def get(self, name: str, size: int = 2**64) -> VirtualArray:
        """Get or create a named VirtualArray."""
        if name not in self._arrays:
            # Derive seed from base + name
            derived_seed = (self._base_seed ^ fnv1a_string(name)) & MASK64
            self._arrays[name] = VirtualArray(
                seed=derived_seed, size=size, name=name
            )
        return self._arrays[name]

    def __getitem__(self, name: str) -> VirtualArray:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._arrays

    def list_arrays(self) -> list[str]:
        return list(self._arrays.keys())

    def __repr__(self):
        return f"GMemPool(seed=0x{self._base_seed:X}, arrays={list(self._arrays.keys())})"
