# GMem v2.0.0: Dependencies and Decorator System

## 📦 Dependencies

GMem v2.0.0 remains **zero-dependency**, requiring only the **Python Standard Library**.

| Module | Standard Library Dependencies | Purpose |
|---|---|---|
| `vrns`, `fourier`, `morph` | `math`, `enum` | Mathematical primitives, torus projection |
| `hashing`, `hdc` | `hashlib`, `struct` | FNV, fmix64, and bit-packing |
| `number_theory`, `sketch` | `random`, `math` | Primality testing, random projections |
| `persistence` | `os`, `struct`, `binascii` | AOF logging and snapshot IO |
| `decorator` | `functools`, `hashlib` | Wrapper logic and cache key generation |

> [!NOTE]  
> While `gmem` is dependency-free, it is designed to interoperate with high-performance libraries like **NumPy**, **PyTorch**, and **JAX** via bulk IO methods (`ctx.fetch_bulk`).

---

## 🎭 Decorator System Overview

The decorator system in `gmem.decorator` allows you to inject virtual memory into existing Python programs without modifying their core logic.

### 1. `@gmem_context`
Injects a `GMemContext` instance as a keyword argument (default `gmem=None`).
- **Benefit**: Quick access to the virtual manifold for manual reads/writes.

### 2. `@gmem_virtual_array`
Replaces a function argument with a `VirtualArray` object.
- **Benefit**: The function "sees" a standard list-like object but can access petabyte-scale indices with zero RAM cost.

### 3. `@gmem_cached`
Caches function results in the manifold based on a hash of the input arguments.
- **Benefit**: Transparently offloads expensive, deterministic computation results to the generative manifold.

### 4. `@gmem_workspace`
Provides a persistent, scoped `VirtualArray` as the first argument.
- **Benefit**: Ideal for simulations or stateful processing where the "workspace" needs to persist across calls but stay isolated.

---

## Updating & Expanding Decorators


### Potential Expansions
To fully leverage the Matrix-V SDK features, the decorator system could be expanded as follows:

1.  **Algebraic Composition**: A new `@gmem_compose` decorator that uses `ntt.py` to algebraically multiply the manifolds of two input contexts.
2.  **Sketched Bulk Fetch**: Update `VirtualArray` to support a `.sketch()` method using `sketch.py` for compressed data views.
3.  **HDC Fingerprinting**: A `@gmem_validated` decorator that uses `hdc.py` to verify that an input's manifold identity matches a required 1024-bit signature.
4.  **Morph Configuration**: Expansion of `@gmem_virtual_array` to accept Gielis parameters for morphological shape-shifting of the array's data.

---

## Example: The v2.0.0 Enhanced Manifold
Even with a simple decorator, you are now utilizing the upgraded 16-prime engine:

```python
from gmem import gmem_virtual_array

@gmem_virtual_array(seed=0x517)
def analyze_data(v_arr):
    # This now uses 16-prime RNS (~310 bits entropy) 
    # instead of the legacy 8-prime engine.
    val = v_arr[10**12] 
    print(f"Synthesized value at 1TB: {val}")
```
