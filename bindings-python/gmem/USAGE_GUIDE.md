# Generative Memory — Python Usage Guide

> How to import, connect, and use Generative Memory in **any** Python program.

---

## 1. Installation (Zero Dependencies)

The `gmem` package is pure Python — no pip install, no compilation, no dependencies.

```python
# Option A: Copy the gmem/ folder into your project
your_project/
├── gmem/           # ← drop here
├── your_script.py
└── ...

# Option B: Add the parent directory to sys.path
import sys
sys.path.insert(0, r"C:\Users\kross\Downloads\PB")

# Then import normally
from gmem import GMemContext
```

---

## 2. Quick Start — 3 Lines to Virtual Memory

```python
from gmem import GMemContext

ctx = GMemContext(seed=42)
print(ctx[0])          # Synthesized value — zero RAM consumed
print(ctx[10**15])     # 1 Petabyte offset — still zero RAM
ctx[100] = 3.14        # Only written addresses cost memory
```

**The Rule:** Reads are free. Writes are sparse. The address space is 2⁶⁴ (~16 Exabytes).

---

## 3. VirtualArray — Drop-in List Replacement

For code that expects a list or array, `VirtualArray` provides the same `[]` interface backed by generative memory:

```python
from gmem import VirtualArray

# "Allocate" a 1 Terabyte array (costs ~0 bytes)
data = VirtualArray(seed=12345, size=10**12)

# Use exactly like a list
data[0] = 1.0
data[999_999_999_999] = 2.0
print(data[500_000_000_000])    # Synthesized — never stored

# Slicing works
chunk = data[0:10]              # Returns list of 10 values

# Bulk read
block = data.bulk_read(0, 1000) # Efficient contiguous read

# Search
idx = data.search(0.5)         # Find address nearest to 0.5
```

---

## 4. Decorators — Seamless Integration

### `@gmem_context` — Inject Virtual Memory Into Any Function

```python
from gmem.decorator import gmem_context

@gmem_context(seed=42)
def process_data(input_val, *, gmem=None):
    """gmem is automatically injected as a GMemContext."""
    gmem[0] = input_val * 2
    gmem[1] = input_val ** 0.5

    # Read back (or from any address in 16 EB)
    return gmem[0] + gmem[1]

result = process_data(100.0)

# Access the context externally
print(process_data.gmem_ctx.overlay_count)
```

### `@gmem_virtual_array` — Replace an Argument With Virtual Memory

```python
from gmem.decorator import gmem_virtual_array

@gmem_virtual_array(seed=99, size=10**9)
def analyze(data):
    """The first argument is replaced with a VirtualArray."""
    # Write some values
    for i in range(100):
        data[i] = i * 0.01

    # Read across the full billion-element space
    return data[0], data[50], data[999_999_999]

start, mid, end = analyze(None)  # arg is replaced automatically
```

### `@gmem_cached` — Cache Results in the Manifold

```python
from gmem.decorator import gmem_cached

@gmem_cached(seed=42, namespace="physics")
def expensive_simulation(x: float, y: float) -> float:
    """Result is cached in the manifold by argument hash."""
    import time
    time.sleep(2)  # simulate heavy computation
    return x ** 2 + y ** 2

r1 = expensive_simulation(3.0, 4.0)  # Takes 2 seconds
r2 = expensive_simulation(3.0, 4.0)  # Instant (from manifold)

expensive_simulation.cache_clear()    # Reset the cache
```

### `@gmem_workspace` — Scoped Working Memory

```python
from gmem.decorator import gmem_workspace

@gmem_workspace(seed=42, size=10**6)
def simulation(ws):
    """ws is a persistent VirtualArray workspace."""
    ws[0] = 1.0  # initial condition
    for i in range(1, 1000):
        prev = ws[i - 1]
        ws[i] = prev * 0.999 + ws[i] * 0.001  # blend with synth noise
    return ws[999]

result = simulation()
print(simulation.workspace)  # VirtualArray 'simulation'(...)
```

---

## 5. GMemPool — Named Memory Regions

Organize multiple virtual arrays under named labels:

```python
from gmem import GMemPool

pool = GMemPool(base_seed=100)

# Get or create named regions
weights   = pool.get("model_weights", size=10**9)
gradients = pool.get("gradients", size=10**9)
activations = pool["activations"]  # shorthand, default size

# Use independently
weights[0] = 0.5
gradients[0] = 0.01

print(pool)
# GMemPool(seed=0x64, arrays=['model_weights', 'gradients', 'activations'])
```

---

## 6. Integration Patterns

### Pattern A: Data Pipeline With Virtual Staging

```python
from gmem import GMemContext

def etl_pipeline(raw_data: list[float]):
    stage1 = GMemContext(seed=1)  # Extract
    stage2 = GMemContext(seed=2)  # Transform
    stage3 = GMemContext(seed=3)  # Load

    # Extract: write raw data
    for i, val in enumerate(raw_data):
        stage1[i] = val

    # Transform: morph to double + offset
    stage2.morph_attach(stage1, mode=1, a=2.0, b=0.5)

    # Load: read transformed values
    return [stage2[i] for i in range(len(raw_data))]
```

### Pattern B: Mirrored Read Replicas

```python
from gmem import GMemContext

primary = GMemContext(seed=42)
replica1 = GMemContext(seed=0)
replica2 = GMemContext(seed=0)

replica1.mirror_attach(primary)
replica2.mirror_attach(primary)

# Write to primary — all replicas see it
primary[0] = 999.0

assert replica1[0] == 999.0  # ✓
assert replica2[0] == 999.0  # ✓
```

### Pattern C: Persistent Virtual Memory

```python
from gmem import GMemContext

ctx = GMemContext(seed=42)

# Attach AOF — every write is persisted to disk
ctx.persistence_attach("my_state.gvm_delta")
ctx[0] = 1.0
ctx[1] = 2.0

# Later, in a new process:
ctx2 = GMemContext(seed=42)
ctx2.persistence_attach("my_state.gvm_delta", high_precision=True)  # Using bit-exact mode
assert ctx2[0] == 1.0  # ✓ exact recovery
```

> [!TIP]
> **High-Precision Mode**: Pass `high_precision=True` to `persistence_attach` to use `float64` (8-byte) storage. This ensures bit-exact recovery for scientific data at the cost of larger AOF files.

### Pattern D: Multi-Tenant Isolation

```python
from gmem import GMemManager

mgr = GMemManager()

# Each tenant gets an isolated 16 EB space
tenant_a = mgr.get_by_seed(0xAAAA, quota=10000)
tenant_b = mgr.get_by_seed(0xBBBB, quota=10000)

tenant_a[0] = 1.0
tenant_b[0] = 2.0

assert tenant_a[0] != tenant_b[0]  # Completely isolated
print(mgr.list_tenants())
```

### Pattern E: JSON-Path Virtual Documents

```python
from gmem import GMemContext, JSONFilter

ctx = GMemContext(seed=42)
doc = JSONFilter("user_profiles")

# Read/write by semantic path
doc.set_val(ctx, "users[0].name", 0.42)
doc.set_val(ctx, "users[0].score", 0.95)

name_val  = doc.get_val(ctx, "users[0].name")   # 0.42 (written)
email_val = doc.get_val(ctx, "users[0].email")   # synthesized
```

### Pattern F: Bridge — Treat Linear Memory as Virtual

```python
from gmem import GMemContext, phi, psi_read, psi_write

ctx = GMemContext(seed=42)

# Write using system-style linear addresses
psi_write(ctx, 0x00400000, 3.14)  # like writing to a process address

# Read back — page locality is preserved
val = psi_read(ctx, 0x00400000)
assert val == 3.14
```

---

## 7. Connecting to External Libraries

### NumPy Integration

```python
import numpy as np
from gmem import GMemContext

ctx = GMemContext(seed=42)

# Generate a NumPy array from the manifold (lazy → materialized)
data = np.array(ctx.fetch_bulk(0, 10000), dtype=np.float32)

# Process with NumPy as usual
mean = np.mean(data)
std  = np.std(data)
```

### Pandas Integration

```python
import pandas as pd
from gmem import GMemContext

ctx = GMemContext(seed=42)
values = ctx.fetch_bulk(0, 1000)

df = pd.DataFrame({
    'address': range(1000),
    'value': values,
})
```

### Any Framework

The key equation: **read from the manifold, hand off to your framework.**

```python
from gmem import GMemContext

ctx = GMemContext(seed=42)

# For any library expecting a list/array/iterable:
your_library.process(ctx.fetch_bulk(start=0, count=N))
```

---

---

## 9. Advanced Scale-Invariance ($2^{128}$ Addressing)

GMem's 64-bit space can be layered to provide effectively infinite addressing.

```python
import struct
from gmem import GMemContext

def fetch_128(root_seed, addr_hi, addr_lo):
    root = GMemContext(root_seed)
    
    # Use entropy from layer 1 as seed for layer 2
    sub_seed_raw = root.fetch(addr_hi)
    sub_seed = struct.unpack('<Q', struct.pack('<d', sub_seed_raw))[0]
    
    layer2 = GMemContext(sub_seed)
    return layer2.fetch(addr_lo)

val = fetch_128(0x9001, 0xAAAA_BBBB, 0xCCCC_DDDD)
```

## 10. Multi-tenancy & Namespace Signatures

Prevent collisions between different modules by "signing" your seeds.

```python
BASE_SEED = 0x1234
SIG_LOGS = 0x4C4F4753 << 32  # 'LOGS'
SIG_META = 0x4D455441 << 32  # 'META'

# Distinct, non-colliding virtual spaces
ctx_logs = GMemContext(BASE_SEED ^ SIG_LOGS)
ctx_meta = GMemContext(BASE_SEED ^ SIG_META)
```

## 11. PyTorch & Machine Learning

GMem provides a massive, sparse backbone for weights and activations.

```python
import torch
from gmem import GMemContext

# Virtual weight initialization
ctx = GMemContext(seed=42)
weights = torch.tensor(ctx.fetch_bulk(0, 1024), dtype=torch.float32)

# Persistent Activation Cache
from gmem.decorator import gmem_cached

@gmem_cached(seed=777, namespace="ml_cache")
def detect_objects(image_tensor):
    # Result is stored in GMem manifold across restarts
    return model(image_tensor).tolist()
```

---

## 12. Performance Tips

| Operation | Cost | Notes |
|---|---|---|
| `ctx[addr]` read (clean) | O(1) | Pure math — no memory lookup |
| `ctx[addr]` read (dirty) | O(1) | Dict lookup |
| `ctx[addr] = val` write | O(1) amortized | Dict insert |
| `ctx.search(target)` | O(log log 2⁶⁴) | ~4-6 iterations |
| `ctx.fetch_bulk(a, n)` | O(n) | Skips overlay for clean pages |
| Memory per context | ~1 KB | Base overhead |
| Memory per write | ~80 bytes | Dict entry overhead |

**Rule of thumb:** If you have 1 GB of RAM, you can store ~12 million writes while addressing 16 Exabytes of virtual space.
