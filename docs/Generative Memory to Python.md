# Generative Memory → Python Translation Guide

> A complete mathematical and architectural reference for translating the C-based Generative Memory (GMem) system into pure Python.

---

## 1. Core Concept

Generative Memory provides a **virtual address space of 2⁶⁴ addresses** (~16 Exabytes) without allocating physical RAM. Every address deterministically synthesizes a `float ∈ [0, 1]` on-demand via a **procedural function of (address, seed)**. User writes are captured in a sparse overlay (hash map), so only *modified* cells consume memory.

**The Equation of the System:**

```
V(addr) = Overlay(addr)  if written,
           Synthesize(addr, seed)  otherwise
```

Physical cost = `O(writes)`, not `O(address_space)`.

---

## 2. Mathematical Primitives

### 2.1 FNV-1a Hash (Overlay & Path Addressing)

Used for the overlay hash map and JSON filter path resolution.

```
H₀ = 14695981039346656037  (FNV offset basis, 64-bit)
P  = 1099511628211          (FNV prime, 64-bit)

H(addr):
    h = H₀
    h = h XOR addr
    h = h × P
    return h
```

**Python:**
```python
def fnv1a_hash(addr: int) -> int:
    h = 14695981039346656037
    h = (h ^ addr) & 0xFFFFFFFFFFFFFFFF
    h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
```

For string paths (JSON filter):
```python
def fnv1a_string(s: str, salt: int = 0) -> int:
    h = 14695981039346656037 ^ salt
    for ch in s:
        h ^= ord(ch)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
```

---

### 2.2 Feistel Network Shuffle (Deterministic Permutation)

A 4-round Feistel cipher on a 64-bit value. Splits input into two 32-bit halves and applies round function `F(r) = r × φ + seed_low32`, where `φ = 0x9E3779B9` (golden ratio constant).

```
Input:  val (64-bit), seed (64-bit)
Split:  L = val >> 32,  R = val & 0xFFFFFFFF

For i in 0..3:
    temp = R
    R = L XOR ((R × 0x9E3779B9 + seed_low32) mod 2³²)
    L = temp

Output: (L << 32) | R
```

**Python:**
```python
def feistel_shuffle(val: int, seed: int) -> int:
    MASK32 = 0xFFFFFFFF
    l = (val >> 32) & MASK32
    r = val & MASK32
    s = seed & MASK32
    for _ in range(4):
        temp = r
        r = l ^ ((r * 0x9E3779B9 + s) & MASK32)
        l = temp
    return (l << 32) | r
```

---

### 2.3 vRNS Projection (Residue Number System)

The core synthesis engine. Projects a 64-bit value into 8 parallel residue channels using coprime moduli, then reconstructs a float via **Torus Projection**.

**Moduli (8 coprimes near 2⁸):**
```
M = [251, 257, 263, 269, 271, 277, 281, 283]
```

**Product of all moduli:**
```
∏M = 251 × 257 × 263 × 269 × 271 × 277 × 281 × 283
   ≈ 3.68 × 10¹⁹  (fits in 64-bit unsigned)
```

**Step 1 — Residue Projection:**
```
x = addr XOR seed
rᵢ = x mod Mᵢ    for i ∈ [0, 7]
```

**Step 2 — Torus Projection (Analytic Induction):**
```
accumulator = Σᵢ (rᵢ / Mᵢ)
result = frac(accumulator)       // fractional part only
```

This maps the residue vector onto the unit interval `[0, 1)`.

**Python:**
```python
MODULI = [251, 257, 263, 269, 271, 277, 281, 283]

def vrns_project(addr: int, seed: int) -> list[int]:
    x = (addr ^ seed) & 0xFFFFFFFFFFFFFFFF
    return [x % m for m in MODULI]

def vrns_to_float(residues: list[int]) -> float:
    acc = sum(r / m for r, m in zip(residues, MODULI))
    return acc - int(acc)  # fractional part

def synthesize(addr: int, seed: int) -> float:
    return vrns_to_float(vrns_project(addr, seed))
```

---

### 2.4 Monotonic Manifold (Sorted View)

Provides a **strictly increasing** view over the full 2⁶⁴ space. Used for binary/interpolation search.

```
base_ramp(index) = index / (2⁶⁴ - 1)

variety(index, seed) = feistel_shuffle(index, seed) / (2⁶⁴ - 1)²

V_monotonic(index) = base_ramp + variety
```

The variety term is divided by `(2⁶⁴ - 1)²` making it negligibly small — enough for uniqueness but preserving strict monotonicity.

**Python:**
```python
UINT64_MAX = (1 << 64) - 1

def fetch_monotonic(index: int, seed: int) -> float:
    base_ramp = index / UINT64_MAX
    variety_int = feistel_shuffle(index, seed)
    variety = (variety_int / UINT64_MAX) / UINT64_MAX
    return base_ramp + variety
```

---

### 2.5 Interpolation Search

Exploits the near-linear monotonic manifold for `O(log log n)` search.

```
Given target ∈ [0, 1]:
    low = 0, high = 2⁶⁴ - 1

    For up to 64 iterations:
        low_val  = V_monotonic(low)
        high_val = V_monotonic(high)

        scale = (target - low_val) / (high_val - low_val)
        pivot = low + ⌊scale × (high - low)⌋

        pivot_val = V_monotonic(pivot)

        if pivot_val < target:  low  = pivot + 1
        if pivot_val > target:  high = pivot - 1
        if pivot_val == target: return pivot
```

**Python:**
```python
def search_f32(target: float, seed: int) -> int:
    if target <= 0.0: return 0
    if target >= 1.0: return UINT64_MAX
    low, high = 0, UINT64_MAX
    for _ in range(64):
        if high <= low: break
        lv = fetch_monotonic(low, seed)
        hv = fetch_monotonic(high, seed)
        if target <= lv: return low
        if target >= hv: return high
        scale = (target - lv) / (hv - lv)
        pivot = low + int(scale * (high - low))
        pv = fetch_monotonic(pivot, seed)
        if pv < target:   low = pivot + 1
        elif pv > target: high = pivot - 1
        else:             return pivot
    return low
```

---

### 2.6 CRC32 (Integrity Shield)

Standard CRC32 with polynomial `0xEDB88320` (reflected).

```python
def crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1))
    return crc ^ 0xFFFFFFFF
```

---

### 2.7 Xi Mapping (Block Filter Sector→Manifold)

Maps a sector ID into the manifold via a MurmurHash3-style mixer:

```
ξ(sector_id, seed):
    d = sector_id XOR seed
    d = d × 0xBF58476D1CE4E5B9
    d = d XOR (d >> 33)
    d = d × 0x94D049BB133111EB
    d = d XOR (d >> 33)
    return d
```

---

### 2.8 Phi/Psi Bridge (Linear→Manifold Mapping)

Maps system linear addresses to manifold addresses preserving page locality:

```
PAGE_SIZE = 4096

Φ(linear_addr, seed):
    page_index = linear_addr // PAGE_SIZE
    offset     = linear_addr  % PAGE_SIZE
    base = (page_index XOR seed) × 0xBF58476D1CE4E5B9
    base = base XOR (base >> 33)
    return (base & ~(PAGE_SIZE-1)) | offset

Ψ_read(ctx, linear_addr)  = fetch(ctx, Φ(linear_addr, seed))
Ψ_write(ctx, linear_addr) = write(ctx, Φ(linear_addr, seed))
```

---

### 2.9 Hilbert Curve (2D→1D Spatial Mapping)

Used by Trinity for locality-preserving 2D coordinate mapping:

```python
def hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return d
```

---

### 2.10 Chinese Remainder Theorem (Trinity RNS Reconstruction)

Uses 3 Goldilocks primes for 128-bit resolution:

```
M₁ = 0xFFFFFFFF00000001  (Goldilocks)
M₂ = 0xFFFFFFFF7FFFFFFF  (Safe)
M₃ = 0xFFFFFFFFFFFFFFC5  (Ultra)

M_prod = M₁ × M₂ × M₃   (~192-bit number)

CRT(r₁, r₂, r₃):
    For each i:
        Mᵢ_partial = M_prod / Mᵢ
        yᵢ = modular_inverse(Mᵢ_partial, Mᵢ)   // Extended Euclidean
    return (Σ rᵢ × Mᵢ_partial × yᵢ) mod M_prod
```

---

## 3. Architecture Layers

### 3.1 Overlay (Sparse Physical Storage)

Open-addressing hash map with linear probing. Load factor threshold: 0.7.

| Field | Description |
|---|---|
| `virtual_addr` | 64-bit key |
| `value` | `float32` payload |
| `active` | Slot occupied flag |

On resize: capacity doubles, all entries rehashed.

### 3.2 Hierarchical Z-Mask (Dirty Page Tracking)

Two-level bitmap for tracking which pages have been written to:

- **Macro mask**: 1 bit per 1 TB chunk (covers 1 PB with 1024 bits = 128 bytes)
- **Detail mask**: 1 bit per 4 KB page within a TB chunk (allocated lazily)

```
TB_SIZE = 1 TB / sizeof(float)
PAGES_PER_TB = 1 TB / (1024 × sizeof(float))

is_dirty(addr):
    tb = addr // TB_SIZE
    if macro_mask[tb] not set → clean
    page = (addr % TB_SIZE) // PAGE_SIZE
    return detail_mask[tb][page] is set
```

### 3.3 Variety Morphing

Derives one context's values from another via real-time transforms:

| Mode | Equation |
|---|---|
| Identity | `y = x` |
| Linear | `y = a·x + b` |
| Add | `y = x + b` |
| Multiply | `y = x · a` |

### 3.4 Mirroring

Shadow context delegates reads to a source context (same seed, same overlay).

### 3.5 Persistence (AOF — Append-Only File)

Every write appends `(addr: u64, value: f32)` to a delta log. On reload, the log is replayed to reconstruct the overlay.

### 3.6 Multi-Tenant Manager

A hypervisor that maps `seed → context`. Supports quota-limited overlays per tenant.

---

## 4. Python Module Structure

```
gmem/
├── __init__.py          # Public API exports
├── core.py              # GMemContext class, fetch/write/search
├── vrns.py              # vRNS projection + torus float
├── overlay.py           # Sparse hash map overlay
├── monotonic.py         # Monotonic manifold + search
├── hashing.py           # FNV-1a, Feistel, CRC32, Xi, DJB2
├── bridge.py            # Phi/Psi linear↔manifold mapping
├── zmask.py             # Hierarchical Z-Mask
├── morph.py             # Variety morphing
├── trinity.py           # Hilbert curve, CRT, Trinity solve
├── filter.py            # JSON + Block filters
├── persistence.py       # AOF delta log save/load
├── archetype.py         # Virtual filesystem archetypes
├── manager.py           # Multi-tenant hypervisor
└── allocator.py         # g_malloc / g_free style allocator
```

---

## 5. Key Translation Decisions

| C Feature | Python Equivalent |
|---|---|
| `uint64_t` overflow | Mask with `& 0xFFFFFFFFFFFFFFFF` after every op |
| `uint32_t` overflow | Mask with `& 0xFFFFFFFF` |
| `float` (32-bit) | `struct.pack('<f', v)` round-trip for bit-exact |
| `__int128` | Python `int` (arbitrary precision — native) |
| `calloc` / `free` | Python `dict` (overlay), class lifecycle |
| Mutex / Locks | `threading.Lock()` |
| AVX2 SIMD | NumPy vectorized ops (optional fast path) |
| Pointer arithmetic | Array indexing / `bytearray` |
| File I/O (`fwrite`) | `struct.pack` + `open(..., 'wb')` |
| UDP sockets | `socket` module |

---

## 6. Minimal Working Core (Copy-Paste Ready)

```python
"""gmem_core.py — Generative Memory in Pure Python"""
import struct, math, threading

# === Constants ===
UINT64_MAX = (1 << 64) - 1
MASK64 = UINT64_MAX
MASK32 = 0xFFFFFFFF
MODULI = [251, 257, 263, 269, 271, 277, 281, 283]
OVERLAY_INITIAL = 1024
OVERLAY_LOAD = 0.7

# === Hash ===
def _fnv1a(addr):
    h = 14695981039346656037
    h = ((h ^ addr) * 1099511628211) & MASK64
    return h

# === Feistel Shuffle ===
def _feistel(val, seed):
    l, r, s = (val >> 32) & MASK32, val & MASK32, seed & MASK32
    for _ in range(4):
        l, r = r, l ^ ((r * 0x9E3779B9 + s) & MASK32)
    return ((l << 32) | r) & MASK64

# === vRNS Core ===
def _vrns_synth(addr, seed):
    x = (addr ^ seed) & MASK64
    acc = sum((x % m) / m for m in MODULI)
    return acc - int(acc)

# === Context ===
class GMemContext:
    def __init__(self, seed):
        self.seed = seed & MASK64
        self._overlay = {}
        self._lock = threading.Lock()
        self._morph_source = None
        self._morph_mode = 0
        self._morph_a = 1.0
        self._morph_b = 0.0
        self._mirror_source = None

    def fetch(self, addr):
        addr &= MASK64
        with self._lock:
            if addr in self._overlay:
                return self._overlay[addr]
        if self._morph_source:
            v = self._morph_source.fetch(addr)
            if self._morph_mode == 1: return v * self._morph_a + self._morph_b
            if self._morph_mode == 2: return v + self._morph_b
            if self._morph_mode == 3: return v * self._morph_a
            return v
        if self._mirror_source:
            return self._mirror_source.fetch(addr)
        return _vrns_synth(addr, self.seed)

    def write(self, addr, value):
        addr &= MASK64
        with self._lock:
            self._overlay[addr] = float(value)

    def fetch_monotonic(self, index):
        index &= MASK64
        base = index / UINT64_MAX
        variety = (_feistel(index, self.seed) / UINT64_MAX) / UINT64_MAX
        return base + variety

    def search(self, target):
        if target <= 0.0: return 0
        if target >= 1.0: return UINT64_MAX
        lo, hi = 0, UINT64_MAX
        for _ in range(64):
            if hi <= lo: break
            lv, hv = self.fetch_monotonic(lo), self.fetch_monotonic(hi)
            if target <= lv: return lo
            if target >= hv: return hi
            scale = (target - lv) / (hv - lv)
            pivot = lo + int(scale * (hi - lo))
            pv = self.fetch_monotonic(pivot)
            if pv < target:   lo = pivot + 1
            elif pv > target: hi = pivot - 1
            else:             return pivot
        return lo

    def fetch_bulk(self, start, count):
        return [self.fetch(start + i) for i in range(count)]

    def morph_attach(self, source, mode, a=1.0, b=0.0):
        self._morph_source = source
        self._morph_mode = mode
        self._morph_a, self._morph_b = a, b

    def mirror_attach(self, source):
        self._mirror_source = source

    def save_overlay(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<II', 0x4D454D47, 1))
            f.write(struct.pack('<Q', self.seed))
            items = list(self._overlay.items())
            f.write(struct.pack('<Q', len(items)))
            for addr, val in items:
                f.write(struct.pack('<Qf', addr, val))

    def load_overlay(self, path):
        with open(path, 'rb') as f:
            magic, ver = struct.unpack('<II', f.read(8))
            assert magic == 0x4D454D47 and ver == 1
            self.seed = struct.unpack('<Q', f.read(8))[0]
            count = struct.unpack('<Q', f.read(8))[0]
            for _ in range(count):
                addr, val = struct.unpack('<Qf', f.read(12))
                self._overlay[addr] = val

    @property
    def overlay_count(self):
        return len(self._overlay)
```

### Usage Example

```python
ctx = GMemContext(seed=12345)

# Read from anywhere in 16 EB space — O(1), zero RAM
v0 = ctx.fetch(0)
v1 = ctx.fetch(10**15)         # ~1 PB offset
v2 = ctx.fetch(2**63)          # 8 EB offset

# Sparse write — only this costs memory
ctx.write(42, 3.14)
assert ctx.fetch(42) == 3.14   # overlay wins

# Monotonic search
idx = ctx.search(0.5)          # find address closest to 0.5
print(ctx.fetch_monotonic(idx))

# Mirroring
shadow = GMemContext(seed=99999)
shadow.mirror_attach(ctx)
assert shadow.fetch(0) == ctx.fetch(0)

# Morphing (derived manifold)
derived = GMemContext(seed=0)
derived.morph_attach(ctx, mode=1, a=2.0, b=0.1)  # y = 2x + 0.1
```

---

## 7. Address Space Equation Summary

| Symbol | Definition | Value |
|---|---|---|
| `A` | Address width | 64 bits |
| `\|A\|` | Total addressable cells | 2⁶⁴ ≈ 1.8 × 10¹⁹ |
| `S` | Seed | 64-bit integer |
| `V(a,S)` | Synthesized value | `frac(Σ (a⊕S mod Mᵢ)/Mᵢ)` |
| `O` | Overlay (writes) | `dict{addr → float}` |
| `RAM_phys` | Physical memory | `O(|O|)` not `O(|A|)` |
| `M_mono(i)` | Monotonic value | `i/2⁶⁴ + feistel(i,S)/(2⁶⁴)²` |

**The fundamental invariant:**

```
∀ addr ∈ [0, 2⁶⁴):
    fetch(addr) = O[addr]              if addr ∈ O
                  V(addr, seed)        otherwise
    
    where V is pure, deterministic, and O(1)
```

This gives you **infinite read bandwidth** with **zero storage cost** for unwritten addresses, and **sparse O(1) writes** for modifications.
