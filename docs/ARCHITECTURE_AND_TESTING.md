# GMem v0.2.0: Architecture & Testing Strategy

Focus: **Generative Procedural Memory (O(1) Synthetic Storage)**

## General Codebase Architecture

The GMem codebase is structured as a **multi-layered synthetic substrate**. It moves from raw bit-mixing to structured mathematical manifolds, and finally to a Pythonic object interface.

### 1. The Core Manifold (Synthesis Layer)
- **`vrns.py`**: The heart of the system. Uses 16-prime Virtual Residue Number Systems to project an address into a unique residue vector, then collapses it to a float.
- **`hashing.py`**: Provides the entropy source. Includes `vl_mask` (invertible Feistel) and `fmix64` (MurmurMix) for high-diffusion bit distribution.
- **`ntt.py`**: Advanced algebraic layer. Operates in the Goldilocks prime field to allow contexts to be multiplied or factored.

### 2. State & Persistence (Physical Layer)
- **`overlay.py`**: A thread-safe sparse dictionary. This is the **only** part of GMem that uses physical RAM (only for addresses you explicitly write to).
- **`persistence.py`**: Implements Append-Only-Files (AOF) for delta logging and snapshotting of the sparse overlay.
- **`zmask.py`**: A hierarchical dirty-page tracker used to optimize persistence and synchronization.

### 3. Analysis & Structure (Analytical Layer)
- **`fourier.py` / `hdc.py`**: Treats the manifold as a signal or a high-dimensional vector space for similarity search and probabilistic queries.
- **`number_theory.py`**: Replaces randomness with structured mathematical patterns (Möbius/Mertens/Primes).
- **`sketch.py`**: Uses JL Projections to provide "blurry" but consistent views of massive data blocks.

### 4. Integration (Interface Layer)
- **`core.py`**: The `GMemContext` class that orchestrates all modules into a single `ctx[addr]` API.
- **`decorator.py`**: Transparently injects the system into external code via Python decorators and the `VirtualArray` bridge.

---

## Key Tests & Verification Strategies

To ensure the integrity of a 2⁶⁴-address virtual space, the following tests are critical:

### 1. Numerical Invariance (Round-trip Tests)
- **Identity**: Ensure `vl_inverse_mask(vl_mask(x, s), s) == x` for edge cases (`0`, `2^64-1`, `seed=0`).
- **CRT Exactness**: Verify that `crt_reconstruct(vrns_project_16(x))` exactly recovers `x` for values up to the ~310-bit dynamic range.
- **Algebraic Recovery**: Verify that factoring a product context (`ntt.py`) recovers the original parent law seeds without bit-drift.

### 2. Statistical Distribution (Entropy Audit)
- **Uniformity**: Run a Chi-Squared test on 1,000,000 synthesized values to ensure they are uniformly distributed in `[0, 1)`.
- **Avalanche Effect**: Change 1 bit in the address and measure the Hamming distance change in the resulting hash (target: 32 bits change for a 64-bit hash).
- **Collision Resistance**: Sample 10^7 random addresses and verify zero collisions in the 1024-bit HDC manifolds.

### 3. Structural Integrity (Stress Tests)
- **Large-Scale AOF Replay**: Write 1,000,000 sparse values, kill the process, and verify that persistence reloads every value with bit-exactness (CRC32 check).
- **HDC Similarity Stability**: Verify that `a.similarity(b)` remains stable across different seeds and that `bind()` logic preserves distances as expected for HDC vector math.
- **Slicing Consistency**: Verify that `v_arr[100:200]` produces the exact same values as 100 individual `fetch()` calls.

### 4. Mathematical Edge Cases
- **Superformula Singularities**: Test `Gielis` morphing with `params=0` or `n1=0` to ensure no `ZeroDivisionError` or `NaN` propagation.
- **Prime Density**: Verify `is_prime` against known Mersenne primes and large composite numbers to ensure the number-theoretic synthesizer doesn't misidentify divisors.
- **Interpolation Search Accuracy**: Test searching in a monotonic manifold where values are very close together (e.g., `slope = 10^-12`) to verify the binary search fallback.

---

## Recommendation: Integration Testing
The most important test for **User Application** is the **Backward Compatibility Audit**.
- **Task**: Run a script that generates values in v1.0.0 (8-prime) and v2.0.0 (16-prime toggle).
- **Goal**: Ensure the legacy `synthesize()` function still produces the *exact* same floats for existing v1.0.0 users while `synthesize_multichannel()` provides the higher-entropy v2.0.0 results.
