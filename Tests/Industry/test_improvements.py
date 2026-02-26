"""Comprehensive verification of all 10 GMem improvements."""
import sys
import traceback

results = []


def test(name, fn):
    try:
        ok = fn()
        tag = "PASS" if ok else "FAIL"
        results.append((name, tag))
        print(f"  [{tag}] {name}")
    except Exception as e:
        results.append((name, f"ERROR: {e}"))
        print(f"  [ERROR] {name}: {e}")
        traceback.print_exc()


print("=== GMem v2.0.0 Verification ===\n")

# 0. Import everything
print("Importing gmem...")
import gmem
print(f"  Version: {gmem.__version__}")
print(f"  Exports: {len(gmem.__all__)} symbols\n")


# 1. Invertible Hash
def test_invertible_hash():
    from gmem.hashing import vl_mask, vl_inverse_mask
    for addr in [0, 1, 42, 2**32, 2**63, 2**64 - 1]:
        for seed in [0, 0x517, 0xDEADBEEF]:
            h = vl_mask(addr, seed)
            recovered = vl_inverse_mask(h, seed)
            if recovered != addr:
                return False
    return True


test("Invertible Hash (vl_mask roundtrip)", test_invertible_hash)


# 2. fmix64
def test_fmix64():
    from gmem.hashing import fmix64
    # fmix64(0) = 0 is mathematically correct (MurmurMix64 fixed point)
    h1 = fmix64(1)
    assert h1 != 1, "fmix64 should mix"
    assert fmix64(42) != fmix64(43), "Distinct inputs should differ"
    return True


test("fmix64 avalanche", test_fmix64)


# 3. 16-Prime CRT
def test_crt():
    from gmem.vrns import CRTEngine, MODULI_16
    crt = CRTEngine(MODULI_16)
    for val in [0, 1, 42, 999999, 2**30]:
        if not crt.roundtrip(val):
            return False
    return True


test("16-Prime CRT Roundtrip", test_crt)


# 4. Multi-channel synthesis
def test_multichannel():
    from gmem.vrns import synthesize, synthesize_multichannel
    v_legacy = synthesize(42, 0x517)
    v_multi = synthesize_multichannel(42, 0x517)
    assert 0.0 <= v_legacy <= 1.0
    assert 0.0 <= v_multi <= 1.0
    assert v_legacy != v_multi
    return True


test("Multi-channel fmix64 Synthesis", test_multichannel)


# 5. NTT Goldilocks
def test_ntt():
    from gmem.ntt import verify_law_roundtrip, resolve_value_at
    assert verify_law_roundtrip(0x517, 0xCAFE)
    assert verify_law_roundtrip(12345, 67890)
    v = resolve_value_at(0x517, 10, 20)
    assert v >= 0
    return True


test("NTT Goldilocks Law Composition", test_ntt)


# 6. HDC Manifolds
def test_hdc():
    from gmem.hdc import HdcManifold
    a = HdcManifold(seed=42, label="A")
    b = HdcManifold(seed=43, label="B")
    assert a.similarity(a) == 1.0
    sim = a.similarity(b)
    assert -0.3 < sim < 0.3, f"Random sim out of range: {sim}"
    ab = a.bind(b)
    recovered = ab.bind(b)
    assert recovered.similarity(a) == 1.0
    val = a.resolve(0)
    assert val in (1.0, -1.0)
    return True


test("HDC Manifold Bind/Similarity/Resolve", test_hdc)


# 7. Gielis Morphing
def test_gielis():
    from gmem.morph import MorphMode, MorphState, GielisParams, r_gielis
    p = GielisParams(m=4.0, a=1.0, b=1.0, n1=0.5, n2=0.5, n3=0.5)
    r = r_gielis(0.0, p)
    assert r > 0
    state = MorphState()
    state.mode = MorphMode.GIELIS
    state.gielis_params = p
    v = state.apply(0.25)
    assert 0 < v < 100
    return True


test("Gielis Superformula Morphing", test_gielis)


# 8. Fourier Probability
def test_fourier():
    from gmem.fourier import born_probability, hadamard_gate, signal_strength
    from gmem.vrns import vrns_project_16
    residues = list(vrns_project_16(42, 0x517))
    p = born_probability(residues)
    assert 0.0 <= p <= 1.0
    sig = signal_strength(residues)
    assert 0.0 <= sig <= 1.0
    h_res = hadamard_gate(residues)
    assert h_res != residues
    return True


test("Fourier Probability on RNS Torus", test_fourier)


# 9. Sidechannel Detection
def test_sidechannel():
    from gmem.sidechannel import cosine_similarity
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9
    c = [-1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, c) - (-1.0)) < 1e-9
    d = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(a, d)) < 1e-9
    return True


test("Sidechannel Cosine Similarity", test_sidechannel)


# 10. JL Sketch
def test_sketch():
    from gmem.sketch import adaptive_dimension, SpectralSketch
    d = adaptive_dimension(1000000, epsilon=0.1)
    assert d > 0
    assert d < 1000000, f"Should reduce: {d}"
    # Use large N for actual reduction
    sk = SpectralSketch(10000, epsilon=0.3, seed=42)
    vals = [float(i) / 10000.0 for i in range(10000)]
    proj = sk.project(vals)
    assert len(proj) == sk.output_dim
    assert len(proj) < 10000, f"Not reduced: {len(proj)}"
    print(f"    JL: 10000 -> {len(proj)} (ratio: {sk.compression_ratio:.1f}x)")
    return True


test("JL Sketch Dimension Reduction", test_sketch)


# 11. Number Theory
def test_number_theory():
    from gmem.number_theory import mobius, mertens_sieve, is_prime, divisor_sigma
    assert is_prime(2) and is_prime(17) and not is_prime(15)
    assert mobius(1) == 1
    assert mobius(2) == -1
    assert mobius(4) == 0
    assert mobius(30) == -1
    M = mertens_sieve(100)
    assert M[1] == 1 and len(M) == 101
    assert divisor_sigma(6) == 12
    return True


test("Number Theory (Mobius/Mertens/Primes)", test_number_theory)

# Summary
passed = sum(1 for _, s in results if s == "PASS")
total = len(results)
print(f"\n=== Results: {passed}/{total} passed ===")
for name, status in results:
    if status != "PASS":
        print(f"  !! {name}: {status}")
if passed == total:
    print("  ALL TESTS PASSED")
