"""
gmem — Generative Memory for Python
====================================

A pure-Python implementation of the Generative Memory (GMem) system.
Provides a 2⁶⁴ virtual address space (~16 Exabytes) of deterministic
procedural values with O(1) fetch, sparse overlay writes, monotonic
search, mirroring, morphing, persistence, and multi-tenant management.

Quick Start:
    >>> from gmem import GMemContext
    >>> ctx = GMemContext(seed=12345)
    >>> ctx.fetch(0)                    # Synthesize value at address 0
    >>> ctx[10**15]                     # Index into 1 PB offset
    >>> ctx[42] = 3.14                  # Sparse write (only this costs RAM)
    >>> ctx.search(0.5)                 # Find address nearest to 0.5

Modules:
    core         — GMemContext main class
    vrns         — vRNS projection + torus float synthesis (16-prime + CRT)
    hashing      — FNV-1a, Feistel, CRC32, Xi, DJB2, fmix64, vl_mask
    overlay      — Sparse physical overlay
    zmask        — Hierarchical dirty page tracking
    monotonic    — Monotonic manifold + interpolation search
    morph        — Variety morphing (affine + Gielis superformula)
    bridge       — Phi/Psi linear↔manifold mapping
    trinity      — Hilbert curve, CRT, Trinity solve
    filter       — JSON and Block filters
    persistence  — AOF delta log + overlay snapshots
    archetype    — Virtual filesystem archetypes
    allocator    — g_malloc/g_free style allocator
    manager      — Multi-tenant hypervisor
    ntt          — NTT Goldilocks field synthesis
    hdc          — 1024-bit Hyperdimensional Computing manifolds
    fourier      — Fourier-basis probability on RNS torus
    sidechannel  — Sidechannel detection / varietal fingerprinting
    sketch       — Johnson-Lindenstrauss dimension reduction
    number_theory — Number-theoretic structured synthesis
"""

# === Primary API ===
from gmem.core import GMemContext

# === Subsystems ===
from gmem.vrns import (
    synthesize, vrns_project, vrns_to_float, MODULI,
    synthesize_multichannel, vrns_project_16, crt_reconstruct,
    CRTEngine, MODULI_LEGACY, MODULI_16,
)
from gmem.hashing import (
    fnv1a_u64, fnv1a_string, djb2,
    feistel_shuffle, crc32, xi_mix,
    fmix64, vl_mask, vl_inverse_mask,
)
from gmem.overlay import Overlay
from gmem.zmask import ZMask
from gmem.monotonic import fetch_monotonic, search
from gmem.morph import (
    MorphMode, MorphParams, MorphState,
    GielisParams, r_gielis,
)
from gmem.bridge import phi, psi_read, psi_write
from gmem.trinity import (
    hilbert_xy_to_d, inductive_resolve_sorted,
    trinity_solve_rns, TRINITY_MODULI,
)
from gmem.filter import JSONFilter, BlockFilter
from gmem.persistence import AOFLog, save_overlay, load_overlay
from gmem.archetype import Archetype, ArchetypeManager, VirtEntry
from gmem.allocator import g_malloc, g_free, g_get_f32, g_set_f32
from gmem.manager import GMemManager
from gmem.decorator import (
    VirtualArray, GMemPool,
    gmem_context, gmem_virtual_array, gmem_cached, gmem_workspace,
)

# === New Matrix-V SDK Improvements ===
from gmem.ntt import (
    P_GOLDILOCKS, multiply_mod, add_mod, sub_mod, mod_inverse,
    synthesize_product_law, inverse_product_law,
    resolve_value_at, verify_law_roundtrip,
)
from gmem.hdc import HdcManifold, HDC_DIM
from gmem.fourier import (
    fourier_project, phase_extract, born_probability, signal_strength,
    hadamard_gate, pauli_x_gate, pauli_z_gate, phase_gate, s_gate, t_gate,
)
from gmem.sidechannel import SidechannelDetector, cosine_similarity
from gmem.sketch import adaptive_dimension, SpectralSketch, sketch_bulk
from gmem.number_theory import (
    is_prime, mobius, mertens_sieve,
    divisor_lattice, divisor_sigma,
    NumberTheoreticSynthesizer,
)

__version__ = "2.0.0"
__all__ = [
    # Core
    'GMemContext',
    # Synthesis (legacy + enhanced)
    'synthesize', 'vrns_project', 'vrns_to_float', 'MODULI',
    'synthesize_multichannel', 'vrns_project_16', 'crt_reconstruct',
    'CRTEngine', 'MODULI_LEGACY', 'MODULI_16',
    # Hashing (original + invertible + fmix64)
    'fnv1a_u64', 'fnv1a_string', 'djb2',
    'feistel_shuffle', 'crc32', 'xi_mix',
    'fmix64', 'vl_mask', 'vl_inverse_mask',
    # Overlay
    'Overlay',
    # Z-Mask
    'ZMask',
    # Monotonic
    'fetch_monotonic', 'search',
    # Morphing (affine + Gielis)
    'MorphMode', 'MorphParams', 'MorphState',
    'GielisParams', 'r_gielis',
    # Bridge
    'phi', 'psi_read', 'psi_write',
    # Trinity
    'hilbert_xy_to_d', 'inductive_resolve_sorted',
    'trinity_solve_rns', 'TRINITY_MODULI',
    # Filters
    'JSONFilter', 'BlockFilter',
    # Persistence
    'AOFLog', 'save_overlay', 'load_overlay',
    # Archetypes
    'Archetype', 'ArchetypeManager', 'VirtEntry',
    # Allocator
    'g_malloc', 'g_free', 'g_get_f32', 'g_set_f32',
    # Manager
    'GMemManager',
    # Decorators
    'VirtualArray', 'GMemPool',
    'gmem_context', 'gmem_virtual_array', 'gmem_cached', 'gmem_workspace',
    # NTT Goldilocks Field
    'P_GOLDILOCKS', 'multiply_mod', 'add_mod', 'sub_mod', 'mod_inverse',
    'synthesize_product_law', 'inverse_product_law',
    'resolve_value_at', 'verify_law_roundtrip',
    # HDC Manifolds
    'HdcManifold', 'HDC_DIM',
    # Fourier Probability
    'fourier_project', 'phase_extract', 'born_probability', 'signal_strength',
    'hadamard_gate', 'pauli_x_gate', 'pauli_z_gate', 'phase_gate',
    's_gate', 't_gate',
    # Sidechannel Detection
    'SidechannelDetector', 'cosine_similarity',
    # JL Sketch
    'adaptive_dimension', 'SpectralSketch', 'sketch_bulk',
    # Number Theory
    'is_prime', 'mobius', 'mertens_sieve',
    'divisor_lattice', 'divisor_sigma',
    'NumberTheoreticSynthesizer',
]
