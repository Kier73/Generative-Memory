"""
gmem.fourier — Fourier-Basis Probability on the RNS Torus.

Treats RNS residues as oscillators on a torus and computes
interference patterns via Fourier projection.

Mathematics:
    ψₖ = e^(i·2π·rₖ/pₖ)
    Ψ  = Σ ψₖ = Σ cos(2πrₖ/pₖ) + i·Σ sin(2πrₖ/pₖ)
    θ  = atan2(Im(Ψ), Re(Ψ))
    P  = cos²(θ/2)          (Born rule isomorphism)

Gate Transforms (phase operations on residues):
    Hadamard:    r'ᵢ = (pᵢ/4 - rᵢ) mod pᵢ
    Pauli-X:     r'ᵢ = (rᵢ + pᵢ/2) mod pᵢ
    Phase(θ):    r'ᵢ = (rᵢ + ⌊pᵢ·θ/2π⌉) mod pᵢ
"""

import math
from gmem.vrns import MODULI_16


def fourier_project(residues: tuple[int, ...] | list[int],
                    primes: tuple[int, ...] = MODULI_16) -> tuple[float, float]:
    """
    Project residues onto the complex unit circle.

    Each residue rₖ with prime pₖ contributes a phase angle 2π·rₖ/pₖ.
    Returns (real_sum, imag_sum) — the interference pattern.
    """
    real_sum = 0.0
    imag_sum = 0.0
    for r, p in zip(residues, primes):
        theta = 2.0 * math.pi * (r / p)
        real_sum += math.cos(theta)
        imag_sum += math.sin(theta)
    return real_sum, imag_sum


def phase_extract(residues: tuple[int, ...] | list[int],
                  primes: tuple[int, ...] = MODULI_16) -> float:
    """
    Extract the resultant phase from residue interference.

    θ = atan2(Im(Ψ), Re(Ψ))
    """
    re, im = fourier_project(residues, primes)
    return math.atan2(im, re)


def born_probability(residues: tuple[int, ...] | list[int],
                     primes: tuple[int, ...] = MODULI_16) -> float:
    """
    Born Rule Isomorphism: P = cos²(θ/2).

    Treats the interference of RNS oscillators as a state vector
    and yields a probability in [0, 1].
    """
    theta = phase_extract(residues, primes)
    return math.cos(theta / 2.0) ** 2


def signal_strength(residues: tuple[int, ...] | list[int],
                    primes: tuple[int, ...] = MODULI_16) -> float:
    """
    Interference amplitude: |Ψ| / N.

    1.0 = perfectly coherent (all oscillators in phase)
    0.0 = fully destructive (uniform distribution of phases)
    """
    re, im = fourier_project(residues, primes)
    n = len(residues)
    return math.sqrt(re * re + im * im) / n


# ── Gate Operations ────────────────────────────────────────────

def hadamard_gate(residues: list[int],
                  primes: tuple[int, ...] = MODULI_16) -> list[int]:
    """
    Hadamard gate: reflection on the RNS torus.

    r'ᵢ = (pᵢ/4 - rᵢ) mod pᵢ
    """
    return [(p // 4 - r) % p for r, p in zip(residues, primes)]


def pauli_x_gate(residues: list[int],
                 primes: tuple[int, ...] = MODULI_16) -> list[int]:
    """
    Pauli-X (bit flip) gate.

    r'ᵢ = (rᵢ + pᵢ/2) mod pᵢ
    """
    return [(r + p // 2) % p for r, p in zip(residues, primes)]


def pauli_z_gate(residues: list[int],
                 primes: tuple[int, ...] = MODULI_16) -> list[int]:
    """
    Pauli-Z (phase flip) gate.

    r'ᵢ = (rᵢ + pᵢ/2) mod pᵢ
    """
    return [(r + p // 2) % p for r, p in zip(residues, primes)]


def phase_gate(residues: list[int], theta: float,
               primes: tuple[int, ...] = MODULI_16) -> list[int]:
    """
    Analog phase shift gate.

    r'ᵢ = (rᵢ + ⌊pᵢ·θ/2π⌉) mod pᵢ
    """
    return [(r + round(p * theta / (2 * math.pi))) % p
            for r, p in zip(residues, primes)]


def s_gate(residues: list[int],
           primes: tuple[int, ...] = MODULI_16) -> list[int]:
    """S gate (π/2 phase shift)."""
    return [(r + p // 4) % p for r, p in zip(residues, primes)]


def t_gate(residues: list[int],
           primes: tuple[int, ...] = MODULI_16) -> list[int]:
    """T gate (π/4 phase shift)."""
    return [(r + p // 8) % p for r, p in zip(residues, primes)]
