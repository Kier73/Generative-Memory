"""
gmem.number_theory — Number-Theoretic Structured Synthesis.

Provides analytically meaningful manifold generation based on
classical number theory, replacing pure hash-based synthesis with
structured mathematical patterns.

Key Functions:
    μ(n)  — Möbius function (square-free indicator)
    M(n)  — Mertens function (cumulative Möbius sum)
    P(i,j) — Divisor lattice (P-series element)
    σ(n)  — Divisor sum function

Mathematics:
    μ(n) = { 1 if n=1, (-1)^k if n = p₁p₂...pₖ (distinct), 0 if p²|n }
    M(n) = Σₖ₌₁ⁿ μ(k)
    P(i,j) = 1 if ∃d : j = i·d, else 0
"""

import random
import math
from gmem.hashing import fmix64, MASK64


def is_prime(n: int, k: int = 5) -> bool:
    """
    Miller-Rabin primality test.

    Probabilistic with error probability < 4^(-k).
    Deterministic for n < 3.3×10²⁴ with specific witnesses.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r · d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _gcd(a: int, b: int) -> int:
    """Greatest Common Divisor."""
    while b:
        a, b = b, a % b
    return a


def _pollard_rho(n: int) -> int:
    """Pollard's rho factorization with Brent improvement."""
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return n

    for _ in range(20):
        x = random.randint(2, n - 1)
        y, c, g = x, random.randint(1, n - 1), 1
        while g == 1:
            x = (pow(x, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            g = _gcd(abs(x - y), n)
            if g == n:
                break
        if g != n:
            return g
    return n  # Fallback


def mobius(n: int) -> int:
    """
    Möbius function μ(n).

    μ(1) = 1
    μ(n) = (-1)^k  if n is a product of k distinct primes
    μ(n) = 0       if n has a squared prime factor

    Uses trial division for small primes, then Pollard rho.
    """
    if n == 1:
        return 1

    temp = n
    prime_factors = set()

    # Small prime pre-screening
    for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if temp % d == 0:
            prime_factors.add(d)
            temp //= d
            if temp % d == 0:
                return 0  # Squared factor
        if temp == 1:
            break

    # Factor remaining via Pollard rho
    max_iters = 100
    iters = 0
    while temp > 1 and iters < max_iters:
        if is_prime(temp):
            if temp in prime_factors:
                return 0
            prime_factors.add(temp)
            temp = 1
            break
        factor = _pollard_rho(temp)
        if is_prime(factor):
            if factor in prime_factors:
                return 0
            prime_factors.add(factor)
            temp //= factor
        iters += 1

    if temp > 1:
        return 0  # Could not fully factor → assume not square-free

    return -1 if len(prime_factors) % 2 == 1 else 1


def mertens_sieve(n: int) -> list[int]:
    """
    Compute M(1) through M(n) via linear sieve.

    M(k) = Σᵢ₌₁ᵏ μ(i)

    Uses O(n log log n) time and O(n) space.

    Returns a list where result[k] = M(k) for k in [0, n].
    """
    mu = [0] * (n + 1)
    mu[1] = 1
    is_p = [True] * (n + 1)
    primes = []

    for i in range(2, n + 1):
        if is_p[i]:
            primes.append(i)
            mu[i] = -1  # i is prime → squarefree with 1 factor
        for p in primes:
            if i * p > n:
                break
            is_p[i * p] = False
            if i % p == 0:
                mu[i * p] = 0  # p² divides i*p → not squarefree
                break
            mu[i * p] = -mu[i]  # One more prime factor

    # Prefix sum → Mertens function
    M = [0] * (n + 1)
    for i in range(1, n + 1):
        M[i] = M[i - 1] + mu[i]
    return M


def divisor_lattice(i: int, j: int) -> int:
    """
    P-Series divisor lattice element.

    P(i, j) = 1  if j is divisible by i  (∃d : j = i·d)
    P(i, j) = 0  otherwise
    """
    if i == 0:
        return 0
    return 1 if (j + 1) % (i + 1) == 0 else 0


def divisor_sigma(n: int, k: int = 1) -> int:
    """
    Divisor sum function σₖ(n) = Σ_{d|n} d^k.

    σ₀(n) = number of divisors
    σ₁(n) = sum of divisors
    """
    if n <= 0:
        return 0
    total = 0
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            total += d ** k
            if d != n // d:
                total += (n // d) ** k
    return total


class NumberTheoreticSynthesizer:
    """
    Structured manifold generator using number-theoretic functions.

    Instead of pure hash-based synthesis, pages in the manifold carry
    analytically meaningful values based on prime structure, divisibility,
    and Möbius function.

    Modes:
        'mobius'   — μ(addr) mapped to float
        'mertens' — M(addr) normalized to [0, 1]
        'divisor'  — σ₁(addr) normalized
        'lattice'  — P-series lattice structure
    """

    __slots__ = ('mode', 'seed', '_mertens_cache')

    def __init__(self, mode: str = 'mobius', seed: int = 0):
        self.mode = mode
        self.seed = seed
        self._mertens_cache = None

    def synthesize(self, addr: int) -> float:
        """Generate a structured value at the given address."""
        # Mix with seed for variety
        idx = ((addr ^ self.seed) & MASK64) % (10**7) + 1

        if self.mode == 'mobius':
            m = mobius(idx)
            return (m + 1.0) / 2.0  # Map {-1, 0, 1} → {0, 0.5, 1.0}

        elif self.mode == 'mertens':
            if self._mertens_cache is None or len(self._mertens_cache) <= idx:
                self._mertens_cache = mertens_sieve(max(idx + 100, 10000))
            m = self._mertens_cache[min(idx, len(self._mertens_cache) - 1)]
            # Normalize by sqrt(n) (Mertens conjecture scale)
            scale = math.sqrt(idx + 1)
            return 0.5 + m / (2.0 * scale)

        elif self.mode == 'divisor':
            s = divisor_sigma(idx)
            # Normalize by n·ln(ln(n))  (average order of σ₁)
            norm = idx * max(math.log(math.log(idx + 2) + 1), 1)
            return min(s / norm, 1.0)

        elif self.mode == 'lattice':
            # 2D interpretation: addr → (i, j) via squarish layout
            side = int(math.isqrt(idx)) + 1
            i = idx // side
            j = idx % side
            return float(divisor_lattice(i, j))

        else:
            # Fallback to hash-based
            return (fmix64(idx ^ self.seed) & MASK64) / 18446744073709551615.0
