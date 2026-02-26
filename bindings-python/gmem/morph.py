"""
gmem.morph — Variety Morphing (derived manifolds).

Allows one context to derive its values from another via real-time
transforms. This creates virtual "views" of the same underlying
data with zero additional storage.

Modes:
    IDENTITY (0) — y = x            (pass-through)
    LINEAR   (1) — y = a·x + b      (scale + shift)
    ADD      (2) — y = x + b        (shift only)
    MUL      (3) — y = x · a        (scale only)
    GIELIS   (4) — y = r_gielis(x·2π, m, a, b, n1, n2, n3)

Gielis Superformula:
    r(φ) = [ |cos(mφ/4)/a|^n2 + |sin(mφ/4)/b|^n3 ]^(-1/n1)
"""

import math
from enum import IntEnum


class MorphMode(IntEnum):
    """Variety morphing transformation modes."""
    IDENTITY = 0
    LINEAR = 1
    ADD = 2
    MUL = 3
    GIELIS = 4


class MorphParams:
    """Parameters for affine morphing transforms."""
    __slots__ = ('a', 'b')

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b

    def __repr__(self):
        return f"MorphParams(a={self.a}, b={self.b})"


class GielisParams:
    """
    Parameters for Gielis Superformula morphing.

    r(φ) = [ |cos(mφ/4)/a|^n2 + |sin(mφ/4)/b|^n3 ]^(-1/n1)

    Attributes:
        m:  Symmetry order (3=triangle, 4=square, 5=pentagon, ...)
        a:  Horizontal scale
        b:  Vertical scale
        n1: Curvature sharpness
        n2: Edge rounding (cos term)
        n3: Edge rounding (sin term)
    """
    __slots__ = ('m', 'a', 'b', 'n1', 'n2', 'n3')

    def __init__(self, m: float = 4.0, a: float = 1.0, b: float = 1.0,
                 n1: float = 0.5, n2: float = 0.5, n3: float = 0.5):
        self.m = m
        self.a = a
        self.b = b
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

    def __repr__(self):
        return (f"GielisParams(m={self.m}, a={self.a}, b={self.b}, "
                f"n1={self.n1}, n2={self.n2}, n3={self.n3})")


def r_gielis(phi: float, params: GielisParams) -> float:
    """
    Evaluate the Gielis Superformula at angle φ.

    r(φ) = [ |cos(mφ/4)/a|^n2 + |sin(mφ/4)/b|^n3 ]^(-1/n1)

    Returns a radius value, typically in (0, 1] for unit params.
    """
    m, a, b = params.m, params.a, params.b
    n1, n2, n3 = params.n1, params.n2, params.n3

    cos_term = math.cos(m * phi / 4.0) / a if a != 0 else 0.0
    sin_term = math.sin(m * phi / 4.0) / b if b != 0 else 0.0

    val = abs(cos_term) ** n2 + abs(sin_term) ** n3

    if val == 0 or n1 == 0:
        return 1.0

    return val ** (-1.0 / n1)


class MorphState:
    """
    Morphing attachment state for a context.

    When attached, the owning context will compute its values as
    a real-time transform of the source context's values.
    """
    __slots__ = ('source', 'mode', 'params', 'gielis_params')

    def __init__(self):
        self.source = None          # Source GMemContext (or None)
        self.mode = MorphMode.IDENTITY
        self.params = MorphParams()
        self.gielis_params = GielisParams()

    @property
    def active(self) -> bool:
        return self.source is not None

    def attach(self, source, mode: MorphMode, a: float = 1.0, b: float = 0.0):
        """Attach a source context for affine morphing."""
        self.source = source
        self.mode = MorphMode(mode)
        self.params = MorphParams(a, b)

    def attach_gielis(self, source, m: float = 4.0, a: float = 1.0,
                      b: float = 1.0, n1: float = 0.5,
                      n2: float = 0.5, n3: float = 0.5):
        """Attach a source context with Gielis superformula morphing."""
        self.source = source
        self.mode = MorphMode.GIELIS
        self.gielis_params = GielisParams(m, a, b, n1, n2, n3)

    def detach(self):
        """Detach morphing."""
        self.source = None
        self.mode = MorphMode.IDENTITY
        self.params = MorphParams()
        self.gielis_params = GielisParams()

    def apply(self, value: float) -> float:
        """Apply the morph transform to a source value."""
        if self.mode == MorphMode.LINEAR:
            return value * self.params.a + self.params.b
        elif self.mode == MorphMode.ADD:
            return value + self.params.b
        elif self.mode == MorphMode.MUL:
            return value * self.params.a
        elif self.mode == MorphMode.GIELIS:
            # Map value [0, 1) → angle [0, 2π), apply superformula
            phi = value * 2.0 * math.pi
            return r_gielis(phi, self.gielis_params)
        return value  # IDENTITY

