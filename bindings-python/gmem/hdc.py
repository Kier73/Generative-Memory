"""
gmem.hdc — 1024-bit Hyperdimensional Computing (HDC) Manifolds.

Provides context identity via 1024-bit vectors (16 × u64) with:
    - XOR binding for compositional products
    - Hamming-based cosine similarity
    - Majority-vote bundling (superposition)
    - Circular shift for directional permutation

Mathematics:
    Generation:  v[i] = fmix64(seed + i·φ)     for i ∈ [0, 16)
    Binding:     v_C = v_A ⊕ v_B                (XOR, associative, invertible)
    Similarity:  sim = 1 - 2·d_H(v_A, v_B)/1024 (normalized cosine via Hamming)
    Bundling:    v_out[k] = majority(v₁[k], v₂[k], ..., vₙ[k])
"""

from gmem.hashing import fmix64, MASK64

HDC_DIM = 1024
U64_COUNT = 16  # 1024 / 64
PHI = 0x9E3779B97F4A7C15  # Golden ratio constant


def _popcount(x: int) -> int:
    """Count set bits in a 64-bit integer."""
    return bin(x & MASK64).count('1')


class HdcManifold:
    """
    1024-bit bit-packed vector for hyperdimensional computing.

    Bit semantics: 0 → +1, 1 → -1 (Rademacher convention).
    XOR binding = compositional product.
    """

    __slots__ = ('data', 'label')

    def __init__(self, data: list[int] | None = None,
                 seed: int | None = None, label: str | None = None):
        if data is not None:
            self.data = [d & MASK64 for d in data]
        elif seed is not None:
            self.data = self._from_seed(seed)
        else:
            self.data = [0] * U64_COUNT
        self.label = label

    @staticmethod
    def _from_seed(seed: int) -> list[int]:
        """Deterministic 1024-bit vector generation via fmix64."""
        data = [0] * U64_COUNT
        s = seed
        for i in range(U64_COUNT):
            s = (s + PHI) & MASK64
            data[i] = fmix64(s)
        return data

    def bind(self, other: 'HdcManifold') -> 'HdcManifold':
        """
        XOR Binding: v_C = v_A ⊕ v_B

        Properties:
            - Associative: (A⊕B)⊕C = A⊕(B⊕C)
            - Invertible:  A⊕A = 0
            - Approximately orthogonal for random vectors
        """
        res = [self.data[i] ^ other.data[i] for i in range(U64_COUNT)]
        new_label = None
        if self.label and other.label:
            new_label = f"({self.label}*{other.label})"
            if len(new_label) > 128:
                new_label = new_label[:120] + "...)"
        return HdcManifold(data=res, label=new_label)

    def shift(self, n: int) -> 'HdcManifold':
        """
        Circular Shift (Directional Permutation).

        Shifts the entire 1024-bit vector by n positions.
        """
        n = n % HDC_DIM
        if n == 0:
            return HdcManifold(data=list(self.data), label=self.label)

        word_shift = n // 64
        bit_shift = n % 64
        out = [0] * U64_COUNT

        for i in range(U64_COUNT):
            src = (i - word_shift) % U64_COUNT
            if bit_shift == 0:
                out[i] = self.data[src]
            else:
                src_prev = (src - 1) % U64_COUNT
                out[i] = ((self.data[src] << bit_shift) |
                          (self.data[src_prev] >> (64 - bit_shift))) & MASK64

        return HdcManifold(data=out, label=self.label)

    def bundle(self, others: list['HdcManifold']) -> 'HdcManifold':
        """
        Majority-Vote Bundling (superposition of multiple manifolds).

        For each bit position, take the majority vote across all inputs.
        """
        all_vecs = [self] + others
        threshold = len(all_vecs) // 2
        out = [0] * U64_COUNT

        for word in range(U64_COUNT):
            for bit in range(64):
                count = sum(1 for v in all_vecs if (v.data[word] >> bit) & 1)
                if count > threshold:
                    out[word] |= (1 << bit)

        return HdcManifold(data=out)

    def similarity(self, other: 'HdcManifold') -> float:
        """
        Normalized Cosine Similarity via Hamming distance.

        sim(A, B) = 1 - 2·d_H(A,B) / 1024

        Returns:
            1.0 for identical, 0.0 for orthogonal, -1.0 for anti-correlated.
        """
        hamming = 0
        for i in range(U64_COUNT):
            hamming += _popcount(self.data[i] ^ other.data[i])
        return 1.0 - (2.0 * hamming / HDC_DIM)

    def resolve(self, index: int) -> float:
        """
        Resolve a single bit to a Rademacher value (+1 or -1).

        Convention: bit 0 → +1, bit 1 → -1.
        """
        word = (index % HDC_DIM) // 64
        bit = index % 64
        val = (self.data[word] >> bit) & 1
        return 1.0 if val == 0 else -1.0

    def fingerprint(self) -> int:
        """Collapse 1024 bits to a single 64-bit signature."""
        h = 0
        for i in range(U64_COUNT):
            h ^= fmix64(self.data[i] | (i << 48))
        return h & MASK64

    def __eq__(self, other):
        if not isinstance(other, HdcManifold):
            return NotImplemented
        return self.data == other.data

    def __repr__(self):
        fp = self.fingerprint()
        lbl = f" '{self.label}'" if self.label else ""
        return f"HdcManifold{lbl}(fp=0x{fp:016X})"
