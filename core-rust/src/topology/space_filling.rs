//! Space-Filling Curves
//!
//! Maps multi-dimensional physical coordinates (X, Y) into a contiguous
//! 1-dimensional topological address space for Generative Memory.
//! This ensures that structural locality (e.g. image pixels) remains intact
//! when collapsed into the $2^{64}$ scalar line.

/// Encodes 2D (i, j) coordinates into a 1D Hilbert index.
/// `order` dictates the size of the grid ($2^{\text{order}} \times 2^{\text{order}}$).
#[inline(always)]
pub fn hilbert_encode(mut i: u64, mut j: u64, order: u32) -> u64 {
    let mut d: u64 = 0;
    // Safe shift since order is usually under 32 for practical grids
    let mut s: u64 = if order == 0 { 0 } else { 1 << (order - 1) };

    while s > 0 {
        let rx = if (i & s) > 0 { 1 } else { 0 };
        let ry = if (j & s) > 0 { 1 } else { 0 };

        d += s * s * ((3 * rx) ^ ry);

        // Rotate/Flip Quadrant
        if ry == 0 {
            if rx == 1 {
                i = s - 1 - i;
                j = s - 1 - j;
            }
            // Swap i and j
            let temp = i;
            i = j;
            j = temp;
        }

        s >>= 1;
    }
    d
}

/// Decodes a 1D Hilbert index back into 2D (i, j) coordinates.
#[inline(always)]
pub fn hilbert_decode(d: u64, order: u32) -> (u64, u64) {
    let mut i: u64 = 0;
    let mut j: u64 = 0;
    let mut t = d;
    let mut s: u64 = 1;

    while s < (1u64 << order) {
        let rx = 1 & (t / 2);
        let ry = 1 & (t ^ rx);

        if ry == 0 {
            if rx == 1 {
                i = s - 1 - i;
                j = s - 1 - j;
            }
            let temp = i;
            i = j;
            j = temp;
        }

        i += s * rx;
        j += s * ry;

        t /= 4;
        s <<= 1;
    }
    (i, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_bijectivity() {
        let order = 8; // 256x256 grid

        // Test a few specific coordinates
        let coords = [(0, 0), (255, 0), (0, 255), (255, 255), (100, 150)];
        for &(i, j) in &coords {
            let h = hilbert_encode(i, j, order);
            let (dec_i, dec_j) = hilbert_decode(h, order);
            assert_eq!(
                (i, j),
                (dec_i, dec_j),
                "Hilbert decoding failed for original coord!"
            );
        }
    }
}
