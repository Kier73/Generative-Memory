//! Invertible Bijective Address Transform
//!
//! Generative Memory requires $O(1)$ invertible addressing to project sectors
//! safely without associative search. This module supplies the core Feistel-based variety generator.
//!
//! Every bit pattern output hashes back onto a unique origin, making this a perfect bijection.

/// Spectral Anchor Constant (Golden Ratio Fractional Part)
/// $ C_1 = \text{0x9E3779B97F4A7C15} $
const C_MAGIC: u64 = 0x9E3779B97F4A7C15;

/// First multiplier for avalanche
const C2: u64 = 0xBF58476D1CE4E5B9;
/// Second multiplier for avalanche
const C3: u64 = 0x94D049BB133111EB;

/// Modular multiplicative inverse for $C_2 \pmod{2^{64}}$
const INV_C2: u64 = 0x96DE1B173F119089;
/// Modular multiplicative inverse for $C_3 \pmod{2^{64}}$
const INV_C3: u64 = 0x319642B2D24D8EC3;

/// Invertible Feistel-based variety generator.
///
/// Maps (address, seed) to a 64-bit value with full avalanche properties.
///
/// The mathematical flow is identical to the Formal Specification:
/// 1. $ z = (\text{addr} + \text{seed} + \phi) \pmod{2^{64}} $
/// 2. $ z = (z \oplus (z \gg 30)) \times C_2 \pmod{2^{64}} $
/// 3. $ z = (z \oplus (z \gg 27)) \times C_3 \pmod{2^{64}} $
/// 4. $ z = z \oplus (z \gg 31) $
#[inline(always)]
pub fn vl_mask(addr: u64, seed: u64) -> u64 {
    let mut z = addr.wrapping_add(seed).wrapping_add(C_MAGIC);
    z = (z ^ (z >> 30)).wrapping_mul(C2);
    z = (z ^ (z >> 27)).wrapping_mul(C3);
    z ^ (z >> 31)
}

/// Helper function to invert the xor-shift operation: $ v = z \oplus (z \gg k) $
///
/// Since $z$ is 64-bit, we progressively shift and XOR to recover the original $z$.
#[inline(always)]
fn invert_shift_xor(val: u64, k: u32) -> u64 {
    let mut v = val;
    let mut res = val;
    loop {
        v >>= k;
        if v == 0 {
            break;
        }
        res ^= v;
    }
    res
}

/// Exact inverse of `vl_mask`.
///
/// Recovers the exact input address from the generated variety hash.
/// Given $ h = \text{vl\_mask}(\text{addr}, \text{seed}) $, then $ \text{vl\_inverse\_mask}(h, \text{seed}) == \text{addr} $.
#[inline(always)]
pub fn vl_inverse_mask(h: u64, seed: u64) -> u64 {
    // 1. Undo final xorshift: z ^ (z >> 31)
    let mut z = invert_shift_xor(h, 31);

    // 2. Undo multiplication by C3
    z = z.wrapping_mul(INV_C3);

    // 3. Undo xorshift: z ^ (z >> 27)
    z = invert_shift_xor(z, 27);

    // 4. Undo multiplication by C2
    z = z.wrapping_mul(INV_C2);

    // 5. Undo initial xorshift: z ^ (z >> 30)
    z = invert_shift_xor(z, 30);

    // 6. Undo initial addition
    z.wrapping_sub(seed).wrapping_sub(C_MAGIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bijective_recovery() {
        let seed = 0x1337BEEF9000CAFE;
        let addr = 0xDEADBEEFBADC0DED;

        let masked = vl_mask(addr, seed);
        let recovered = vl_inverse_mask(masked, seed);

        // The mathematical bijection MUST hold true.
        assert_eq!(
            addr, recovered,
            "vl_mask bijection failed! {} != {}",
            addr, recovered
        );
    }

    #[test]
    fn test_bijective_edges() {
        let seed = 0;
        let cases = [0, 1, u64::MAX, u64::MAX - 1];

        for c in cases.iter() {
            let m = vl_mask(*c, seed);
            let r = vl_inverse_mask(m, seed);
            assert_eq!(*c, r, "Failed edge case {}", c);
        }
    }
}
