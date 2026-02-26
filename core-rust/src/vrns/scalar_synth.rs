//! Scalar version of the 16-prime synthesis.
//! 
//! Because `MODULI_16` is a known static array at compile time, LLVM's optimizer
//! will automatically unroll the loop, compute magic multiplication constants for 
//! each prime (to avoid expensive `div`/`mod` CPU instructions), and auto-vectorize 
//! the 64-bit integer processing into AVX2 registers.

use crate::math::hashing::fmix64;
use super::MODULI_16;

/// Standard multi-channel fmix64 synthesis step.
/// 
/// Returns a high-entropy pseudo-random `u64` representing the deterministic state 
/// generated from evaluating 16 parallel residue channels over the $2^{64}$ address space.
///
/// Mathematical Model:
/// $ H = \bigoplus_{i=0}^{15} \text{fmix64} \left( (\text{val} \pmod{p_i}) \lor (p_i \ll 16) \lor (i \ll 32) \right) $
///
/// # Arguments
/// * `addr` - The 64-bit virtual address.
/// * `seed` - The 64-bit context base seed.
///
/// # Returns
/// A full-entropy 64-bit hash. To get a $[0, 1)$ float, divide by $2^{64}-1$.
#[inline(always)]
pub fn synthesize_multichannel_u64(addr: u64, seed: u64) -> u64 {
    let x = addr ^ seed;
    let mut h = 0u64;
    
    // The compiler unrolls this loop completely.
    for i in 0..16 {
        let p = MODULI_16[i];
        
        // Fast constant-time modulo (LLVM computes the magic inverse multiplier)
        let residue = x % p;
        
        let channel = residue | (p << 16) | ((i as u64) << 32);
        
        h ^= fmix64(channel);
    }
    h
}

/// Helper: Returns the $[0, 1)$ floating point projection.
#[inline(always)]
pub fn synthesize_multichannel_float(addr: u64, seed: u64) -> f64 {
    let h = synthesize_multichannel_u64(addr, seed);
    // 18446744073709551615.0 == 2^64 - 1
    (h as f64) / 18446744073709551615.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism_and_bounds() {
        let h1 = synthesize_multichannel_u64(0xBEEFCACE, 0x1337);
        let h2 = synthesize_multichannel_u64(0xBEEFCACE, 0x1337);
        assert_eq!(h1, h2, "Synthesis is not purely deterministic");
        
        let f1 = synthesize_multichannel_float(0xBEEFCACE, 0x1337);
        assert!(f1 >= 0.0 && f1 < 1.0, "Float projection out of bounds");
    }
}
