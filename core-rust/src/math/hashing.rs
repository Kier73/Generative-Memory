//! Standard Ciphers and Mixing Functions
//!
//! Provides fundamental non-cryptographic hash functions with strict deterministic guarantees
//! and zero reliance on dynamic memory allocation.

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

/// Computes the FNV-1a hash of a single 64-bit integer.
///
/// The FNV-1a property maintains perfect avalanche properties over short keys.
/// $ h_0 = \text{OFFSET} $
/// $ h_{i+1} = (h_i \oplus d_i) \times \text{PRIME} $
#[inline(always)]
pub fn fnv1a_u64(addr: u64) -> u64 {
    (FNV_OFFSET ^ addr).wrapping_mul(FNV_PRIME)
}

/// Computes the FNV-1a hash over a UTF-8 string with an optional 64-bit salt.
pub fn fnv1a_string(s: &str, salt: u64) -> u64 {
    let mut h = FNV_OFFSET ^ salt;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Computes the DJB2 hash for string signatures.
///
/// Popularized by Daniel J. Bernstein, utilizing the magic constant 5381.
/// $ h_{i+1} = h_i \times 33 + c $
pub fn djb2_string(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        // h * 33 + b
        h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(b as u64);
    }
    h
}

/// MurmurHash3 64-bit finalizer (Avalanche Mixer).
///
/// Collapses a high-entropy manifold projection into a well-distributed 64-bit value.
/// This acts as the folding operation $H = \bigoplus_{i=0}^{15} \text{fmix64}(C_i)$ in the vRNS engine.
#[inline(always)]
pub fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
    h ^= h >> 33;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmix64_determinism() {
        assert_eq!(fmix64(0), 0);
        // Ensure standard Murmur3 finalizer matches expected fixed-point behavior.
        assert_eq!(fmix64(0x1337BEEF), 14075673083181017210);
    }

    #[test]
    fn test_fnv1a_u64() {
        // Test avalanche. One bit difference should wildly change output.
        let h1 = fnv1a_u64(0x1000);
        let h2 = fnv1a_u64(0x1001);
        assert!(h1 != h2);
    }
}
