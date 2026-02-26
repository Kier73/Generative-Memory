//! Goldilocks Prime Field Arithmetic
//!
//! Generative Memory uses the Goldilocks Prime $ P = 2^{64} - 2^{32} + 1 $ to perform
//! exact law composition using Number Theoretic Transform constraints.
//!
//! Algebraic identities:
//! - Composition: $ L_C = (S_A \times S_B) \pmod P $
//! - Decompilation: $ S_B = (L_C \times S_A^{-1}) \pmod P $

/// The Goldilocks Prime: $ 2^{64} - 2^{32} + 1 $
pub const P_GOLDILOCKS: u64 = 0xFFFFFFFF00000001;

/// Add two values in the Goldilocks Field.
/// $ R = (a + b) \pmod P $
/// We carefully manage 64-bit overflow by promoting to u128.
#[inline(always)]
pub fn add_mod(a: u64, b: u64) -> u64 {
    let res = (a as u128) + (b as u128);
    (res % (P_GOLDILOCKS as u128)) as u64
}

/// Subtract two values in the Goldilocks Field.
/// $ R = (a - b) \pmod P $
#[inline(always)]
pub fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        // P - (b - a)
        P_GOLDILOCKS - (b - a)
    }
}

/// Multiply two values in the Goldilocks Field.
/// $ R = (a \times b) \pmod P $
#[inline(always)]
pub fn multiply_mod(a: u64, b: u64) -> u64 {
    let res = (a as u128) * (b as u128);
    (res % (P_GOLDILOCKS as u128)) as u64
}

/// Computes the modular inverse $ a^{-1} \pmod P $.
/// 
/// Utilizes the Extended Euclidean Algorithm since elements are bounded strictly to `u64`.
pub fn mod_inverse(a: u64) -> u64 {
    let mut t = 0i128;
    let mut new_t = 1i128;
    let mut r = P_GOLDILOCKS as i128;
    let mut new_r = (a % P_GOLDILOCKS) as i128;

    if new_r == 0 {
        return 0; // The generator prevents passing 0, fallback.
    }

    while new_r != 0 {
        let quotient = r / new_r;

        let temp_t = t;
        t = new_t;
        new_t = temp_t - quotient * new_t;

        let temp_r = r;
        r = new_r;
        new_r = temp_r - quotient * new_r;
    }

    if t < 0 {
        t += P_GOLDILOCKS as i128;
    }

    t as u64
}

/// Create a synthesized product law $ L_{A \otimes B} $ from two parent seeds.
#[inline(always)]
pub fn synthesize_product_law(seed_a: u64, seed_b: u64) -> u64 {
    multiply_mod(seed_a, seed_b)
}

/// Recover the unknown parent $ S_B $ from product law $ L_{A \otimes B} $ and known $ S_A $.
#[inline(always)]
pub fn inverse_product_law(product_seed: u64, known_parent_a_seed: u64) -> u64 {
    let inv_a = mod_inverse(known_parent_a_seed);
    multiply_mod(product_seed, inv_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goldilocks_arithmetic() {
        let a = 0xFFFFFFFF;
        let b = 2;
        let sum = add_mod(a, b);
        let sub = sub_mod(b, a); // 2 - 0xFFFFFFFF == P - 0xFFFFFFFD
        let mul = multiply_mod(a, b);

        assert_eq!(sum, a + b); // Sub-prime addition stays standard
        assert!(sub < P_GOLDILOCKS);
        assert_eq!(mul, ((a as u128 * b as u128) % P_GOLDILOCKS as u128) as u64);
    }

    #[test]
    fn test_law_composition_recovery() {
        let seed_a = 0xBAADF00DCAFEBABE;
        let seed_b = 0x8BADF00DDEADBEEF;

        // Algebraic composition: law_C = A * B mod P
        let product = synthesize_product_law(seed_a, seed_b);

        // Decompilation: Recover B given C and A
        let recovered_b = inverse_product_law(product, seed_a);

        // Verification of mathematical inverse stability
        assert_eq!(
            seed_b % P_GOLDILOCKS,
            recovered_b,
            "Goldilocks Extended Euclidean Inverse failed to safely reconstruct the seed"
        );
    }
}
