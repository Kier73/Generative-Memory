//! Virtual Residue Number System (vRNS) Synthesis
//!
//! Exposes the enhanced 16-prime multichannel fmix64 synthesis engine.

pub mod scalar_synth;

/// The 16 Extended Primes near $ 2^{16} $.
/// 
/// Kept strictly `const` so that LLVM can optimize modulo (`%`) reduction operations 
/// into fast Barrett multiplications at compile time, achieving SIMD-like throughput 
/// without needing explicitly unsafe intrinsics for integer division.
pub const MODULI_16: [u64; 16] = [
    65447, 65449, 65479, 65497, 65519, 65521, 65437, 65423,
    65419, 65413, 65407, 65393, 65381, 65371, 65357, 65353,
];
