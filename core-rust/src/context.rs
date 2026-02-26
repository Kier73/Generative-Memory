//! Generative Memory Context Router
//!
//! The central `GMemContext` object that weaves the topological invariants.
//!
//! The Fundamental Invariant:
//! $\forall \text{addr} \in [0, 2^{64}):$
//! $$
//! \text{fetch(addr)} =
//! \begin{cases}
//! \text{overlay[addr]} & \text{if } \text{addr} \in \text{overlay} \\
//! \text{synthesize(addr, seed)} & \text{otherwise}
//! \end{cases}
//! $$
//!
//! (Note: Mirroring and Morphing topology integration follow in Phase 4).

extern crate alloc;
use crate::driver::persistence::AOFLog;
use crate::physical::overlay::Overlay;
use crate::physical::zmask::ZMask;
use crate::vrns::scalar_synth::synthesize_multichannel_float;
use alloc::vec::Vec;
use std::sync::Mutex;

pub struct GMemContext {
    /// Base seed defining this unqiue structural manifold.
    seed: u64,

    /// Sparse physical memory layer mapping explicit writes.
    overlay: Overlay,

    /// Lock-free hierarchical dirty boundary bitfield.
    zmask: ZMask,

    /// Optional Append-Only File for state persistence.
    pub aof: Mutex<AOFLog>,
}

impl GMemContext {
    /// Create a new Generative Memory context mapped to the given cryptographic seed.
    pub fn new(seed: u64) -> Self {
        GMemContext {
            seed,
            overlay: Overlay::new(),
            zmask: ZMask::new(12), // 4096 values per dirty bit
            aof: Mutex::new(AOFLog::new()),
        }
    }

    /// Get the base seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Total bytes footprint theoretically mapped by overlay.
    pub fn overlay_count(&self) -> usize {
        self.overlay.count()
    }

    /// Activate delta state persistence.
    pub fn persistence_attach(&self, path: &str) -> std::io::Result<()> {
        let mut aof = self.aof.lock().unwrap();
        aof.attach(path)
    }

    /// Fetch a value from the memory context.
    /// This resolves the hardware/software boundary in $O(1)$.
    #[inline(always)]
    pub fn fetch(&self, addr: u64) -> f64 {
        // 1. Hardware-layer: Is it in the explicit user-written overlay?
        if let Some(val) = self.overlay.lookup(addr) {
            return val;
        }

        // 2. Mathematically synthesize from the manifold topological laws.
        synthesize_multichannel_float(addr, self.seed)
    }

    /// Write a value to the memory context.
    /// Overrides the mathematical manifold and soils the structural ZMask.
    #[inline(always)]
    pub fn write(&self, addr: u64, value: f64) {
        self.overlay.insert(addr, value);
        self.zmask.mark_dirty(addr);

        // Push 16-byte tuple to the Zero-Copy AOF if attached
        if let Ok(mut aof) = self.aof.lock() {
            let _ = aof.log_write(addr, value);
        }
    }

    /// Vectorized/Bulk fetch optimizing sequential clean page reads.
    /// Yields a heavily accelerated synthesis loop without hashtable checks
    /// if the ZMask reports the target page footprint is pristine.
    pub fn fetch_bulk(&self, start_addr: u64, count: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(count);
        let mut i = 0;

        while i < count {
            let addr = start_addr.wrapping_add(i as u64);

            if !self.zmask.is_dirty(addr) {
                // EXTREMELY FAST PATH
                // Skip DashMap lookup overhead and directly invoke LLVM unrolled synthesis inline
                result.push(synthesize_multichannel_float(addr, self.seed));
            } else {
                // SLOW PATH
                // Must interrogate the concurrent overlay before falling back to math synthesis
                result.push(self.fetch(addr));
            }
            i += 1;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariant_fetch_write() {
        let ctx = GMemContext::new(0x1337);
        let addr = 0xABCDEF;

        // Original synthetic fetch
        let origin = ctx.fetch(addr);
        assert!(origin >= 0.0 && origin < 1.0);

        // Write overlay
        ctx.write(addr, 42.42);

        // Invariant holds
        assert_eq!(ctx.fetch(addr), 42.42);

        // Adjacent fetches still purely synthetic
        assert_ne!(ctx.fetch(addr + 1), 42.42);
    }

    #[test]
    fn test_bulk_fetch_optimization() {
        let ctx = GMemContext::new(0x1337);
        let count = 100_000;

        // Bulk fetch from a completely clean page
        let bulk = ctx.fetch_bulk(0, count);
        assert_eq!(bulk.len(), count);
        assert_eq!(bulk[0], ctx.fetch(0));

        // Dirty a page at 50000
        ctx.write(50000, 99.99);
        let dirty_bulk = ctx.fetch_bulk(49990, 20);
        assert_eq!(dirty_bulk[10], 99.99);
    }
}
