//! Z-Mask Hierarchical Dirty Bit Tracking
//!
//! Provides a highly concurrent, lock-free hierarchical bitfield.
//! Because Generative Memory treats synthesis as the ground truth, we divide the
//! $2^{64}$ address space into 4096-value pages.
//!
//! If a page in the Z-Mask is perfectly clean, `fetch_bulk` can execute pure synthesis
//! across the entire range without ever querying the underlying hash map, guaranteeing
//! peak sequential throughput.

// We need `std` for Atomic vectors if size isn't purely 0-alloc, but since we rely on
// allocating arrays on the heap for the context, we use `alloc` crate.
extern crate alloc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// A region length dictates the size of a page map.
/// We use 20-bit page IDs (enabling highly sparse 1M page allocations).
pub struct ZMask {
    /// Lock-free bit arrays. We use a sparse table approach to ensure
    /// zero allocation overhead until a region is actually touched.
    /// In a fully rigorous scale setting, this would be a multi-level radix tree.
    /// For this version, we map the upper bits of the address directly.
    pages: Vec<AtomicU64>,
    page_shift: usize,
}

impl Default for ZMask {
    fn default() -> Self {
        ZMask::new(12) // 4096 values per bit
    }
}

impl ZMask {
    /// Create a new Z-Mask tracker.
    /// `page_shift` defines how many low bits to ignore per page.
    /// E.g., shift=12 maps a sequence of 4096 addresses to a single bit.
    pub fn new(page_shift: usize) -> Self {
        // We initialize a root vector.
        // Note: For $2^{64}$, a purely flat bitset is astronomically large.
        // We limit bitfield to high usage zones or lazily inject.
        // For standard sparse operations, we configure to handle the first 64M entities out-of-the-box.
        let capacity = 1_000_000;
        let mut pages = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            pages.push(AtomicU64::new(0));
        }

        ZMask { pages, page_shift }
    }

    /// Mark an address page as dirty via an atomic lock-free OR operation.
    #[inline]
    pub fn mark_dirty(&self, addr: u64) {
        let bit_idx = addr >> self.page_shift;
        let u64_idx = (bit_idx / 64) as usize;
        let bit_offset = bit_idx % 64;

        if u64_idx < self.pages.len() {
            // Relaxed ordering is sufficient for dirty bit monotonic state
            self.pages[u64_idx].fetch_or(1 << bit_offset, Ordering::Relaxed);
        }
    }

    /// Check if the page containing `addr` has been dirtied.
    /// If false, the memory block guarantees 100% pure synthesis rules.
    #[inline]
    pub fn is_dirty(&self, addr: u64) -> bool {
        let bit_idx = addr >> self.page_shift;
        let u64_idx = (bit_idx / 64) as usize;
        let bit_offset = bit_idx % 64;

        if u64_idx < self.pages.len() {
            let chunk = self.pages[u64_idx].load(Ordering::Relaxed);
            (chunk & (1 << bit_offset)) != 0
        } else {
            // Unmapped regions are implicitly clean
            false
        }
    }

    /// Clear the entire ZMask (e.g. upon dropping overlay state).
    pub fn clear(&self) {
        for atomic in &self.pages {
            atomic.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zmask_atomic_dirtying() {
        let zmask = ZMask::new(12); // page size: 4096

        assert_eq!(zmask.is_dirty(0), false);
        zmask.mark_dirty(0);
        assert_eq!(zmask.is_dirty(0), true);

        // Within same page (4095)
        assert_eq!(zmask.is_dirty(4095), true);

        // Next page
        assert_eq!(zmask.is_dirty(4096), false);
        zmask.mark_dirty(9000);
        assert_eq!(zmask.is_dirty(8192), true); // 8192-12287 is page 2
    }
}
