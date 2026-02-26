//! Sparse Physical Overlay
//!
//! Generative Memory is $2^{64}$ vast, but physical memory is bounded.
//! The `Overlay` sits on top of the mathematical manifold. Reads check the
//! overlay first. If $x \notin Overlay$, the engine synthesizes the value.
//!
//! We utilize `dashmap` to provide a highly concurrent, lock-free sharded hashtable,
//! eliminating the catastrophic `threading.Lock()` overhead found in the Python prototype.

use dashmap::DashMap;

pub struct Overlay {
    /// A lock-free sharded concurrent map.
    /// Maps 64-bit addresses to 64-bit IEEE floats.
    table: DashMap<u64, f64>,
}

impl Default for Overlay {
    fn default() -> Self {
        Self::new()
    }
}

impl Overlay {
    pub fn new() -> Self {
        Overlay {
            // Initialize with capacity heuristics suitable for typical sparse data
            table: DashMap::with_capacity(1_000_000),
        }
    }

    /// Retrieve a value from the physical overlay if it was explicitly written.
    #[inline(always)]
    pub fn lookup(&self, addr: u64) -> Option<f64> {
        self.table.get(&addr).map(|v| *v)
    }

    /// Insert a value into the physical overlay, shadowing the mathematical manifold.
    #[inline(always)]
    pub fn insert(&self, addr: u64, value: f64) {
        self.table.insert(addr, value);
    }

    /// Total number of explicitly written memory cells.
    pub fn count(&self) -> usize {
        self.table.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_concurrency_surface() {
        let overlay = Overlay::new();
        assert_eq!(overlay.lookup(0), None);

        overlay.insert(0xDEADBEEF, 42.0);

        assert_eq!(overlay.lookup(0xDEADBEEF), Some(42.0));
        assert_eq!(overlay.count(), 1);
    }
}
