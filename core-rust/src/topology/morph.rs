//! Morph (Real-Time Transformation) Topology
//!
//! Provides the mathematical mechanics to warp the resolution of a Context
//! on the fly via linear affine transformations or arithmetic shifts.

/// The transformation algorithm applied to the topological fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MorphMode {
    Identity = 0,
    Linear = 1,
    Add = 2,
    Mul = 3,
}

impl Default for MorphMode {
    fn default() -> Self {
        MorphMode::Identity
    }
}

/// Holds the active topological morph constraints.
///
/// We avoid virtual dispatch (`dyn Trait`) to strictly enforce $O(1)$ zero-cost execution
/// on the hot-path.
pub struct MorphState {
    pub active: bool,
    pub mode: MorphMode,
    pub a: f64,
    pub b: f64,
}

impl Default for MorphState {
    fn default() -> Self {
        MorphState {
            active: false,
            mode: MorphMode::Identity,
            a: 1.0,
            b: 0.0,
        }
    }
}

impl MorphState {
    /// Applies the morph constraints in $O(1)$.
    #[inline(always)]
    pub fn transform(&self, base_val: f64) -> f64 {
        if !self.active {
            return base_val;
        }

        match self.mode {
            MorphMode::Identity => base_val,
            MorphMode::Linear => base_val * self.a + self.b,
            MorphMode::Add => base_val + self.b,
            MorphMode::Mul => base_val * self.a,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morph_transformations() {
        let mut morph = MorphState {
            active: true,
            mode: MorphMode::Linear,
            a: 10.0,
            b: -5.0,
        };

        // Linear: 10 * 0.5 - 5.0 = 0.0
        assert_eq!(morph.transform(0.5), 0.0);

        morph.mode = MorphMode::Mul;
        // Mul: 10 * 0.5 = 5.0
        assert_eq!(morph.transform(0.5), 5.0);

        morph.mode = MorphMode::Add;
        // Add: 0.5 - 5.0 = -4.5
        assert_eq!(morph.transform(0.5), -4.5);
    }
}
