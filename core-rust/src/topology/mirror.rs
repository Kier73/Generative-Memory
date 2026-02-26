//! Mirror (Shadow Context) Topology
//!
//! Shadow contexts operate as virtual symlinks to an underlying context.
//! Because Generative Memory contexts are fundamentally just parameters (Seed, Overlay, ZMask),
//! we can orchestrate arbitrary graph-based hierarchies via `alloc::sync::Arc` with
//! strictly zero duplication of physical RAM footprint.

extern crate alloc;
use crate::context::GMemContext;
use alloc::sync::Arc;

/// Holder for the active shadow mirror connection.
pub struct MirrorState {
    pub source: Option<Arc<GMemContext>>,
    pub mode: u8, // 0 = exact identity shadow
}

impl Default for MirrorState {
    fn default() -> Self {
        MirrorState {
            source: None,
            mode: 0,
        }
    }
}

impl MirrorState {
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.source.is_some()
    }
}
