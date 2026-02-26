//! Generative Memory mathematical core: zero-allocation topological address spaces mapping.
//!
//! Provides bare-metal topological, geometric, and mathematical guarantees tailored for
//! scaling up to $2^{64}$ elements.

pub mod context;
pub mod driver;
pub mod ffi;
pub mod math;
pub mod physical;
pub mod topology;
pub mod vrns;
