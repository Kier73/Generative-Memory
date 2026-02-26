//! Physical Memory Mapping Subsystem
//!
//! Provides the data structures that map the $2^{64}$ address space to underlying
//! sparse physical RAM through lock-free overlays and dirty-page masks.

pub mod overlay;
pub mod zmask;
