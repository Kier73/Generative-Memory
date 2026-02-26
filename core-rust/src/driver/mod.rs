//! Hard Drive Persistence and Logging Drivers
//!
//! Enables zero-copy memory-mapped operations for preserving context states
//! across reboots via AOF (Append-Only File) delta logs and bulk snapshotting.

pub mod persistence;
