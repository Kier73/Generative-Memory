//! C-FFI API for Generative Memory
//!
//! Exposes a highly optimized `extern "C"` ABI making the Rust mathematical engine
//! linkable to Python (via `ctypes`), C++, and other systems.

use crate::context::GMemContext;

/// C-FFI Handle to the underlying Generative Memory context.
/// We use an opaque struct pointer across the FFI boundary.
#[repr(C)]
pub struct CGMemContext {
    _private: [u8; 0],
}

/// Create a new topological memory context.
///
/// Returns a raw pointer to the dynamically allocated `GMemContext`.
/// The caller MUST call `gmem_context_free` to avoid memory leaks.
#[no_mangle]
pub extern "C" fn gmem_context_new(seed: u64) -> *mut CGMemContext {
    let ctx = Box::new(GMemContext::new(seed));
    Box::into_raw(ctx) as *mut CGMemContext
}

/// Fetch a 64-bit IEEE float from the synthetic address space.
#[no_mangle]
pub extern "C" fn gmem_fetch(ctx_ptr: *const CGMemContext, addr: u64) -> f64 {
    if ctx_ptr.is_null() {
        return 0.0;
    }
    let ctx = unsafe { &*(ctx_ptr as *const GMemContext) };
    ctx.fetch(addr)
}

/// Write an explicit physical override, dirtying the topological manifold.
#[no_mangle]
pub extern "C" fn gmem_write(ctx_ptr: *mut CGMemContext, addr: u64, value: f64) {
    if ctx_ptr.is_null() {
        return;
    }
    let ctx = unsafe { &*(ctx_ptr as *const GMemContext) };
    ctx.write(addr, value);
}

/// Get the number of explicitly mapped hardware entries.
#[no_mangle]
pub extern "C" fn gmem_overlay_count(ctx_ptr: *const CGMemContext) -> usize {
    if ctx_ptr.is_null() {
        return 0;
    }
    let ctx = unsafe { &*(ctx_ptr as *const GMemContext) };
    ctx.overlay_count()
}

/// Free the context memory.
#[no_mangle]
pub extern "C" fn gmem_context_free(ctx_ptr: *mut CGMemContext) {
    if !ctx_ptr.is_null() {
        unsafe {
            // Re-take ownership to drop cleanly
            let _ = Box::from_raw(ctx_ptr as *mut GMemContext);
        }
    }
}

use std::ffi::CStr;
use std::os::raw::c_char;

/// Mount the physical memory overlay directly to an Append-Only File (AOF).
#[no_mangle]
pub extern "C" fn gmem_persistence_attach(
    ctx_ptr: *mut CGMemContext,
    path_ptr: *const c_char,
) -> i32 {
    if ctx_ptr.is_null() || path_ptr.is_null() {
        return 0; // Failure
    }

    let ctx = unsafe { &mut *(ctx_ptr as *mut GMemContext) };

    let c_str = unsafe { CStr::from_ptr(path_ptr) };
    if let Ok(path_str) = c_str.to_str() {
        if ctx.persistence_attach(path_str).is_ok() {
            return 1; // Success
        }
    }
    0 // Failure
}

/// Populates a contiguous block of memory directly from the virtual manifold.
/// This allows Python (via NumPy/PyTorch) to instantiate a matrix and let Rust fill it
/// instantly over the C-FFI, bypassing scalar Python loop overhead for billion-parameter layers.
#[no_mangle]
pub extern "C" fn gmem_fill_tensor(
    ctx_ptr: *const CGMemContext,
    out_ptr: *mut f64,
    size: usize,
    start_addr: u64,
) {
    if ctx_ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let ctx = unsafe { &*(ctx_ptr as *const GMemContext) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, size) };

    // High-speed sequential extraction directly onto the provided C-pointer
    for i in 0..size {
        out_slice[i] = ctx.fetch(start_addr + i as u64);
    }
}

use crate::topology::space_filling::{hilbert_decode, hilbert_encode};

/// FFI Export: Encodes 2D coordinates into a 1D Hilbert string index.
#[no_mangle]
pub extern "C" fn gmem_hilbert_encode(i: u64, j: u64, order: u32) -> u64 {
    hilbert_encode(i, j, order)
}

/// FFI Export: Decodes a 1D Hilbert string index back to 2D coordinates.
/// To return two `u64` via C-FFI smoothly, we accept out-pointers.
#[no_mangle]
pub extern "C" fn gmem_hilbert_decode(d: u64, order: u32, out_i: *mut u64, out_j: *mut u64) {
    let (i, j) = hilbert_decode(d, order);
    if !out_i.is_null() {
        unsafe {
            *out_i = i;
        }
    }
    if !out_j.is_null() {
        unsafe {
            *out_j = j;
        }
    }
}

use crate::topology::anchor::AnchorNavigator;
use ndarray::Array2;

use crate::math::gielis::GielisGPUCompiler;
use crate::math::koopman::koopman_sonar;

/// FFI Export: Executes the Koopman Spectral Sonar over a 1D Hilbert float array.
/// Returns the dominant angular frequency and populates a 5-element array with the continuous Beta regression.
#[no_mangle]
pub extern "C" fn gmem_koopman_sonar(
    signal_ptr: *const f64,
    size: usize,
    out_beta: *mut f64,
) -> f64 {
    let signal = unsafe { std::slice::from_raw_parts(signal_ptr, size) };
    let state = koopman_sonar(signal);
    if !out_beta.is_null() {
        let beta_slice = unsafe { std::slice::from_raw_parts_mut(out_beta, 5) };
        beta_slice.copy_from_slice(&state.beta);
    }
    state.omega
}

/// FFI Export: Executes the Gielis Lattice Lock against 80,000 Ghost Topologies natively in VRAM.
/// Returns 1 on success, 0 on failure (e.g. no CUDA device).
/// Populates the out variables with (m, n1, n2, n3).
#[no_mangle]
pub extern "C" fn gmem_gielis_search(
    signal_ptr: *const f32,
    size: usize,
    freq: f32,
    out_m: *mut f32,
    out_n1: *mut f32,
    out_n2: *mut f32,
    out_n3: *mut f32,
) -> i32 {
    let signal = unsafe { std::slice::from_raw_parts(signal_ptr, size) };

    if let Ok(compiler) = GielisGPUCompiler::new() {
        if let Ok((m, n1, n2, n3)) = compiler.search(signal, freq) {
            unsafe {
                if !out_m.is_null() {
                    *out_m = m;
                }
                if !out_n1.is_null() {
                    *out_n1 = n1;
                }
                if !out_n2.is_null() {
                    *out_n2 = n2;
                }
                if !out_n3.is_null() {
                    *out_n3 = n3;
                }
            }
            return 1;
        }
    }
    0
}

/// Instantiates the Semantic Compiler (Isomorphic Anchor Projection)
#[no_mangle]
pub extern "C" fn gmem_anchor_new(
    a_ptr: *const f64,
    a_rows: usize,
    a_cols: usize,
    b_ptr: *const f64,
    b_rows: usize,
    b_cols: usize,
    s: usize,
) -> *mut AnchorNavigator {
    // Unsafe conversion of raw pointers from Python to ndarray Array2 views
    let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, a_rows * a_cols) };
    let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, b_rows * b_cols) };

    let a_view = Array2::from_shape_vec((a_rows, a_cols), a_slice.to_vec()).unwrap();
    let b_view = Array2::from_shape_vec((b_rows, b_cols), b_slice.to_vec()).unwrap();

    let nav = Box::new(AnchorNavigator::new(&a_view, &b_view, s));
    Box::into_raw(nav)
}

/// Retrieve the synthetically compressed inverse-bound data
#[no_mangle]
pub extern "C" fn gmem_anchor_navigate(nav_ptr: *const AnchorNavigator, i: usize, j: usize) -> f64 {
    if nav_ptr.is_null() {
        return 0.0;
    }
    let nav = unsafe { &*nav_ptr };
    nav.navigate(i, j)
}

/// Free the Semantic Compiler
#[no_mangle]
pub extern "C" fn gmem_anchor_free(nav_ptr: *mut AnchorNavigator) {
    if !nav_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(nav_ptr));
        }
    }
}
