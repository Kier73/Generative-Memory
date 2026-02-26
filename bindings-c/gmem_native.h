// gmem_native.h
// Generative Memory: C-Native OS Mapping Foundation
// Exports the strict mathematical GMemContext opaque pointer across the FFI.

#ifndef GMEM_NATIVE_H
#define GMEM_NATIVE_H

#include <stddef.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the Generative Memory Mathematical Manifold
typedef struct CGMemContext CGMemContext;

// Instantiates a new lock-free virtual space bound by the 64-bit scalar seed.
// O(1) allocation overhead.
CGMemContext *gmem_context_new(uint64_t seed);

// Resolves a 64-bit coordinate into a highly entropic [0, 1) float.
// Mathematically: H = \bigoplus_{i=0}^{15} fmix64( (addr \pmod{p_i}) \lor (p_i
// \ll 16) \lor (i \ll 32) ) / (2^{64}-1)
double gmem_fetch(const CGMemContext *ctx, uint64_t addr);

// Imprints a dirty physical value into the mathematical matrix.
void gmem_write(CGMemContext *ctx, uint64_t addr, double value);

// Returns the number of dirty overrides physically materialized in RAM.
size_t gmem_overlay_count(const CGMemContext *ctx);

// Bulk instantiates a contiguous memory buffer identically to the GPU/PyTorch
// hooks.
void gmem_fill_tensor(const CGMemContext *ctx, double *out_ptr, size_t size,
                      uint64_t start_addr);

// Frees the context and its internal lock-free trees.
void gmem_context_free(CGMemContext *ctx);

// --------------------------------------------------------------------------
// Semantic Compiler anchors (SVD Morphological constraints)
// K \in \mathbb{R}^{m \times s}, R \in \mathbb{R}^{s \times n}, W^{-1} \in
// \mathbb{R}^{s \times s}
// --------------------------------------------------------------------------
typedef struct AnchorNavigator AnchorNavigator;

AnchorNavigator *gmem_anchor_new(const double *a_ptr, size_t a_rows,
                                 size_t a_cols, const double *b_ptr,
                                 size_t b_rows, size_t b_cols, size_t s);

// val = \sum_{col=0}^{s} \left( \sum_{l=0}^{s} K[i,l] \cdot W^{-1}[l,col]
// \right) \cdot R[col, j]
double gmem_anchor_navigate(const AnchorNavigator *nav, size_t i, size_t j);

void gmem_anchor_free(AnchorNavigator *nav);

#ifdef __cplusplus
}
#endif

#endif // GMEM_NATIVE_H
