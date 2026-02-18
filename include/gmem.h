#ifndef GMEM_V1_H
#define GMEM_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Handle to a Generative Memory Context.
 * Stores the synthetic addressing state and master procedural seed.
 */
typedef struct gmem_context *gmem_ctx_t;

/**
 * Create a new Generative Memory Context.
 * @param seed The master seed for deterministic procedural synthesis.
 * @return Handle to the context, or NULL on failure.
 */
gmem_ctx_t gmem_create(uint64_t seed);

/**
 * Destroy a Generative Memory Context and free resources.
 */
void gmem_destroy(gmem_ctx_t ctx);

/**
 * Fetch a single 32-bit floating point value from the synthetic address space.
 * This operation is O(1) and synthesized on-demand.
 * @param ctx Valid GMC context.
 * @param virtual_addr The synthetic address to resolve.
 * @return The synthesized value in range [0.0, 1.0].
 */
float gmem_fetch_f32(gmem_ctx_t ctx, uint64_t virtual_addr);

/**
 * Write a value to the synthetic address space.
 * This value is stored in a sparse physical overlay.
 */
void gmem_write_f32(gmem_ctx_t ctx, uint64_t virtual_addr, float value);

/**
 * Fetch from a monotonic (presorted) synthetic address view.
 * @param index The rank/index in the sorted manifold.
 */
float gmem_fetch_monotonic_f32(gmem_ctx_t ctx, uint64_t index);

/**
 * Perform a predictive (interpolation) search for a target value.
 * Leveraging the monotonic nature of the synthetic manifold.
 * @return The synthetic address (index) closest to the target.
 */
uint64_t gmem_search_f32(gmem_ctx_t ctx, float target);

/**
 * Persist the modified overlay state to disk.
 * @param path File path to save the state delta.
 * @return 0 on success, non-zero on error.
 */
int gmem_save_overlay(gmem_ctx_t ctx, const char *path);

/**
 * Load a previously saved overlay state from disk.
 */
int gmem_load_overlay(gmem_ctx_t ctx, const char *path);

/**
 * Fetch a bulk range of synthesized data into a destination buffer.
 * Optimized for high-throughput procedural materialization.
 */
void gmem_fetch_bulk_f32(gmem_ctx_t ctx, uint64_t start_addr, float *buffer,
                         size_t count);

// --- SPARSE ALLOCATOR INTERFACE (Malloc-Wrapper) ---

/**
 * Allocate a sparse virtual buffer of 'size' bytes.
 * This buffer is backed by a dedicated GVM context.
 * @return A handle (pointer) to the sparse allocation.
 */
void *g_malloc(size_t size);

/**
 * Free a sparse virtual buffer.
 */
void g_free(void *ptr);

/**
 * Optimized element fetch for g_malloc allocated buffers.
 */
float g_get_f32(void *ptr, size_t index);

/**
 * Mirroring Modes
 */
typedef enum {
  GMEM_MIRROR_IDENTITY = 0,  // Shadow exact seed and overlay
  GMEM_MIRROR_TRANSFORM = 1, // Reserved for variety morphing
} gmem_mirror_mode_t;

/**
 * Variety Morphing Modes
 */
typedef enum {
  GMEM_MORPH_IDENTITY = 0,
  GMEM_MORPH_LINEAR = 1, // y = ax + b
  GMEM_MORPH_ADD = 2,    // y = x + b
  GMEM_MORPH_MUL = 3,    // y = x * a
} gmem_morph_mode_t;

typedef struct {
  float a;
  float b;
} gmem_morph_params_t;

/**
 * Attach a source context to a target context for Variety Morphing.
 * The target context will compute its values as a real-time transform of the
 * source.
 */
void gmem_morph_attach(gmem_ctx_t target, gmem_ctx_t source,
                       gmem_morph_mode_t mode, gmem_morph_params_t params);

/**
 * Attach a shadow context to a source context for mirroring.
 */
void gmem_mirror_attach(gmem_ctx_t shadow, gmem_ctx_t source,
                        gmem_mirror_mode_t mode);

/**
 * Detach mirroring from a context.
 */
void gmem_mirror_detach(gmem_ctx_t ctx);

/**
 * Variety Archetypes
 */
typedef enum {
  GMEM_ARCHETYPE_RAW = 0,
  GMEM_ARCHETYPE_FAT = 1, // Virtual FAT-like structure
} gmem_archetype_t;

/**
 * Configure the semantic archetype for a context.
 */
void gmem_set_archetype(gmem_ctx_t ctx, gmem_archetype_t archetype);

/**
 * Attach a physical file for persistent modification logging (AOF).
 * @param path Path to the .gvm_delta file.
 * @return 0 on success, non-zero on error.
 */
int gmem_persistence_attach(gmem_ctx_t ctx, const char *path);

#ifdef __cplusplus
}
#endif

#endif // GMEM_V1_H
