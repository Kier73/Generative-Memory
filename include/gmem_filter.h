#ifndef GMEM_FILTER_H
#define GMEM_FILTER_H

#include "gmem.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gmem_filter *gmem_filter_t;

/**
 * Create a filter for JSON structural projection.
 * @param schema_name A name or identifier for the structure type.
 */
gmem_filter_t gmem_create_json_filter(const char *schema_name);

/**
 * Destroy a filter.
 */
void gmem_destroy_filter(gmem_filter_t filter);

/**
 * Map a semantic path to a synthetic manifold address.
 * Example Path: "users[0].profile.email"
 */
uint64_t gmem_filter_resolve_path(gmem_filter_t filter, const char *path);

/**
 * Synthesize a value with structural integrity.
 * Fetches the value at the manifold address resolved from the path.
 */
float gmem_filter_get_val(gmem_filter_t filter, gmem_ctx_t ctx,
                          const char *path);

/**
 * Perform a surgical edit on a structured value.
 */
void gmem_filter_set_val(gmem_filter_t filter, gmem_ctx_t ctx, const char *path,
                         float value);

#ifdef __cplusplus
}
#endif

#endif // GMEM_FILTER_H
