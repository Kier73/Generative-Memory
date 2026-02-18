#include "../include/gmem_filter.h"
#include <stdlib.h>
#include <string.h>

struct gmem_filter {
  char type[32];
  uint64_t salt;
};

gmem_filter_t gmem_create_json_filter(const char *schema_name) {
  gmem_filter_t filter = (gmem_filter_t)malloc(sizeof(struct gmem_filter));
  if (filter) {
    strncpy(filter->type, "JSON", 32);
    // Salt the manifold based on schema name for uniqueness
    filter->salt = 0xDEADC0DE;
    if (schema_name) {
      for (int i = 0; schema_name[i]; i++) {
        filter->salt = (filter->salt * 31) + schema_name[i];
      }
    }
  }
  return filter;
}

void gmem_destroy_filter(gmem_filter_t filter) {
  if (filter)
    free(filter);
}

/**
 * FNV-1a based Path Hashing (Topological Projection)
 * Inspired by vGPU's recursive binding.
 */
uint64_t gmem_filter_resolve_path(gmem_filter_t filter, const char *path) {
  if (!path)
    return 0;

  uint64_t h = 14695981039346656037ULL;
  h ^= filter->salt;

  for (int i = 0; path[i]; i++) {
    h ^= (uint64_t)path[i];
    h *= 1099511628211ULL;
  }

  return h;
}

float gmem_filter_get_val(gmem_filter_t filter, gmem_ctx_t ctx,
                          const char *path) {
  uint64_t addr = gmem_filter_resolve_path(filter, path);
  return gmem_fetch_f32(ctx, addr);
}

void gmem_filter_set_val(gmem_filter_t filter, gmem_ctx_t ctx, const char *path,
                         float value) {
  uint64_t addr = gmem_filter_resolve_path(filter, path);
  gmem_write_f32(ctx, addr, value);
}
