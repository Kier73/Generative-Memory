#include "../include/gmem_filter.h"
#include "gmem_internal.h"
#include <stdlib.h>
#include <string.h>

struct gmem_filter {
  uint32_t sector_size;
  uint32_t type_id; // 0=JSON, 1=Block
};

gmem_filter_t gmem_create_block_filter(uint32_t sector_size) {
  gmem_filter_t filter = (gmem_filter_t)malloc(sizeof(struct gmem_filter));
  if (filter) {
    filter->sector_size = sector_size ? sector_size : 4096;
    filter->type_id = 1;
  }
  return filter;
}

/**
 * Xi Mapping: SectorID to Manifold Domain
 */
static uint64_t gmem_xi_resolve(uint64_t sector_id, uint64_t seed) {
  uint64_t d = sector_id ^ seed;
  // Mix to spread sectors across the manifold
  d *= 0xBF58476D1CE4E5B9ULL;
  d ^= d >> 33;
  d *= 0x94D049BB133111EBULL;
  d ^= d >> 33;
  return d;
}

void gmem_filter_read_block(gmem_filter_t filter, gmem_ctx_t ctx,
                            uint64_t sector_id, void *buffer) {
  if (!filter || !ctx || !buffer || filter->type_id != 1)
    return;

  uint64_t base_d = gmem_xi_resolve(sector_id, ctx->seed);

  // Fill buffer with synthetic data from the manifold
  // Since GMC uses floats, we translate bits directly for the block filter
  float *f_buf = (float *)buffer;
  size_t float_count = filter->sector_size / sizeof(float);

  for (size_t i = 0; i < float_count; i++) {
    f_buf[i] = gmem_fetch_f32(ctx, base_d + i);
  }
}
