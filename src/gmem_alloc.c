#include "../include/gmem.h"
#include <stdint.h>
#include <stdlib.h>

/**
 * Internal header for g_malloc handles.
 * Stores the context and size of the sparse allocation.
 */
typedef struct {
  gmem_ctx_t ctx;
  size_t size;
  uint64_t magic;
} gmem_alloc_header_t;

#define GMEM_MAGIC 0x474D454D // "GMEM"

#define GMEM_ALIGNMENT 16

void *g_malloc(size_t size) {
  // 1. Create a dedicated GVM context for this allocation
  gmem_ctx_t ctx = gmem_create((uint64_t)size ^ 0xFEEDFACE);
  if (!ctx)
    return NULL;

  // 2. Allocate the header + payload
  // We align the header size to ensure the payload is aligned
  size_t header_size = sizeof(gmem_alloc_header_t);
  size_t aligned_header_size =
      (header_size + GMEM_ALIGNMENT - 1) & ~(GMEM_ALIGNMENT - 1);

  // Create the physical allocation
  char *raw_ptr = (char *)malloc(aligned_header_size + size);
  if (!raw_ptr) {
    gmem_destroy(ctx);
    return NULL;
  }

  gmem_alloc_header_t *header = (gmem_alloc_header_t *)raw_ptr;
  header->ctx = ctx;
  header->size = size;
  header->magic = GMEM_MAGIC;

  // 3. Return the pointer to the payload (after header)
  return (void *)(raw_ptr + aligned_header_size);
}

void g_free(void *ptr) {
  if (!ptr)
    return;

  // Resolve header from payload pointer
  size_t header_size = sizeof(gmem_alloc_header_t);
  size_t aligned_header_size =
      (header_size + GMEM_ALIGNMENT - 1) & ~(GMEM_ALIGNMENT - 1);

  char *raw_ptr = (char *)ptr - aligned_header_size;
  gmem_alloc_header_t *header = (gmem_alloc_header_t *)raw_ptr;

  if (header->magic != GMEM_MAGIC)
    return; // Safety Check: Not a GVM pointer

  gmem_destroy(header->ctx);
  free(raw_ptr); // Free the underlying allocation
}

float g_get_f32(void *ptr, size_t index) {
  if (!ptr)
    return 0.0f;

  // Resolve header from payload pointer
  size_t header_size = sizeof(gmem_alloc_header_t);
  size_t aligned_header_size =
      (header_size + GMEM_ALIGNMENT - 1) & ~(GMEM_ALIGNMENT - 1);

  char *raw_ptr = (char *)ptr - aligned_header_size;
  gmem_alloc_header_t *header = (gmem_alloc_header_t *)raw_ptr;

  if (header->magic != GMEM_MAGIC)
    return 0.0f;

  // Fetch from the associated GVM context
  return gmem_fetch_f32(header->ctx, (uint64_t)index);
}
