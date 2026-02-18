#include "../include/gmem.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Mock Internal Structures for verification
typedef struct {
  void *ctx;
  size_t size;
  uint32_t magic;
  // Padding might exist here due to alignment
} gmem_alloc_header_t;

#define GMEM_MAGIC 0x474D454D
#define GMEM_ALIGNMENT 16

extern void *g_malloc(size_t size);
extern void g_free(void *ptr);
extern float g_get_f32(void *ptr, size_t index);

int main() {
  printf("[TEST] Allocator Safety & Alignment...\n");

  size_t payload_size = 1024;
  void *ptr = g_malloc(payload_size);

  // 1. Check Alignment
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % GMEM_ALIGNMENT != 0) {
    printf("[FAIL] Pointer %p is not aligned to %d bytes\n", ptr,
           GMEM_ALIGNMENT);
    return 1;
  }
  printf("[PASS] Alignment Check\n");

  // 2. Check Header Integrity (Reverse Lookup)
  // We replicate the logic from g_free/g_get_f32 to find the header
  size_t header_struct_size = sizeof(gmem_alloc_header_t);
  size_t aligned_header_size =
      (header_struct_size + GMEM_ALIGNMENT - 1) & ~(GMEM_ALIGNMENT - 1);

  gmem_alloc_header_t *header =
      (gmem_alloc_header_t *)((char *)ptr - aligned_header_size);

  if (header->magic != GMEM_MAGIC) {
    printf("[FAIL] Magic Number mismatch. Expected %X, Got %X\n", GMEM_MAGIC,
           header->magic);
    return 1;
  }
  printf("[PASS] Header Magic Check\n");

  if (header->size != payload_size) {
    printf("[FAIL] Size mismatch. Expected %zu, Got %zu\n", payload_size,
           header->size);
    return 1;
  }
  printf("[PASS] Header Metadata Check\n");

  // 3. functional Test
  float val = g_get_f32(ptr, 0);
  printf("[INFO] Fetched Value: %f\n", val);

  g_free(ptr);
  printf("[PASS] Free Operation\n");

  return 0;
}
