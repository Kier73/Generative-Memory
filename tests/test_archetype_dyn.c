#include "../include/gmem.h"
#include "../src/gmem_internal.h"
#include <stdio.h>
#include <string.h>

// We need to access internal dynamic_files list for white-box testing
// Usually we'd use the public API gmem_archetype_create (which we added to
// internal, but exposed via some header?) For now, let's assume we link against
// the object files that define it.

extern int gmem_archetype_create(gmem_ctx_t ctx, const char *path, int is_dir);

int main() {
  printf("[TEST] Dynamic Archetypes...\n");

  gmem_ctx_t ctx = gmem_create(999);
  // Enable FAT mode
  // (We need to poke internals or use a specific seed that triggers it,
  // but gmem_create initializes archetype based on phase. Let's force it
  // manually if needed)
  ctx->archetype = GMEM_ARCHETYPE_FAT;

  // 1. Create a Dynamic File
  const char *new_file = "/assets/dynamic_texture.png";
  int res = gmem_archetype_create(ctx, new_file, 0);
  if (res != 0) {
    printf("[FAIL] Failed to create dynamic file\n");
    return 1;
  }
  printf("[PASS] gmem_archetype_create returned success\n");

  // 2. List Directory to verify it exists
  gmem_virt_entry_t entries[10];
  size_t count = gmem_archetype_get_entries(ctx, "/assets", entries, 10);

  int found = 0;
  for (size_t i = 0; i < count; i++) {
    printf("  Entry: %s\n", entries[i].name);
    if (strcmp(entries[i].name, "dynamic_texture.png") == 0) {
      found = 1;
      if (entries[i].offset & 0xF000000000000000ULL) {
        printf("[PASS] Verified High Memory Offset: %llX\n", entries[i].offset);
      } else {
        printf("[FAIL] Offset check failed: %llX\n", entries[i].offset);
        return 1;
      }
    }
  }

  if (!found) {
    printf("[FAIL] Newly created file not found in directory listing\n");
    return 1;
  }
  printf("[PASS] Dynamic File Listing\n");

  gmem_destroy(ctx);
  return 0;
}
