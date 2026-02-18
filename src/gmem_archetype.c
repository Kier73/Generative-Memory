#include "../include/gmem.h"
#include "gmem_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal Hash Helper
static uint64_t fnv1a_hash(const char *str) {
  uint64_t hash = 14695981039346656037ULL;
  while (*str) {
    hash ^= (unsigned char)*str++;
    hash *= 1099511628211ULL;
  }
  return hash;
}

// Helper to get basename
static void get_basename(const char *path, char *out) {
  const char *last_slash = strrchr(path, '/');
  if (last_slash)
    strcpy(out, last_slash + 1);
  else
    strcpy(out, path);
}

// Helper: Get parent path
static void get_parent(const char *path, char *out) {
  const char *last_slash = strrchr(path, '/');
  if (last_slash && last_slash != path) {
    size_t len = last_slash - path;
    strncpy(out, path, len);
    out[len] = '\0';
  } else if (last_slash == path) {
    strcpy(out, "/");
  } else {
    strcpy(out, ".");
  }
}

int gmem_archetype_create(gmem_ctx_t ctx, const char *path, int is_dir) {
  if (!ctx)
    return -1;

  // Check if exists
  gmem_dyn_file_t *curr = ctx->dynamic_files;
  while (curr) {
    if (strcmp(curr->path, path) == 0)
      return 0; // Already exists
    curr = curr->next;
  }

  gmem_dyn_file_t *new_file =
      (gmem_dyn_file_t *)calloc(1, sizeof(gmem_dyn_file_t));
  if (!new_file)
    return -1;

  strncpy(new_file->path, path, 255);
  new_file->is_dir = is_dir;

  // Assign a virtual offset in the "High Memory" area
  new_file->offset =
      0xF000000000000000ULL | (fnv1a_hash(path) & 0xFFFFFFFFFFULL);
  new_file->size = 0;

  new_file->next = ctx->dynamic_files;
  ctx->dynamic_files = new_file;
  return 0;
}

size_t gmem_archetype_get_entries(gmem_ctx_t ctx, const char *path,
                                  gmem_virt_entry_t *entries,
                                  size_t max_entries) {
  if (!ctx || ctx->archetype == GMEM_ARCHETYPE_RAW)
    return 0;

  size_t count = 0;

  // 1. Static/Procedural Entries (Hardcoded for Demo)
  if (strcmp(path, "/") == 0) {
    // Root directory synthesis
    if (ctx->archetype == GMEM_ARCHETYPE_FAT) {
      // Entry 1: Documentation/README
      if (count < max_entries) {
        strcpy(entries[count].name, "README.txt");
        entries[count].offset = 0;
        entries[count].size = 1024;
        entries[count].is_dir = 0;
        count++;
      }
      // Entry 2: Data Volume
      if (count < max_entries) {
        strcpy(entries[count].name, "data_volume.bin");
        entries[count].offset = 1024 * 1024;
        entries[count].size = 1024ULL * 1024 * 1024 * 1024; // 1TB
        entries[count].is_dir = 0;
        count++;
      }
      // Entry 3: Procedural Assets Folder
      if (count < max_entries) {
        strcpy(entries[count].name, "assets");
        entries[count].offset = 2048;
        entries[count].size = 0;
        entries[count].is_dir = 1;
        count++;
      }
    }
  } else if (strcmp(path, "/assets") == 0) {
    // Synthesis of sub-directory
    for (int i = 0; i < 5 && count < max_entries; i++) {
      snprintf(entries[count].name, 32, "asset_%03d.raw", i);
      entries[count].offset = (uint64_t)(i + 1) * 1024ULL * 1024 * 1024;
      entries[count].size = 64 * 1024 * 1024; // 64MB each
      entries[count].is_dir = 0;
      count++;
    }
  }

  // 2. Dynamic Entries (Overlay)
  gmem_dyn_file_t *curr = ctx->dynamic_files;
  while (curr && count < max_entries) {
    char parent[256];
    get_parent(curr->path, parent);

    if (strcmp(parent, path) == 0) {
      get_basename(curr->path, entries[count].name);
      entries[count].offset = curr->offset;
      entries[count].size = curr->size;
      entries[count].is_dir = curr->is_dir;
      count++;
    }
    curr = curr->next;
  }

  return count;
}
