#ifndef GMEM_INTERNAL_H
#define GMEM_INTERNAL_H

#include "../include/gmem.h"
#include <stdio.h>

#include "gmem_os.h"
#include <stdio.h>

#define GMEM_LOCK(ctx) gmem_mutex_lock(&ctx->lock)
#define GMEM_UNLOCK(ctx) gmem_mutex_unlock(&ctx->lock)

typedef struct {
  uint64_t virtual_addr;
  float value;
  int active;
} gmem_entry_t;

// Hierarchical Z-Masking Constants
#define GMEM_TB_SIZE (1024ULL * 1024 * 1024 * 1024 / sizeof(float))
#define GMEM_PAGES_PER_TB                                                      \
  (1024ULL * 1024 * 1024 * 1024 / (1024 * sizeof(float)))

// --- Algebraic Rewrite Optimization (ARO) AST ---

typedef enum {
  GMEM_OP_CONST = 0,
  GMEM_OP_VAR_X = 1,
  GMEM_OP_ADD = 2,
  GMEM_OP_SUB = 3,
  GMEM_OP_MUL = 4,
  GMEM_OP_DIV = 5,
  GMEM_OP_MOD = 6,
  GMEM_OP_XOR = 7
} gmem_op_type_t;

typedef struct gmem_ast_node {
  gmem_op_type_t type;
  float val; // For GMEM_OP_CONST
  struct gmem_ast_node *left;
  struct gmem_ast_node *right;
} gmem_ast_node_t;

#include "../include/gmem_aro.h"

typedef struct {
  uint64_t variety_sig;
  gmem_arr_rule_t rule;
  float p1; // intercept or constant
  float p2; // slope (for linear)
} gmem_law_entry_t;

// Phase 20: Dynamic Archetypes (Forward Declaration)
typedef struct gmem_dyn_file {
  char path[256];
  uint64_t offset;
  uint64_t size;
  int is_dir;
  struct gmem_dyn_file *next;
} gmem_dyn_file_t;

// Internal Context Structure
struct gmem_context {
  uint64_t seed;
  gmem_entry_t *overlay;
  size_t overlay_size;
  size_t overlay_count;
  gmem_mutex_t lock; // Portable Mutex

  uint8_t *macro_zmask; // 1 bit per 1TB chunk (Macro-Level)
  uint8_t **zmask;      // Array of pointers to 4KB page masks (Detail-Level)
  size_t macro_pages;   // Number of 1TB chunks tracked

  // Phase 14: Law Registry
  gmem_law_entry_t *law_registry;
  size_t law_count;

  // Phase 15: Resource Quotas
  size_t overlay_limit; // Max overlay entries

  // Phase 16: Mirroring
  struct gmem_context *source_ctx;
  gmem_mirror_mode_t mirror_mode;

  // Phase 18: Persistence
  FILE *persist_file;
  char *persist_path;

  // Phase 18: Archetypes
  gmem_archetype_t archetype;

  // Phase 19: Variety Morphing
  struct gmem_context *morph_source;
  gmem_morph_mode_t morph_mode;
  gmem_morph_params_t morph_params;

  // Phase 20: Dynamic Files (Hardening)
  gmem_dyn_file_t *dynamic_files;
};

// Architecture: Semantic Archetype Helpers
typedef struct {
  char name[256];
  uint64_t offset;
  uint64_t size;
  int is_dir;
} gmem_virt_entry_t;

size_t gmem_archetype_get_entries(gmem_ctx_t ctx, const char *path,
                                  gmem_virt_entry_t *entries,
                                  size_t max_entries);

// Add this to gmem_context in C file, or if this is the header, just
// declarations. Since gmem_context is defined here, we add the field.

#endif // GMEM_INTERNAL_H
