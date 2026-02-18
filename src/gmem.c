#include "../include/gmem.h"
#include "gmem_internal.h"
#include <stdlib.h>
#include <string.h>

#define OVERLAY_INITIAL_SIZE 1024
#define OVERLAY_LOAD_FACTOR 0.7
#define GMEM_PAGE_SIZE 1024 // 4KB in floats
#define GMEM_ZMASK_PAGES                                                       \
  (1024 * 1024 * 256) // 1TB coverage by default (32MB mask)

// Internal Context Logic

// --- INTERNAL OVERLAY LOGIC (Simple Hash Map) ---

static uint64_t gmem_hash(uint64_t addr) {
  uint64_t h = 14695981039346656037ULL;
  h ^= addr;
  h *= 1099511628211ULL;
  return h;
}

static void gmem_overlay_insert(gmem_ctx_t ctx, uint64_t addr, float value) {
  GMEM_LOCK(ctx);

  // Quota Check
  if (ctx->overlay_limit > 0 && ctx->overlay_count >= ctx->overlay_limit) {
    // Check if updating existing entry first
    uint64_t h_check = gmem_hash(addr);
    size_t idx_check = (size_t)(h_check % ctx->overlay_size);
    int exists = 0;
    while (ctx->overlay[idx_check].active) {
      if (ctx->overlay[idx_check].virtual_addr == addr) {
        exists = 1;
        break;
      }
      idx_check = (idx_check + 1) % ctx->overlay_size;
    }
    if (!exists) {
      GMEM_UNLOCK(ctx);
      return; // Quota exceeded
    }
  }

  if (ctx->overlay_count >= ctx->overlay_size * OVERLAY_LOAD_FACTOR) {
    // printf("DEBUG: Resizing overlay from %zu to %zu\n", ctx->overlay_size,
    // ctx->overlay_size * 2);
    size_t old_size = ctx->overlay_size;
    gmem_entry_t *old_entries = ctx->overlay;

    ctx->overlay_size *= 2;
    ctx->overlay =
        (gmem_entry_t *)calloc(ctx->overlay_size, sizeof(gmem_entry_t));
    if (!ctx->overlay) {
      printf("FATAL: Resize allocation failed for size %zu\n",
             ctx->overlay_size);
      exit(1);
    }
    ctx->overlay_count = 0;

    for (size_t i = 0; i < old_size; i++) {
      if (old_entries[i].active) {
        // DIRECT INSERT to avoid lock recursion/overhead and ensure simple
        // hashing
        uint64_t h = gmem_hash(old_entries[i].virtual_addr);
        size_t idx = (size_t)(h % ctx->overlay_size);
        size_t start_idx = idx;
        while (ctx->overlay[idx].active) {
          idx = (idx + 1) % ctx->overlay_size;
          if (idx == start_idx) {
            printf("FATAL: Resize collision loop full!\n");
            exit(1);
          }
        }
        ctx->overlay[idx].virtual_addr = old_entries[i].virtual_addr;
        ctx->overlay[idx].value = old_entries[i].value;
        ctx->overlay[idx].active = 1;
        ctx->overlay_count++;
      }
    }
    free(old_entries);
  }

  uint64_t h = gmem_hash(addr);
  size_t idx = (size_t)(h % ctx->overlay_size);

  /*
  if (ctx->overlay_count > 500000 && ctx->overlay_count % 10000 == 0) {
      printf("Debug: Count %zu, Size %zu, Idx %zu, OverlayPtr %p\n",
  ctx->overlay_count, ctx->overlay_size, idx, ctx->overlay);
  }
  */

  size_t start_idx = idx;
  while (ctx->overlay[idx].active) {
    if (ctx->overlay[idx].virtual_addr == addr) {
      ctx->overlay[idx].value = value;
      GMEM_UNLOCK(ctx);
      return;
    }
    idx = (idx + 1) % ctx->overlay_size;
    if (idx == start_idx) {
      printf("FATAL: Insert loop full!\n");
      exit(1);
    }
  }

  ctx->overlay[idx].virtual_addr = addr;
  ctx->overlay[idx].value = value;
  ctx->overlay[idx].active = 1;
  ctx->overlay_count++;
  GMEM_UNLOCK(ctx);
}

static int gmem_overlay_lookup(gmem_ctx_t ctx, uint64_t addr, float *out_val) {
  int found = 0;
  GMEM_LOCK(ctx);
  if (ctx->overlay_size == 0) {
    GMEM_UNLOCK(ctx);
    return 0;
  }

  uint64_t h = gmem_hash(addr);
  size_t idx = (size_t)(h % ctx->overlay_size);
  size_t start_idx = idx;

  while (ctx->overlay[idx].active) {
    if (ctx->overlay[idx].virtual_addr == addr) {
      *out_val = ctx->overlay[idx].value;
      found = 1;
      break;
    }
    idx = (idx + 1) % ctx->overlay_size;
    if (idx == start_idx)
      break;
  }
  GMEM_UNLOCK(ctx);
  return found;
}

// --- CORE PROCEDURAL SYNTHESIS (Deterministic Shuffle) ---

static uint64_t gmem_shuffle_u64(uint64_t val, uint64_t seed) {
  uint32_t l = (uint32_t)(val >> 32);
  uint32_t r = (uint32_t)val;

  for (int i = 0; i < 4; i++) {
    uint32_t temp = r;
    r = l ^ (r * 0x9E3779B9 + (uint32_t)seed);
    l = temp;
  }

  return ((uint64_t)l << 32) | (uint64_t)r;
}

// --- PUBLIC API IMPLEMENTATION ---

gmem_ctx_t gmem_create(uint64_t seed) {
  gmem_ctx_t ctx = (gmem_ctx_t)calloc(1, sizeof(struct gmem_context));
  if (ctx) {
    ctx->seed = seed;
    ctx->overlay_size = OVERLAY_INITIAL_SIZE;
    ctx->overlay_count = 0;
    ctx->overlay =
        (gmem_entry_t *)calloc(ctx->overlay_size, sizeof(gmem_entry_t));
    if (!ctx->overlay) {
      free(ctx);
      return NULL;
    }
    gmem_mutex_init(&ctx->lock);

    // Phase 13: Hierarchical Z-Mask initialization
    ctx->macro_pages = 1024; // Cover 1024 TB (1 Petabyte)
    ctx->macro_zmask = (uint8_t *)calloc(ctx->macro_pages / 8, 1);
    ctx->zmask = (uint8_t **)calloc(ctx->macro_pages, sizeof(uint8_t *));
  }
  return ctx;
}

void gmem_destroy(gmem_ctx_t ctx) {
  if (ctx) {
    if (ctx->zmask) {
      for (size_t i = 0; i < ctx->macro_pages; i++) {
        if (ctx->zmask[i])
          free(ctx->zmask[i]);
      }
      free(ctx->zmask);
    }
    if (ctx->macro_zmask)
      free(ctx->macro_zmask);
    if (ctx->law_registry)
      free(ctx->law_registry);

    gmem_mutex_destroy(&ctx->lock);
    if (ctx->persist_file)
      fclose((FILE *)ctx->persist_file);
    if (ctx->persist_path)
      free(ctx->persist_path);

    if (ctx->overlay)
      free(ctx->overlay);
    free(ctx);
  }
}

#include "gmem_vrns.h"

float gmem_fetch_f32(gmem_ctx_t ctx, uint64_t virtual_addr) {
  if (!ctx)
    return 0.0f;

  float cached_val;
  if (gmem_overlay_lookup(ctx, virtual_addr, &cached_val)) {
    return cached_val;
  }

  // Phase 19: Variety Morphing
  if (ctx->morph_source) {
    float src_val = gmem_fetch_f32(ctx->morph_source, virtual_addr);
    switch (ctx->morph_mode) {
    case GMEM_MORPH_LINEAR:
      return src_val * ctx->morph_params.a + ctx->morph_params.b;
    case GMEM_MORPH_ADD:
      return src_val + ctx->morph_params.b;
    case GMEM_MORPH_MUL:
      return src_val * ctx->morph_params.a;
    default:
      return src_val;
    }
  }

  // Phase 16: Mirroring Support
  if (ctx->source_ctx) {
    return gmem_fetch_f32(ctx->source_ctx, virtual_addr);
  }

  // Phase 9: Use vRNS Accelerator (Theta) instead of Feistel
  gmem_rns_t rns = gmem_vrns_project(virtual_addr, ctx->seed);
  return gmem_vrns_to_float(rns);
}

float gmem_fetch_monotonic_f32(gmem_ctx_t ctx, uint64_t index) {
  if (!ctx)
    return 0.0f;

  // 1. High-precision ramp [0, 1]
  double base_ramp = (double)index / 18446744073709551615.0;

  // 2. Local variety (constrained to maintain strict monotonicity)
  uint64_t variety_int = gmem_shuffle_u64(index, ctx->seed);
  double variety =
      ((double)variety_int / 18446744073709551615.0) / 18446744073709551615.0;

  return (float)(base_ramp + variety);
}

uint64_t gmem_search_f32(gmem_ctx_t ctx, float target) {
  if (!ctx)
    return 0;
  if (target <= 0.0f)
    return 0;
  if (target >= 1.0f)
    return 18446744073709551615ULL;

  uint64_t low = 0;
  uint64_t high = 18446744073709551615ULL;

  // Predictive Search (Interpolation-based)
  // Since the law is nearly linear, we can converge extremely fast.
  for (int i = 0; i < 64; i++) { // Max 64 iterations for bit-exact resolution
    if (high <= low)
      break;

    float low_val = gmem_fetch_monotonic_f32(ctx, low);
    float high_val = gmem_fetch_monotonic_f32(ctx, high);

    if (target <= low_val)
      return low;
    if (target >= high_val)
      return high;

    // Linear Interpolation: pivot = low + (target - low_val) * (high - low) /
    // (high_val - low_val)
    double scale = (double)(target - low_val) / (double)(high_val - low_val);
    uint64_t pivot = low + (uint64_t)(scale * (double)(high - low));

    float pivot_val = gmem_fetch_monotonic_f32(ctx, pivot);

    if (pivot_val < target) {
      low = pivot + 1;
    } else if (pivot_val > target) {
      high = pivot - 1;
    } else {
      return pivot;
    }
  }

  return low;
}

void gmem_write_internal(gmem_ctx_t ctx, uint64_t virtual_addr, float value) {
  gmem_overlay_insert(ctx, virtual_addr, value);

  // Phase 13: Hierarchical Z-Mask Update
  uint64_t tb_idx = virtual_addr / GMEM_TB_SIZE;
  if (tb_idx < ctx->macro_pages) {
    // 1. Mark Macro-Mask
    ctx->macro_zmask[tb_idx >> 3] |= (1 << (tb_idx & 7));

    // 2. Allocate/Get Detail-Mask
    if (!ctx->zmask[tb_idx]) {
      ctx->zmask[tb_idx] = (uint8_t *)calloc(GMEM_PAGES_PER_TB / 8, 1);
    }

    if (ctx->zmask[tb_idx]) {
      uint64_t page_in_tb = (virtual_addr % GMEM_TB_SIZE) / GMEM_PAGE_SIZE;
      ctx->zmask[tb_idx][page_in_tb >> 3] |= (1 << (page_in_tb & 7));
    }
  }
}

void gmem_write_f32(gmem_ctx_t ctx, uint64_t virtual_addr, float value) {
  if (!ctx)
    return;
  GMEM_LOCK(ctx);
  gmem_write_internal(ctx, virtual_addr, value);

  // Phase 18: AOF Logging (Persistent Delta-Log)
  if (ctx->persist_file) {
    fwrite(&virtual_addr, sizeof(uint64_t), 1, (FILE *)ctx->persist_file);
    fwrite(&value, sizeof(float), 1, (FILE *)ctx->persist_file);
    fflush((FILE *)ctx->persist_file);
  }

  // Phase 20: Mesh Sync (Delta Broadcast)
  extern void gmem_net_broadcast_delta(gmem_ctx_t ctx, uint64_t addr,
                                       float val);
  gmem_net_broadcast_delta(ctx, virtual_addr, value);

  GMEM_UNLOCK(ctx);
}

#include <stdio.h>

// --- INTEGRITY SHIELD (Simple CRC32) ---

static uint32_t gmem_crc32(const void *data, size_t size) {
  const uint8_t *p = (const uint8_t *)data;
  uint32_t crc = 0xFFFFFFFF;
  for (size_t i = 0; i < size; i++) {
    crc ^= p[i];
    for (int j = 0; j < 8; j++) {
      crc = (crc >> 1) ^ (0xEDB88320 & (-(int32_t)(crc & 1)));
    }
  }
  return ~crc;
}

#define GMEM_FILE_MAGIC 0x4D454D47 // "GMEM"
#define GMEM_FILE_VERSION 1

int gmem_save_overlay(gmem_ctx_t ctx, const char *path) {
  if (!ctx || !path)
    return -1;

  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;

  // 1. Header
  uint32_t magic = GMEM_FILE_MAGIC;
  uint32_t version = GMEM_FILE_VERSION;
  fwrite(&magic, sizeof(uint32_t), 1, f);
  fwrite(&version, sizeof(uint32_t), 1, f);
  fwrite(&ctx->seed, sizeof(uint64_t), 1, f);
  fwrite(&ctx->overlay_count, sizeof(size_t), 1, f);

  // 2. Entries & Checksum calculation
  uint32_t checksum = 0;
  for (size_t i = 0; i < ctx->overlay_size; i++) {
    if (ctx->overlay[i].active) {
      fwrite(&ctx->overlay[i].virtual_addr, sizeof(uint64_t), 1, f);
      fwrite(&ctx->overlay[i].value, sizeof(float), 1, f);
      // Simple incremental checksum for performance
      checksum ^= gmem_crc32(&ctx->overlay[i].virtual_addr, sizeof(uint64_t));
      checksum ^= gmem_crc32(&ctx->overlay[i].value, sizeof(float));
    }
  }

  fwrite(&checksum, sizeof(uint32_t), 1, f);

  fclose(f);
  return 0;
}

int gmem_load_overlay(gmem_ctx_t ctx, const char *path) {
  if (!ctx || !path)
    return -1;

  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;

  uint32_t magic, version;
  if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != GMEM_FILE_MAGIC) {
    fclose(f);
    return -1;
  }
  if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
      version != GMEM_FILE_VERSION) {
    fclose(f);
    return -1;
  }

  uint64_t seed;
  size_t count;
  if (fread(&seed, sizeof(uint64_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fread(&count, sizeof(size_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }

  ctx->seed = seed;
  uint32_t checksum_calc = 0;

  for (size_t i = 0; i < count; i++) {
    uint64_t addr;
    float val;
    if (fread(&addr, sizeof(uint64_t), 1, f) != 1)
      break;
    if (fread(&val, sizeof(float), 1, f) != 1)
      break;
    gmem_overlay_insert(ctx, addr, val);
    checksum_calc ^= gmem_crc32(&addr, sizeof(uint64_t));
    checksum_calc ^= gmem_crc32(&val, sizeof(float));
  }

  uint32_t checksum_file;
  if (fread(&checksum_file, sizeof(uint32_t), 1, f) != 1 ||
      checksum_file != checksum_calc) {
    // Integrity failure (In a real system we might want to alert more loudly)
    fclose(f);
    return -2;
  }

  fclose(f);
  return 0;
}

#include <immintrin.h>

static void gmem_theta_single(uint64_t addr, uint64_t seed, float *out) {
  gmem_rns_t rns = gmem_vrns_project(addr, seed);
  *out = gmem_vrns_to_float(rns);
}

void gmem_fetch_bulk_f32(gmem_ctx_t ctx, uint64_t start_addr, float *buffer,
                         size_t count) {
  if (!ctx || !buffer)
    return;

  size_t i = 0;

#if defined(__AVX2__) || defined(GMEM_USE_AVX2)
#pragma message("GMEM: AVX2 Path Enabled")
  __m256i seed_v = _mm256_set1_epi32((uint32_t)ctx->seed);
  __m256i prime_v = _mm256_set1_epi32(0x9E3779B9);
  __m256i stride_v = _mm256_set1_epi32(8 * 0x9E3779B9);
  __m256 scale_v = _mm256_set1_ps(1.0f / 2147483647.0f);
  __m256i mask_v = _mm256_set1_epi32(0x7FFFFFFF);

  size_t count_simd8 = (count / 8) * 8;

  // --- Phase 14: Algebraic Rewrite & Optimization (ARO) ---
  float p1 = 0.0f, p2 = 0.0f;
  gmem_arr_rule_t rule = gmem_aro_simplify(ctx, ctx->seed, &p1, &p2);

  // Global Shunt for Constant/Zero/Linear manifolds.
  if (rule == GMEM_ARR_CONST || rule == GMEM_ARR_ZERO ||
      rule == GMEM_ARR_LINEAR) {
    int range_is_clean = 1;
    uint64_t tb_start = start_addr / GMEM_TB_SIZE;
    uint64_t tb_end = (start_addr + count) / GMEM_TB_SIZE;
    for (uint64_t t = tb_start; t <= tb_end && t < ctx->macro_pages; t++) {
      if (ctx->macro_zmask[t >> 3] & (1 << (t & 7))) {
        range_is_clean = 0;
        break;
      }
    }

    if (range_is_clean) {
      if (rule == GMEM_ARR_CONST || rule == GMEM_ARR_ZERO) {
        __m256 const_v = _mm256_set1_ps(p1);
        for (size_t k = 0; k < count_simd8; k += 8) {
#ifdef GMEM_DEBUG_INTEGRITY
          // Verify first few elements to ensure the 'Constant' assumption holds
          for (int v = 0; v < 8; v++) {
            gmem_rns_t rns = gmem_vrns_project(start_addr + k + v, ctx->seed);
            float expected = gmem_vrns_to_float(rns);
            if (expected != p1) {
              fprintf(
                  stderr,
                  "GMC INTEGRITY FAILURE: Constant Shunt mismatch at 0x%llx. "
                  "Expected %f, got %f\n",
                  (unsigned long long)(start_addr + k + v), expected, p1);
              exit(1);
            }
          }
#endif
          _mm256_storeu_ps(&buffer[k], const_v);
        }
      } else if (rule == GMEM_ARR_LINEAR) {
        // V(x) = p1 + x * p2
        for (size_t k = 0; k < count_simd8; k += 8) {
          float base_x = (float)(start_addr + k);
          __m256 x_v = _mm256_add_ps(_mm256_set1_ps(base_x),
                                     _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0));
          __m256 val_v = _mm256_add_ps(_mm256_set1_ps(p1),
                                       _mm256_mul_ps(x_v, _mm256_set1_ps(p2)));
          _mm256_storeu_ps(&buffer[k], val_v);
        }
      }
      i = count_simd8;
      goto scalar_fallback;
    }
  }

  while (i < count_simd8) {
    uint64_t current_addr = start_addr + i;
    uint64_t page_idx = current_addr / GMEM_PAGE_SIZE;

    // Calculate blocks remaining in the current page
    size_t floats_in_page = GMEM_PAGE_SIZE - (current_addr % GMEM_PAGE_SIZE);
    size_t blocks_in_page = floats_in_page / 8;

    // Limit to remaining count
    if (blocks_in_page * 8 > (count_simd8 - i)) {
      blocks_in_page = (count_simd8 - i) / 8;
    }

    int is_dirty = 1;
    uint64_t tb_idx = current_addr / GMEM_TB_SIZE;
    if (tb_idx < ctx->macro_pages) {
      // 1. Check Macro-Mask
      if (!(ctx->macro_zmask[tb_idx >> 3] & (1 << (tb_idx & 7)))) {
        is_dirty = 0; // The entire TB is clean
      } else if (ctx->zmask[tb_idx]) {
        // 2. Check Detail-Mask
        uint64_t page_in_tb = (current_addr % GMEM_TB_SIZE) / GMEM_PAGE_SIZE;
        is_dirty =
            (ctx->zmask[tb_idx][page_in_tb >> 3] & (1 << (page_in_tb & 7)));
      }
    }

    if (!is_dirty && blocks_in_page > 0) {
      if (ctx->morph_source) {
        // Phase 19: Morphing Fetch (Derived)
        gmem_fetch_bulk_f32(ctx->morph_source, current_addr, &buffer[i],
                            blocks_in_page * 8);

        // Apply Vectorized Transform
        __m256 a_v = _mm256_set1_ps(ctx->morph_params.a);
        __m256 b_v = _mm256_set1_ps(ctx->morph_params.b);

        for (size_t k = 0; k < blocks_in_page * 8; k += 8) {
          __m256 data = _mm256_loadu_ps(&buffer[i + k]);
          switch (ctx->morph_mode) {
          case GMEM_MORPH_LINEAR:
            data = _mm256_add_ps(_mm256_mul_ps(data, a_v), b_v);
            break;
          case GMEM_MORPH_ADD:
            data = _mm256_add_ps(data, b_v);
            break;
          case GMEM_MORPH_MUL:
            data = _mm256_mul_ps(data, a_v);
            break;
          default:
            break;
          }
          _mm256_storeu_ps(&buffer[i + k], data);
        }

        i += blocks_in_page * 8;
        continue;
      }

      if (ctx->source_ctx) {
        // Phase 16: Recursive Mirror Fetch
        gmem_fetch_bulk_f32(ctx->source_ctx, current_addr, &buffer[i],
                            blocks_in_page * 8);
        i += blocks_in_page * 8;
        continue;
      }

      if (rule == GMEM_ARR_CONST || rule == GMEM_ARR_ZERO) {
        __m256 const_v = _mm256_set1_ps(p1);
        for (size_t b = 0; b < blocks_in_page; b++) {
          _mm256_storeu_ps(&buffer[i], const_v);
          i += 8;
        }
        continue;
      }
      // Else proceed with normal Page-Run synthesis...
      // --- FAST PAGE RUN (Loop Unrolled / No Branches) ---
      __m256i variety_v = _mm256_xor_si256(
          _mm256_mullo_epi32(
              _mm256_set_epi32((int)(current_addr + 7), (int)(current_addr + 6),
                               (int)(current_addr + 5), (int)(current_addr + 4),
                               (int)(current_addr + 3), (int)(current_addr + 2),
                               (int)(current_addr + 1),
                               (int)(current_addr + 0)),
              prime_v),
          seed_v);

      for (size_t b = 0; b < blocks_in_page; b++) {
        __m256 floats = _mm256_cvtepi32_ps(_mm256_and_si256(variety_v, mask_v));
        _mm256_storeu_ps(&buffer[i], _mm256_mul_ps(floats, scale_v));
        variety_v = _mm256_add_epi32(variety_v, stride_v);
        i += 8;
      }
    } else {
      // --- COHERENT BLOCK (or untracked page) ---
      for (int j = 0; j < 8; j++) {
        buffer[i + j] = gmem_fetch_f32(ctx, current_addr + j);
      }
      i += 8;
    }
  }
#endif

  // Scalar Fallback
scalar_fallback:
  for (; i < count; i++) {
    buffer[i] = gmem_fetch_f32(ctx, start_addr + i);
  }
}

// --- Phase 16: Mirroring ---

void gmem_mirror_attach(gmem_ctx_t shadow, gmem_ctx_t source,
                        gmem_mirror_mode_t mode) {
  if (!shadow || !source || shadow == source)
    return;

  GMEM_LOCK(shadow);
  shadow->source_ctx = source;
  shadow->mirror_mode = mode;
  GMEM_UNLOCK(shadow);
}

// --- Phase 18: Persistence & Archetypes ---

int gmem_persistence_attach(gmem_ctx_t ctx, const char *path) {
  if (!ctx || !path)
    return -1;

  GMEM_LOCK(ctx);
  if (ctx->persist_file) {
    fclose((FILE *)ctx->persist_file);
  }

  // Try to open existing or create new for append+read
  FILE *f = fopen(path, "ab+");
  if (!f) {
    GMEM_UNLOCK(ctx);
    return -1;
  }

  ctx->persist_file = f;
  ctx->persist_path = strdup(path);

  // Hydration: Replay the log
  fseek(f, 0, SEEK_SET);
  uint64_t addr;
  float val;
  while (fread(&addr, sizeof(uint64_t), 1, f) == 1) {
    if (fread(&val, sizeof(float), 1, f) == 1) {
      gmem_write_internal(ctx, addr, val);
    }
  }
  // Ensure we are at the end for future appends
  fseek(f, 0, SEEK_END);

  GMEM_UNLOCK(ctx);
  return 0;
}

void gmem_set_archetype(gmem_ctx_t ctx, gmem_archetype_t archetype) {
  if (ctx) {
    ctx->archetype = archetype;
    // Phase 20: Broadcast Law Change
    extern void gmem_net_broadcast_law(gmem_ctx_t ctx);
    gmem_net_broadcast_law(ctx);
  }
}

// --- Phase 19: Variety Morphing ---

void gmem_morph_attach(gmem_ctx_t target, gmem_ctx_t source,
                       gmem_morph_mode_t mode, gmem_morph_params_t params) {
  if (!target || !source || target == source)
    return;

  GMEM_LOCK(target);
  target->morph_source = source;
  target->morph_mode = mode;
  target->morph_params = params;

  // Phase 20: Broadcast Law Change
  extern void gmem_net_broadcast_law(gmem_ctx_t ctx);
  gmem_net_broadcast_law(target);

  GMEM_UNLOCK(target);
}
