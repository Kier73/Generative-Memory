#include "../include/gmem_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif
#include "../include/gmem_net.h"
#include "gmem_internal.h"

gmem_manager_t g_manager_singleton = NULL;

static void on_seed_discovered(uint64_t seed, const char *address,
                               uint16_t port) {
  if (!g_manager_singleton)
    return;
  // Automatically provision discovered seeds as remote tenants
  gmem_manager_get_by_seed(g_manager_singleton, seed);
}

typedef struct {
  uint64_t seed;
  gmem_ctx_t ctx;
  char path[256];
} gmem_tenant_t;

struct gmem_manager {
  gmem_tenant_t *tenants;
  size_t count;
  size_t capacity;
  void *lock;
  uint16_t port;
};

#ifdef _WIN32
#define MGR_LOCK(mgr) EnterCriticalSection((LPCRITICAL_SECTION)mgr->lock)
#define MGR_UNLOCK(mgr) LeaveCriticalSection((LPCRITICAL_SECTION)mgr->lock)
#else
#define MGR_LOCK(mgr)
#define MGR_UNLOCK(mgr)
#endif

gmem_manager_t gmem_manager_init() {
  gmem_manager_t mgr = (gmem_manager_t)calloc(1, sizeof(struct gmem_manager));
  mgr->capacity = 16;
  mgr->tenants = (gmem_tenant_t *)calloc(mgr->capacity, sizeof(gmem_tenant_t));

#ifdef _WIN32
  mgr->lock = malloc(sizeof(CRITICAL_SECTION));
  InitializeCriticalSection((LPCRITICAL_SECTION)mgr->lock);
#endif

  if (!g_manager_singleton)
    g_manager_singleton = mgr;

  return mgr;
}

void gmem_manager_shutdown(gmem_manager_t mgr) {
  if (!mgr)
    return;

  if (g_manager_singleton == mgr)
    g_manager_singleton = NULL;

  // Wait for any pending network operations to quiesce
  gmem_manager_net_stop(mgr);
#ifdef _WIN32
  Sleep(100);
#endif

  MGR_LOCK(mgr);
  for (size_t i = 0; i < mgr->count; i++) {
    gmem_destroy(mgr->tenants[i].ctx);
  }
  free(mgr->tenants);
  MGR_UNLOCK(mgr);

#ifdef _WIN32
  DeleteCriticalSection((LPCRITICAL_SECTION)mgr->lock);
  free(mgr->lock);
#endif

  free(mgr);
}

// ... (other functions) ...

void gmem_manager_net_start(gmem_manager_t mgr, uint16_t port) {
  if (mgr)
    mgr->port = (port == 0) ? 9999 : port;
  gmem_net_init(port);
  gmem_net_listen_start(on_seed_discovered);
}

void gmem_manager_net_stop(gmem_manager_t mgr) {
  gmem_net_listen_stop();
  gmem_net_shutdown();
}

void gmem_manager_shout(gmem_manager_t mgr, uint64_t seed) {
  uint16_t p = (mgr && mgr->port) ? mgr->port : 9999;
  gmem_net_shout(seed, p, p);
}

gmem_ctx_t gmem_manager_get_by_seed_with_quota(gmem_manager_t mgr,
                                               uint64_t seed,
                                               size_t quota_entries) {
  if (!mgr)
    return NULL;
  MGR_LOCK(mgr);

  for (size_t i = 0; i < mgr->count; i++) {
    if (mgr->tenants[i].seed == seed) {
      gmem_ctx_t ctx = mgr->tenants[i].ctx;
      MGR_UNLOCK(mgr);
      return ctx;
    }
  }

  if (mgr->count >= mgr->capacity) {
    mgr->capacity *= 2;
    mgr->tenants = (gmem_tenant_t *)realloc(
        mgr->tenants, mgr->capacity * sizeof(gmem_tenant_t));
  }

  gmem_ctx_t ctx = gmem_create(seed);
  if (ctx) {
    ctx->overlay_limit = quota_entries; // Set Quota
    mgr->tenants[mgr->count].seed = seed;
    mgr->tenants[mgr->count].ctx = ctx;
    snprintf(mgr->tenants[mgr->count].path, 256, "/seed_0x%llx.raw",
             (unsigned long long)seed);
    mgr->count++;
  }

  MGR_UNLOCK(mgr);
  return ctx;
}

gmem_ctx_t gmem_manager_get_by_seed(gmem_manager_t mgr, uint64_t seed) {
  return gmem_manager_get_by_seed_with_quota(mgr, seed, 0);
}

gmem_ctx_t gmem_manager_get_by_path(gmem_manager_t mgr, const char *path) {
  if (!mgr || !path)
    return NULL;
  MGR_LOCK(mgr);
  for (size_t i = 0; i < mgr->count; i++) {
    if (strcmp(mgr->tenants[i].path, path) == 0) {
      gmem_ctx_t ctx = mgr->tenants[i].ctx;
      MGR_UNLOCK(mgr);
      return ctx;
    }
  }
  MGR_UNLOCK(mgr);
  return NULL;
}

size_t gmem_manager_list_tenants(gmem_manager_t mgr, char **paths_out,
                                 size_t max_count) {
  if (!mgr)
    return 0;
  MGR_LOCK(mgr);
  size_t actual = mgr->count < max_count ? mgr->count : max_count;
  for (size_t i = 0; i < actual; i++) {
    paths_out[i] = strdup(mgr->tenants[i].path);
  }
  MGR_UNLOCK(mgr);
  return actual;
}
