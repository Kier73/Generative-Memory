#ifndef GMEM_MANAGER_H
#define GMEM_MANAGER_H

#include "gmem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Handle to the GVM Hypervisor.
 */
typedef struct gmem_manager *gmem_manager_t;

/**
 * Initialize the GVM Manager.
 */
gmem_manager_t gmem_manager_init();

/**
 * Shutdown the manager and free all contexts.
 */
void gmem_manager_shutdown(gmem_manager_t mgr);

/**
 * Create or get a tenant context by seed.
 */
gmem_ctx_t gmem_manager_get_by_seed(gmem_manager_t mgr, uint64_t seed);

/**
 * Create or get a tenant context by seed, with a specific overlay quota.
 * @param quota_bytes Max size of the sparse overlay in bytes.
 */
gmem_ctx_t gmem_manager_get_by_seed_with_quota(gmem_manager_t mgr,
                                               uint64_t seed,
                                               size_t quota_bytes);

/**
 * Create or get a tenant context by name (derived from seed or explicit).
 */
gmem_ctx_t gmem_manager_get_by_path(gmem_manager_t mgr, const char *path);

/**
 * List all active tenant paths.
 * @param paths_out Array of strings to populate.
 * @param max_count Maximum number of paths to return.
 * @return Actual number of paths found.
 */
size_t gmem_manager_list_tenants(gmem_manager_t mgr, char **paths_out,
                                 size_t max_count);

/**
 * Start the networking substrate and discovery.
 * @param port The UDP port to listen on.
 */
void gmem_manager_net_start(gmem_manager_t mgr, uint16_t port);

/**
 * Stop networking operations.
 */
void gmem_manager_net_stop(gmem_manager_t mgr);

/**
 * Broadcast a hosted seed to the network.
 */
void gmem_manager_shout(gmem_manager_t mgr, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif // GMEM_MANAGER_H
