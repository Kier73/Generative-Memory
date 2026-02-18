#ifndef GMEM_NET_H
#define GMEM_NET_H

#include "gmem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the GVM Networking substrate.
 * @param port The UDP port to bind for listening (0 for random/ephemeral).
 * Returns 0 on success, non-zero on error.
 */
int gmem_net_init(uint16_t port);

/**
 * Set the security token for the mesh.
 * Packets with mismatched tokens will be dropped.
 * @param token The 32-bit integer secret.
 */
void gmem_net_set_token(uint32_t token);

/**
 * Shutdown the networking substrate.
 */
void gmem_net_shutdown();

/**
 * Broadcast a seed announcement.
 * @param seed The generative seed to broadcast.
 * @param target_port The destination port for the broadcast (e.g. 9999).
 * @param my_port The listener port for this node (to be advertised).
 */
void gmem_net_shout(uint64_t seed, uint16_t target_port, uint16_t my_port);

/**
 * Definition for the discovery callback.
 */
typedef void (*gmem_discovery_cb)(uint64_t seed, const char *address,
                                  uint16_t port);

/**
 * Begin listening for peer seeds on the network.
 * @param callback The function to execute upon discovery of a unique seed.
 */
void gmem_net_listen_start(gmem_discovery_cb callback);

/**
 * Stop discovery operations.
 */
void gmem_net_listen_stop();

/**
 * Broadcast a Variety Law (Morph/Archetype) configuration to the mesh.
 */
void gmem_net_broadcast_law(gmem_ctx_t ctx);

/**
 * Broadcast a physical write delta (Materiality) to the mesh.
 */
void gmem_net_broadcast_delta(gmem_ctx_t ctx, uint64_t addr, float val);

#ifdef __cplusplus
}
#endif

#endif // GMEM_NET_H
