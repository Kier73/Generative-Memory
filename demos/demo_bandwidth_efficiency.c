#include "../include/gmem.h"
#include <stdio.h>
#include <stdlib.h>

// Simulation of a Network Packet
typedef struct {
  uint32_t ops;
  uint32_t payload_bytes;
} net_sim_t;

int main() {
  printf("=== Demo: Bandwidth Efficiency ===\n");

  // Scenario: We want to transmit a 256MB noise texture (2k x 2k float).
  size_t width = 8192;
  size_t height = 8192;
  size_t raw_size = width * height * sizeof(float); // 256 MB

  printf("[1] Scenario: Syncing 8k x 8k Texture (%.2f MB)\n",
         (double)raw_size / (1024 * 1024));

  // Approach A: Traditional (Raw Send)
  net_sim_t traditional;
  traditional.payload_bytes = raw_size; // Header overhead negligible
  traditional.ops = 1;

  // Approach B: Generative Memory (Logic Send)
  // We just send the Seed + Rule (Linear Identity or Noise Parm)
  // Packet: [Header: 4b][Seed: 8b][Rule: 4b][Params: 8b]
  net_sim_t gmem_packet;
  gmem_packet.payload_bytes = 24;
  gmem_packet.ops = 1;

  printf("[2] Transmission Results:\n");
  printf("    Traditional: %u bytes\n", traditional.payload_bytes);
  printf("    G-Mem:       %u bytes\n", gmem_packet.payload_bytes);

  double savings =
      (double)traditional.payload_bytes / (double)gmem_packet.payload_bytes;
  printf("[METRICS]\n");
  printf("    Bandwidth Reduction: %.1fx\n", savings);

  // Verify the data is actually accessible "on the other side"
  gmem_ctx_t client_ctx = gmem_create(987654321); // Received seed
  float sample = gmem_fetch_f32(client_ctx, 1024);
  printf("    Client State: Ready to read. Sample[1024] = %f\n", sample);

  gmem_destroy(client_ctx);
  return 0;
}
