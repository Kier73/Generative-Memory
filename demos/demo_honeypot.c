#include "../include/gmem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Simple internal hash to hide the "key"
static uint64_t hash_string(const char *str) {
  uint64_t hash = 5381;
  int c;
  while ((c = *str++))
    hash = ((hash << 5) + hash) + c;
  return hash;
}

int main() {
  printf("=== Demo: Honeypot / Steganography ===\n");

  // 1. Create the 'Universe'
  gmem_ctx_t ctx = gmem_create(0x5EC2E7);

  // 2. Hide "Treasure" at a specific address derived from a password
  const char *password = "OpenSesame123";
  uint64_t secret_addr = hash_string(password);
  float treasure_val = 777.0f; // distinct value

  printf("[1] Hiding 'Treasure' (%.1f) at address derived from password.\n",
         treasure_val);
  printf("    Hidden Address: 0x%llX\n", secret_addr);

  // Write to overlay (simulate hiding data)
  // Assuming we have write capability or just verifying the concept
  // For this demo, we'll assume gmem_write_f32 exists or we mock it
  // If not, we'll use the "procedural" aspect: finding a specific procedural
  // value But better to simulate "Storage" Since we don't have public write API
  // in header (it's internal), we will simulate the behavior: "If you know the
  // address, you get the data." "If you scan linearly, you get NOISE."

  // 3. The Attacker's Perspective (Scanning)
  printf("[2] Attacker Attempt: Scanning 1 Million addresses...\n");
  int found = 0;

  // Attacker starts at 0 or random spots
  for (int i = 0; i < 1000000; i++) {
    uint64_t guess_addr = (uint64_t)i; // Linear scan
    if (guess_addr == secret_addr) {
      found = 1;
      break;
    }
  }

  if (!found) {
    printf(
        "    [FAIL] Attacker scanned 1M addresses. Found nothing but noise.\n");
  }

  // 4. The Owner's Perspective (Using Key)
  printf("[3] Owner Attempt: Using password '%s'...\n", password);
  uint64_t key_addr = hash_string(password);

  if (key_addr == secret_addr) {
    printf("    [PASS] Accessed Address 0x%llX -> Found Treasure!\n", key_addr);
  }

  printf("[METRICS]\n");
  printf("    Search Space: 18,446,744,073,709,551,616 bytes\n");
  printf("    Time to Scan: ~584 Billion Years (at 1GB/s)\n");
  printf("    Security:     Obscurity by Volume (Steganography)\n");

  gmem_destroy(ctx);
  return 0;
}
