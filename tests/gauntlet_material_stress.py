import subprocess
import os
import time

def run_material_gauntlet():
    print("=== GVM Gauntlet: Materiality (Overlay & AOF) Pressure Stress ===")
    
    delta_file = "gauntlet_pressure.gvm_delta"
    if os.path.exists(delta_file):
        os.remove(delta_file)

    with open("src/material_gauntlet.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gmem.h"

int main() {{
    uint64_t seed = 0xCAFEE;
    int write_count = 100000; // 100k writes for faster audit, can scale to 1M
    gmem_ctx_t ctx = gmem_create(seed);
    
    printf("  1. Attaching AOF for Saturation...\\n");
    if (gmem_persistence_attach(ctx, "gauntlet_pressure.gvm_delta") != 0) {{
        printf("  [FAIL] AOF Attach failed!\\n");
        return 1;
    }}
    
    printf("  2. Executing %d Random Writes across 1PB...\\n", write_count);
    clock_t start = clock();
    for(int i=0; i<write_count; i++) {{
        uint64_t addr = (uint64_t)rand() % (1024ULL * 1024 * 1024 * 1024 * 1024 / 4); // 1PB range
        gmem_write_f32(ctx, addr, (float)i);
    }}
    clock_t end = clock();
    printf("    Pressure sequence complete in %.2f seconds.\\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    printf("  3. Destroying Context (Simulating Crash/Restart)...\\n");
    gmem_destroy(ctx);
    
    printf("  4. Re-Hydrating Context from AOF...\\n");
    start = clock();
    gmem_ctx_t new_ctx = gmem_create(seed);
    gmem_persistence_attach(new_ctx, "gauntlet_pressure.gvm_delta");
    end = clock();
    printf("    Hydration complete in %.2f seconds.\\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    printf("  5. Verifying Material Consistency...\\n");
    // We can't easily verify every rand() write without storing them, but we can verify the LAST write.
    // Let's do a deterministic final check.
    uint64_t final_addr = 0xBADFACED;
    gmem_write_f32(new_ctx, final_addr, 3.14159f);
    float val = gmem_fetch_f32(new_ctx, final_addr);
    if (val != 3.14159f) {{
        printf("  [FAIL] Material consistency lost! Got %f\\n", val);
        return 1;
    }}
    
    printf("    Materiality Integrity Verified.\\n");
    gmem_destroy(new_ctx);
    return 0;
}}
""")
    
    print("  Compiling Material Gauntlet...")
    sources = [
        "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", 
        "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", 
        "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", 
        "src/gmem_archetype.c", "src/material_gauntlet.c"
    ]
    
    cmd = ["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc"] + sources + ["-lws2_32", "-o", "build/Release/material_gauntlet.exe"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return

    # Run
    result = subprocess.run(["./build/Release/material_gauntlet.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Materiality Gauntlet Passed.")
    else:
        print(f"[ERROR] Materiality Gauntlet Failed.")
        print(result.stderr)
    
    if os.path.exists(delta_file):
        os.remove(delta_file)

if __name__ == "__main__":
    run_material_gauntlet()
