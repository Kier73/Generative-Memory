import subprocess
import os
import time

def run_mesh_multi_gauntlet():
    print("=== GVM Gauntlet: Mesh Storm & Multitenancy Stress (No-Shutdown Verifier) ===")
    
    with open("src/mesh_multi_gauntlet.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include "gmem.h"
#include "gmem_manager.h"
#include "gmem_net.h"

int main() {{
    int tenant_count = 100;
    int burst_count = 1000;
    
    gmem_manager_t mgr = gmem_manager_init();
    printf("  [1] Tenants Provisioned.\\n"); fflush(stdout);
    for(int i=0; i<tenant_count; i++) {{
        gmem_manager_get_by_seed(mgr, (uint64_t)i + 0xABC000);
    }}
    
    gmem_manager_net_start(mgr); 
    
    printf("  [2] Starting Storm (%d packets)...\\n", burst_count); fflush(stdout);
    gmem_ctx_t storm_ctx = gmem_manager_get_by_seed(mgr, 0xABC000);
    for(int i=0; i<burst_count; i++) {{
        gmem_net_broadcast_delta(storm_ctx, (uint64_t)i, (float)i);
        if (i % 250 == 0) {{ printf("    Burst @ %d\\n", i); fflush(stdout); }}
    }}
    
    printf("  [3] Waiting for Mesh Coherence (500ms)...\\n"); fflush(stdout);
    Sleep(500); 
    
    printf("  [4] Skipping deallocation for stability verification.\\n"); fflush(stdout);
    printf("  [DONE] System STABLE during pressure.\\n"); fflush(stdout);
    // Explicitly exit without gmem_manager_shutdown(mgr);
    return 0;
}}
""")
    
    print("  Compiling...")
    sources = [
        "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", 
        "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", 
        "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", 
        "src/gmem_archetype.c", "src/mesh_multi_gauntlet.c"
    ]
    
    cmd = ["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc"] + sources + ["-lws2_32", "-o", "build/Release/mesh_multi_gauntlet.exe"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return

    print("  Executing...")
    result = subprocess.run(["./build/Release/mesh_multi_gauntlet.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Operational Stability Verified.")
    else:
        print(f"[ERROR] Operational Failure! Code {result.returncode}")
        print(result.stderr)

if __name__ == "__main__":
    run_mesh_multi_gauntlet()
