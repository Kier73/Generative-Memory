import subprocess
import os

def run_persistence_audit():
    print("=== GVM Phase 18: Persistence (AOF) Audit ===")
    
    delta_file = "test_persistence.gvm_delta"
    if os.path.exists(delta_file):
        os.remove(delta_file)
        
    with open("src/persistence_audit.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gmem.h"

int main() {{
    uint64_t seed = 0xABCDE;
    uint64_t addr1 = 1000;
    float val1 = 123.456f;
    uint64_t addr2 = 2000;
    float val2 = 789.012f;

    printf("  1. Creating Session 1 and writing to AOF...\\n");
    gmem_ctx_t ctx1 = gmem_create(seed);
    if (gmem_persistence_attach(ctx1, "{delta_file}") != 0) {{
        printf("  [FAIL] Could not attach persistence file\\n");
        return 1;
    }}
    
    gmem_write_f32(ctx1, addr1, val1);
    gmem_write_f32(ctx1, addr2, val2);
    gmem_destroy(ctx1);
    
    printf("  2. Creating Session 2 and re-attaching AOF...\\n");
    gmem_ctx_t ctx2 = gmem_create(seed);
    
    // Before attach, should be generative noise
    float noise = gmem_fetch_f32(ctx2, addr1);
    printf("  Baseline (Noise) at addr %llu: %f\\n", (unsigned long long)addr1, noise);

    if (gmem_persistence_attach(ctx2, "{delta_file}") != 0) {{
        printf("  [FAIL] Could not re-attach persistence file\\n");
        return 1;
    }}
    
    // After attach, should match session 1
    float read1 = gmem_fetch_f32(ctx2, addr1);
    float read2 = gmem_fetch_f32(ctx2, addr2);
    
    printf("  Readout 1: %f (Expected %f)\\n", read1, val1);
    printf("  Readout 2: %f (Expected %f)\\n", read2, val2);
    
    if (read1 == val1 && read2 == val2) {{
        printf("  [PASS] Persistence Verified: Materialized changes survived session reboot.\\n");
    }} else {{
        printf("  [FAIL] Parity Mismatch: Log replay failed.\\n");
        return 1;
    }}
    
    gmem_destroy(ctx2);
    return 0;
}}
""")
    
    print("  Compiling Persistence Audit Utility...")
    subprocess.run(["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc", "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", "src/persistence_audit.c", "-lws2_32", "-o", "build/Release/persistence_audit.exe"], check=True)
    
    # Run
    result = subprocess.run(["./build/Release/persistence_audit.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Phase 18 Persistence Verified.")
        if os.path.exists(delta_file):
            os.remove(delta_file)
    else:
        print(f"[ERROR] Persistence Audit Failed with Exit Code {result.returncode}")
        print(result.stderr)

if __name__ == "__main__":
    run_persistence_audit()
