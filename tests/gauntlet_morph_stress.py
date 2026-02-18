import subprocess
import os
import time

def run_morph_gauntlet():
    print("=== GVM Gauntlet: Infinite Derivation (Deep Chain) Stress ===")
    
    with open("src/morph_gauntlet.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "gmem.h"

int main() {{
    uint64_t seed = 0xDEEE;
    int chain_depth = 100;
    gmem_ctx_t contexts[101];
    
    printf("  1. Building %d-Level Derivation Chain...\\n", chain_depth);
    contexts[0] = gmem_create(seed);
    for(int i=1; i<=chain_depth; i++) {{
        contexts[i] = gmem_create(seed);
        gmem_morph_params_t p = {{ .a = 1.001f, .b = 0.0001f }}; // Tiny increment each level
        gmem_morph_attach(contexts[i], contexts[i-1], GMEM_MORPH_LINEAR, p);
    }}
    
    printf("  2. Verifying Deep Resolution (Recursive Path)...\\n");
    uint64_t addr = 42;
    float v0 = gmem_fetch_f32(contexts[0], addr);
    float vn = gmem_fetch_f32(contexts[chain_depth], addr);
    
    printf("    Base Value: %f\\n", v0);
    printf("    100-Level Value: %f\\n", vn);
    
    if (vn == 0.0f || isnan(vn) || isinf(vn)) {{
        printf("  [FAIL] Derivation chain collapsed or drifted to infinity!\\n");
        return 1;
    }}
    
    printf("  3. Verifying Bulk Resolution Performance on 100th Level...\\n");
    float buffer[1024];
    clock_t start = clock();
    gmem_fetch_bulk_f32(contexts[chain_depth], 0, buffer, 1024);
    clock_t end = clock();
    printf("    Bulk fetch (1024 floats) through 100 levels took %.4f seconds.\\n", (double)(end-start)/CLOCKS_PER_SEC);

    printf("    Derivation Chain Verified.\\n");
    for(int i=0; i<=chain_depth; i++) gmem_destroy(contexts[i]);
    return 0;
}}
""")
    
    print("  Compiling Morph Gauntlet...")
    sources = [
        "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", 
        "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", 
        "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", 
        "src/gmem_archetype.c", "src/morph_gauntlet.c"
    ]
    
    cmd = ["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc"] + sources + ["-lws2_32", "-o", "build/Release/morph_gauntlet.exe"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return

    # Run
    result = subprocess.run(["./build/Release/morph_gauntlet.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Morph Gauntlet Passed.")
    else:
        print(f"[ERROR] Morph Gauntlet Failed.")
        print(result.stderr)

if __name__ == "__main__":
    run_morph_gauntlet()
