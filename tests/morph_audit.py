import subprocess
import os

def run_morph_audit():
    print("=== GVM Phase 19: Variety Morphing (Functional Synthesis) Audit ===")
    
    with open("src/morph_audit.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "gmem.h"

int main() {{
    uint64_t seed = 0xCAFEE;
    uint64_t addr = 10000;
    
    printf("  1. Creating Source and Derived contexts...\\n");
    gmem_ctx_t source = gmem_create(seed);
    gmem_ctx_t derived = gmem_create(seed); // Seed doesn't matter for derived noise but let's keep it consistent
    
    gmem_morph_params_t params = {{ .a = 2.0f, .b = 0.5f }};
    gmem_morph_attach(derived, source, GMEM_MORPH_LINEAR, params);
    
    printf("  2. Verifying Single Fetch Morphing (y = 2x + 0.5)...\\n");
    float v_src = gmem_fetch_f32(source, addr);
    float v_der = gmem_fetch_f32(derived, addr);
    float expected = v_src * 2.0f + 0.5f;
    
    printf("    Source[%llu]: %f\\n", (unsigned long long)addr, v_src);
    printf("    Derived[%llu]: %f (Expected: %f)\\n", (unsigned long long)addr, v_der, expected);
    
    if (fabsf(v_der - expected) > 0.0001f) {{
        printf("  [FAIL] Morphing mismatch!\\n");
        return 1;
    }}
    
    printf("  3. Verifying Bulk Fetch Morphing (AVX2 Check)...\\n");
    float src_bulk[8], der_bulk[8];
    gmem_fetch_bulk_f32(source, addr, src_bulk, 8);
    gmem_fetch_bulk_f32(derived, addr, der_bulk, 8);
    
    for(int i=0; i<8; i++) {{
        float exp = src_bulk[i] * 2.0f + 0.5f;
        if (fabsf(der_bulk[i] - exp) > 0.0001f) {{
            printf("  [FAIL] Bulk Morphing mismatch at index %d!\\n", i);
            return 1;
        }}
    }}
    printf("    Bulk Parity Verified.\\n");

    printf("  4. Verifying Morphing Override (Substance)...\\n");
    float overlay_val = 1.234f;
    gmem_write_f32(derived, addr, overlay_val);
    float v_write = gmem_fetch_f32(derived, addr);
    printf("    Overwritten Derived[%llu]: %f (Expected: %f - No Morph)\\n", 
           (unsigned long long)addr, v_write, overlay_val);
    
    if (v_write != overlay_val) {{
        printf("  [FAIL] Overlay should override morph!\\n");
        return 1;
    }}

    printf("  5. Verifying Recursive (Chained) Morphing...\\n");
    gmem_ctx_t final = gmem_create(seed);
    gmem_morph_params_t final_params = {{ .a = 1.0f, .b = 1.0f }}; // Add 1
    gmem_morph_attach(final, derived, GMEM_MORPH_ADD, final_params);
    
    float v_final = gmem_fetch_f32(final, addr + 1); // Use clean address
    float v_src_1 = gmem_fetch_f32(source, addr + 1);
    float expected_final = (v_src_1 * 2.0f + 0.5f) + 1.0f;
    
    printf("    Final Chain Result: %f (Expected: %f)\\n", v_final, expected_final);
    if (fabsf(v_final - expected_final) > 0.0001f) {{
        printf("  [FAIL] Recursive Morphing failure!\\n");
        return 1;
    }}

    printf("  [PASS] Variety Morphing Verified: Functional Synthesis is operational.\\n");
    
    gmem_destroy(final);
    gmem_destroy(derived);
    gmem_destroy(source);
    return 0;
}}
""")
    
    print("  Compiling Morph Audit Utility...")
    sources = [
        "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", 
        "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", 
        "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", 
        "src/gmem_archetype.c", "src/morph_audit.c"
    ]
    
    cmd = ["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc"] + sources + ["-lws2_32", "-o", "build/Release/morph_audit.exe"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return

    # Run
    result = subprocess.run(["./build/Release/morph_audit.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Phase 19 Variety Morphing Verified.")
    else:
        print(f"[ERROR] Morph Audit Failed with Exit Code {result.returncode}")
        print(result.stderr)

if __name__ == "__main__":
    run_morph_audit()
