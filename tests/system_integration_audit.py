import subprocess
import os

def run_system_integration_audit():
    print("=== GVM: Full-Stack Integration Audit ===")
    
    delta_file = "system_integration.gvm_delta"
    if os.path.exists(delta_file):
        os.remove(delta_file)
        
    with open("src/system_integration_audit.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gmem.h"
#include "gmem_internal.h"
#include "gmem_manager.h"

int main() {{
    uint64_t master_seed = 0x57ACC;
    uint64_t test_addr = 5000;
    float test_val = 987.654f;

    printf("  [INIT] Step 1: Initialize Source with Persistence...\\n");
    gmem_ctx_t source = gmem_create(master_seed);
    if (gmem_persistence_attach(source, "{delta_file}") != 0) {{
        printf("  [FAIL] Persistence attachment failed.\\n");
        return 1;
    }}

    printf("  [OP] Step 2: Create Mirror (Shadow)...\\n");
    gmem_ctx_t shadow = gmem_create(master_seed); // Same seed for identity
    gmem_mirror_attach(shadow, source, GMEM_MIRROR_IDENTITY);

    printf("  [OP] Step 3: Configure Archetype on Shadow...\\n");
    gmem_set_archetype(shadow, GMEM_ARCHETYPE_FAT);

    printf("  [OP] Step 4: Write to Persistent Source...\\n");
    gmem_write_f32(source, test_addr, test_val);

    printf("  [VERIFY] Step 5: Fetch from Shadow (Mirror Parity)...\\n");
    float mirror_read = gmem_fetch_f32(shadow, test_addr);
    printf("    Mirror Read: %f (Expected %f)\\n", mirror_read, test_val);

    if (mirror_read != test_val) {{
        printf("  [FAIL] Mirror-Persistence coherence failure.\\n");
        return 1;
    }}

    printf("  [VERIFY] Step 6: Audit Shadow Archetype Entries...\\n");
    gmem_virt_entry_t entries[10];
    size_t count = gmem_archetype_get_entries(shadow, "/", entries, 10);
    printf("    Found %zu entries in Shadow Root.\\n", count);
    if (count < 2) {{
        printf("  [FAIL] Archetype not functional through Mirror.\\n");
        return 1;
    }}

    printf("  [OP] Step 7: Restart System (Durability Check)...\\n");
    gmem_destroy(shadow);
    gmem_destroy(source);

    gmem_ctx_t source_v2 = gmem_create(master_seed);
    if (gmem_persistence_attach(source_v2, "{delta_file}") != 0) {{
        printf("  [FAIL] Re-attachment failed.\\n");
        return 1;
    }}

    float durable_read = gmem_fetch_f32(source_v2, test_addr);
    printf("    Durable Read: %f (Expected %f)\\n", durable_read, test_val);

    if (durable_read != test_val) {{
        printf("  [FAIL] Durability failure.\\n");
        return 1;
    }}

    printf("  [PASS] Full-Stack Integration Verified: Persistence, Mirroring, and Archetypes are coherent.\\n");
    
    gmem_destroy(source_v2);
    return 0;
}}
""")
    
    print("  Compiling System Audit Utility...")
    # Add all sources
    sources = [
        "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", 
        "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", 
        "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", 
        "src/gmem_archetype.c", "src/system_integration_audit.c"
    ]
    
    cmd = ["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc"] + sources + ["-lws2_32", "-o", "build/Release/system_integration_audit.exe"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return

    # Run
    result = subprocess.run(["./build/Release/system_integration_audit.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Full-Stack Audit Passed.")
        if os.path.exists(delta_file):
            os.remove(delta_file)
    else:
        print(f"[ERROR] Audit Failed with Exit Code {result.returncode}")
        print(result.stderr)

if __name__ == "__main__":
    run_system_integration_audit()
