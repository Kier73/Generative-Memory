import subprocess

def run_archetype_audit():
    print("=== GVM Phase 18: Archetype (Semantic Synthesis) Audit ===")
    
    with open("src/archetype_audit.c", "w") as f:
        f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gmem.h"
#include "gmem_internal.h"

int main() {{
    gmem_ctx_t ctx = gmem_create(0x5EED);
    gmem_set_archetype(ctx, GMEM_ARCHETYPE_FAT);
    
    printf("  1. Auditing Root Directory (/)...\\n");
    gmem_virt_entry_t entries[10];
    size_t count = gmem_archetype_get_entries(ctx, "/", entries, 10);
    
    printf("  Found %zu entries in Root:\\n", count);
    for (size_t i = 0; i < count; i++) {{
        printf("    [%s] %s (Offset: %llu, Size: %llu)\\n", 
               entries[i].is_dir ? "DIR " : "FILE", 
               entries[i].name, 
               (unsigned long long)entries[i].offset, 
               (unsigned long long)entries[i].size);
    }}
    
    if (count < 2) {{
        printf("  [FAIL] Root synthesis failed to project entries.\\n");
        return 1;
    }}

    printf("  2. Auditing Sub-directory (/assets)...\\n");
    count = gmem_archetype_get_entries(ctx, "/assets", entries, 10);
    printf("  Found %zu entries in /assets:\\n", count);
    for (size_t i = 0; i < count; i++) {{
        printf("    [%s] %s\\n", entries[i].is_dir ? "DIR " : "FILE", entries[i].name);
    }}
    
    if (count == 0) {{
        printf("  [FAIL] Sub-directory synthesis failed.\\n");
        return 1;
    }}

    printf("  [PASS] Archetype Audit Successful: Semantic structure projected from seed.\\n");
    
    gmem_destroy(ctx);
    return 0;
}}
""")
    
    print("  Compiling Archetype Audit Utility...")
    subprocess.run(["gcc", "-O3", "-mavx2", "-Iinclude", "-Isrc", "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c", "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c", "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c", "src/gmem_archetype.c", "src/archetype_audit.c", "-lws2_32", "-o", "build/Release/archetype_audit.exe"], check=True)
    
    # Run
    result = subprocess.run(["./build/Release/archetype_audit.exe"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("[SUCCESS] Phase 18 Archetype Synthesis Verified.")
    else:
        print(f"[ERROR] Archetype Audit Failed with Exit Code {result.returncode}")
        print(result.stderr)

if __name__ == "__main__":
    run_archetype_audit()
