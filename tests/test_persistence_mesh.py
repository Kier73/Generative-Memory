
import subprocess
import os
import time
import sys

def compile_c(source_name, output_name, extra_sources=[]):
    print(f"  Compiling {output_name}...")
    sources = [
        "src/gmem.c", "src/gmem_alloc.c", "src/gmem_filter_json.c",
        "src/gmem_filter_block.c", "src/gmem_vrns.c", "src/gmem_trinity.c",
        "src/gmem_aro.c", "src/gmem_manager.c", "src/gmem_net.c",
        "src/gmem_archetype.c", source_name
    ]
    # Include extra sources if needed
    sources.extend(extra_sources)
    
    cmd = ["gcc", "-O2", "-mavx2", "-Iinclude", "-Isrc"] + sources + ["-lws2_32", "-o", output_name]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return False

def test_persistence():
    print("\n=== Testing Persistence (AOF) ===")
    
    c_source = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gmem.h"

int main(int argc, char* argv[]) {
    uint64_t seed = 0x12345;
    const char* filename = "test_persist.gvm";
    uint64_t addr = 0x1000;
    float val_test = 3.14159f;

    if (argc > 1 && strcmp(argv[1], "write") == 0) {
        // CLEANUP
        remove(filename);
        
        gmem_ctx_t ctx = gmem_create(seed);
        if(gmem_persistence_attach(ctx, filename) != 0) {
            return 1;
        }
        gmem_write_f32(ctx, addr, val_test);
        gmem_destroy(ctx);
        printf("Written\\n");
        return 0;
    } 
    
    if (argc > 1 && strcmp(argv[1], "read") == 0) {
        gmem_ctx_t ctx = gmem_create(seed);
        if(gmem_persistence_attach(ctx, filename) != 0) {
            return 2;
        }
        float val_out = gmem_fetch_f32(ctx, addr);
        gmem_destroy(ctx);
        
        if (fabs(val_out - val_test) < 0.0001) {
             printf("Verified: %f\\n", val_out);
             return 0;
        } else {
             printf("Failed: %f != %f\\n", val_out, val_test);
             return 3;
        }
    }
    return 0;
}
"""
    with open("src/test_persist.c", "w") as f:
        f.write(c_source)
        
    if not compile_c("src/test_persist.c", "test_persist.exe"):
        return

    # Phase 1: Write
    print("  1. Writing to persistent context...")
    res = subprocess.run(["./test_persist.exe", "write"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [FAIL] Write failed: {res.stdout} {res.stderr}")
        return
        
    # Phase 2: Read
    print("  2. Reading from restored context...")
    res = subprocess.run(["./test_persist.exe", "read"], capture_output=True, text=True)
    if res.returncode == 0:
        print("  [SUCCESS] Persistence Verified.")
    else:
        print(f"  [FAIL] Read failed: {res.stdout} {res.stderr}")

def test_mesh():
    print("\n=== Testing Mesh Coherence (UDP) ===")
    
    # LISTENER CODE
    c_listener = """
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "gmem.h"
#include "gmem_manager.h"

int main() {
    uint64_t seed = 0xFACEFEED;
    uint64_t addr = 0x2000;
    float expected = 123.456f;
    
    gmem_manager_t mgr = gmem_manager_init();
    gmem_manager_net_start(mgr);
    
    // Provision the context so we can receive updates for it
    gmem_ctx_t ctx = gmem_manager_get_by_seed(mgr, seed);
    
    printf("Listing...\\n");
    fflush(stdout);
    
    // Poll for update (timeout 5s)
    for(int i=0; i<50; i++) {
        float val = gmem_fetch_f32(ctx, addr);
        if (val == expected) {
            printf("Received Sync: %f\\n", val);
            gmem_manager_shutdown(mgr);
            return 0;
        }
        Sleep(100);
    }
    
    printf("Timeout waiting for sync.\\n");
    gmem_manager_shutdown(mgr);
    return 1;
}
"""
    with open("src/test_mesh_listen.c", "w") as f:
        f.write(c_listener)

    # WRITER CODE
    c_writer = """
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "gmem.h"
#include "gmem_manager.h"

int main() {
    uint64_t seed = 0xFACEFEED;
    uint64_t addr = 0x2000;
    float val = 123.456f;
    
    gmem_manager_t mgr = gmem_manager_init();
    gmem_manager_net_start(mgr);
    
    gmem_ctx_t ctx = gmem_manager_get_by_seed(mgr, seed);
    
    printf("Broadcasting write...\\n");
    // Write triggers broadcast
    gmem_write_f32(ctx, addr, val);
    
    // Keep alive briefly to ensure packet sends
    Sleep(1000);
    
    gmem_manager_shutdown(mgr);
    return 0;
}
"""
    with open("src/test_mesh_write.c", "w") as f:
        f.write(c_writer)
        
    if not compile_c("src/test_mesh_listen.c", "test_listener.exe"): return
    if not compile_c("src/test_mesh_write.c", "test_writer.exe"): return
    
    # Run Listener in background
    print("  1. Starting Listener...")
    listener = subprocess.Popen(["./test_listener.exe"], stdout=subprocess.PIPE, text=True)
    
    time.sleep(1) # Wait for listener to bind port
    
    # Run Writer
    print("  2. Starting Writer...")
    subprocess.run(["./test_writer.exe"], check=True)
    
    # Check Listener result
    try:
        stdout, stderr = listener.communicate(timeout=6)
        if listener.returncode == 0:
             print("  [SUCCESS] Mesh Coherence Verified.")
             print(f"  Listener Output: {stdout.strip()}")
        else:
             print("  [FAIL] Listener timed out or failed.")
             print(f"  Output: {stdout}")
    except subprocess.TimeoutExpired:
        listener.kill()
        print("  [FAIL] Listener timed out.")

if __name__ == "__main__":
    test_persistence()
    test_mesh()
