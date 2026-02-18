
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gmem.h"
#include "gmem_internal.h"

int main() {
    gmem_ctx_t ctx = gmem_create(0x5EED);
    gmem_set_archetype(ctx, GMEM_ARCHETYPE_FAT);
    
    printf("  1. Auditing Root Directory (/)...\n");
    gmem_virt_entry_t entries[10];
    size_t count = gmem_archetype_get_entries(ctx, "/", entries, 10);
    
    printf("  Found %zu entries in Root:\n", count);
    for (size_t i = 0; i < count; i++) {
        printf("    [%s] %s (Offset: %llu, Size: %llu)\n", 
               entries[i].is_dir ? "DIR " : "FILE", 
               entries[i].name, 
               (unsigned long long)entries[i].offset, 
               (unsigned long long)entries[i].size);
    }
    
    if (count < 2) {
        printf("  [FAIL] Root synthesis failed to project entries.\n");
        return 1;
    }

    printf("  2. Auditing Sub-directory (/assets)...\n");
    count = gmem_archetype_get_entries(ctx, "/assets", entries, 10);
    printf("  Found %zu entries in /assets:\n", count);
    for (size_t i = 0; i < count; i++) {
        printf("    [%s] %s\n", entries[i].is_dir ? "DIR " : "FILE", entries[i].name);
    }
    
    if (count == 0) {
        printf("  [FAIL] Sub-directory synthesis failed.\n");
        return 1;
    }

    printf("  [PASS] Archetype Audit Successful: Semantic structure projected from seed.\n");
    
    gmem_destroy(ctx);
    return 0;
}
