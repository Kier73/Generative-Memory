#define FUSE_USE_VERSION 26
#include "../include/gmem.h"
#include "../include/gmem_manager.h"
#include "gmem_internal.h"
#include <errno.h>
#include <fcntl.h>
#include <fuse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef S_IFDIR
#define S_IFDIR 0040000
#endif
#ifndef S_IFREG
#define S_IFREG 0100000
#endif

static gmem_manager_t gmem_mgr = NULL;
static const uint64_t GVM_TOTAL_SIZE =
    1024ULL * 1024 * 1024 * 1024 * 1024; // 1 Petabyte

// Helper to split "/seed_0x.../sub/path" -> (tenant, "/sub/path")
static gmem_ctx_t resolve_path(const char *path, char *subpath_out) {
  if (path[0] != '/')
    return NULL;

  char temp[256];
  strncpy(temp, path, 256);

  char *first_slash = temp;
  char *second_slash = strchr(temp + 1, '/');

  if (second_slash) {
    *second_slash = '\0';
    strcpy(subpath_out, path + (second_slash - temp));
  } else {
    if (subpath_out)
      strcpy(subpath_out, "/");
  }

  return gmem_manager_get_by_path(gmem_mgr, temp);
}

// --- FUSE CALLBACKS ---

static int gmem_getattr(const char *path, struct fuse_stat *stbuf) {
  struct fuse_context *ctx = fuse_get_context();
  memset(stbuf, 0, sizeof(struct fuse_stat));
  stbuf->st_uid = ctx->uid;
  stbuf->st_gid = ctx->gid;

  if (strcmp(path, "/") == 0) {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    return 0;
  }

  char subpath[256];
  gmem_ctx_t tenant = resolve_path(path, subpath);

  if (!tenant)
    return -ENOENT;

  // If path is exactly the tenant root (e.g. "/seed_0x...")
  if (strcmp(subpath, "/") == 0) {
    stbuf->st_mode = S_IFDIR | 0755; // Tenants are now directories
    stbuf->st_nlink = 2;
    return 0;
  }

  // Archetype Lookup
  gmem_virt_entry_t entries[32];
  // We need to find the parent of the subpath and check its entries
  char parent[256], leaf[256];
  char *last_slash = strrchr(subpath, '/');
  if (last_slash == subpath) {
    strcpy(parent, "/");
    strcpy(leaf, subpath + 1);
  } else {
    size_t len = last_slash - subpath;
    strncpy(parent, subpath, len);
    parent[len] = '\0';
    strcpy(leaf, last_slash + 1);
  }

  size_t count = gmem_archetype_get_entries(tenant, parent, entries, 32);
  for (size_t i = 0; i < count; i++) {
    if (strcmp(entries[i].name, leaf) == 0) {
      if (entries[i].is_dir) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
      } else {
        stbuf->st_mode = S_IFREG | 0777;
        stbuf->st_nlink = 1;
        stbuf->st_size = entries[i].size;
      }
      return 0;
    }
  }

  return -ENOENT;
}

static int gmem_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                        fuse_off_t offset, struct fuse_file_info *fi) {
  (void)offset;
  (void)fi;
  filler(buf, ".", NULL, 0);
  filler(buf, "..", NULL, 0);

  if (strcmp(path, "/") == 0) {
    char *paths[64];
    size_t count = gmem_manager_list_tenants(gmem_mgr, paths, 64);
    for (size_t i = 0; i < count; i++) {
      filler(buf, paths[i] + 1, NULL, 0); // Skip leading '/'
      free(paths[i]);
    }
    return 0;
  }

  char subpath[256];
  gmem_ctx_t tenant = resolve_path(path, subpath);
  if (!tenant)
    return -ENOENT;

  gmem_virt_entry_t entries[64];
  size_t count = gmem_archetype_get_entries(tenant, subpath, entries, 64);
  for (size_t i = 0; i < count; i++) {
    filler(buf, entries[i].name, NULL, 0);
  }

  return 0;
}

static int gmem_open(const char *path, struct fuse_file_info *fi) {
  char subpath[256];
  if (!resolve_path(path, subpath))
    return -ENOENT;
  return 0;
}

static int gmem_read(const char *path, char *buf, size_t size,
                     fuse_off_t offset, struct fuse_file_info *fi) {
  (void)fi;
  char subpath[256];
  gmem_ctx_t tenant = resolve_path(path, subpath);
  if (!tenant)
    return -ENOENT;

  // Resolve file offset from archetype
  uint64_t base_offset = 0;
  // This is a naive search, in production we'd cache file handles
  char parent[256], leaf[256];
  char *last_slash = strrchr(subpath, '/');
  if (last_slash == subpath) {
    strcpy(parent, "/");
    strcpy(leaf, subpath + 1);
  } else {
    size_t len = last_slash - subpath;
    strncpy(parent, subpath, len);
    parent[len] = '\0';
    strcpy(leaf, last_slash + 1);
  }

  gmem_virt_entry_t entries[32];
  size_t count = gmem_archetype_get_entries(tenant, parent, entries, 32);
  int found = 0;
  for (size_t i = 0; i < count; i++) {
    if (strcmp(entries[i].name, leaf) == 0 && !entries[i].is_dir) {
      base_offset = entries[i].offset;
      found = 1;
      break;
    }
  }
  if (!found)
    return -ENOENT;

  uint64_t start_addr = (base_offset + (uint64_t)offset) / sizeof(float);
  size_t count_f = size / sizeof(float);
  gmem_fetch_bulk_f32(tenant, start_addr, (float *)buf, count_f);
  return (int)(count_f * sizeof(float));
}

static int gmem_write(const char *path, const char *buf, size_t size,
                      fuse_off_t offset, struct fuse_file_info *fi) {
  (void)fi;
  char subpath[256];
  gmem_ctx_t tenant = resolve_path(path, subpath);
  if (!tenant)
    return -ENOENT;

  // Similar offset resolution...
  // (Left as exercise for writing to virtual files - routing to underlying
  // addr)
  return -EROFS; // Archetype files are read-only by default in this
                 // implementation
}

static struct fuse_operations gmem_oper = {
    .getattr = gmem_getattr,
    .readdir = gmem_readdir,
    .open = gmem_open,
    .read = gmem_read,
    .write = gmem_write,
};

int main(int argc, char *argv[]) {
  printf("Starting GMC Semantic Hypervisor...\n");

  uint16_t port = 9999;
  uint64_t seed = 0x1337BEEF;

  // Argument Parsing (Strip GVM args before passing to FUSE)
  int new_argc = 0;
  char **new_argv = malloc(argc * sizeof(char *));

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
      port = (uint16_t)atoi(argv[++i]);
      printf("[CONF] Net Port: %d\n", port);
    } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
      seed = strtoull(argv[++i], NULL, 16);
      printf("[CONF] Initial Seed: 0x%llx\n", seed);
    } else {
      new_argv[new_argc++] = argv[i];
    }
  }

  gmem_mgr = gmem_manager_init();
  gmem_manager_net_start(gmem_mgr, port);

  // Provision User-Defined Seed
  gmem_ctx_t t1 = gmem_manager_get_by_seed(gmem_mgr, seed);
  gmem_set_archetype(t1, GMEM_ARCHETYPE_FAT);

  // Provision Demo Seed
  gmem_ctx_t t2 = gmem_manager_get_by_seed(gmem_mgr, 0xDEADBEEF);
  gmem_set_archetype(t2, GMEM_ARCHETYPE_RAW);

  int ret = fuse_main(new_argc, new_argv, &gmem_oper, NULL);
  free(new_argv);
  return ret;
}
