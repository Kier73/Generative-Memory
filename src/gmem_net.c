#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#include "../include/gmem_net.h"
#include <string.h>

#include "gmem_os.h"

// --- Portable Networking ---
#ifdef _WIN32
#include <process.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET gmem_socket_t;
#define GMEM_INVALID_SOCKET INVALID_SOCKET
#define GMEM_SOCKET_ERROR SOCKET_ERROR
#define gmem_close_socket(s) closesocket(s)
#define gmem_net_cleanup() WSACleanup()

// Thread wrapper for Windows
static unsigned __stdcall win_thread_wrapper(void *arg) {
  void (*func)(void *) = (void (*)(void *))arg;
  func(arg); // This is a bit hacky, the arg usually contains the real func +
             // data. For this simple case, we'll refactor the thread launch
             // slightly.
  return 0;
}
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
typedef int gmem_socket_t;
#define GMEM_INVALID_SOCKET -1
#define GMEM_SOCKET_ERROR -1
#define gmem_close_socket(s) close(s)
#define gmem_net_cleanup()
#endif

static gmem_socket_t shout_sock = GMEM_INVALID_SOCKET;
static gmem_socket_t listen_sock = GMEM_INVALID_SOCKET;
static uint16_t g_net_port = 9999;
static gmem_thread_t listen_thread;
static int listen_thread_active = 0;
static gmem_discovery_cb discovery_callback = NULL;
static volatile int net_running = 0;

// Security Token (Default 0 = Insecure/Public)
static uint32_t g_net_token = 0;

void gmem_net_set_token(uint32_t token) { g_net_token = token; }

// --- Reliability Layer Constants ---
#define GMEM_NET_MAX_RETRIES 3
#define GMEM_NET_TIMEOUT_MS 200

typedef enum {
  GMEM_NET_MSG_SHOUT = 0,
  GMEM_NET_MSG_LAW = 1,
  GMEM_NET_MSG_SYNC = 2,
  GMEM_NET_MSG_ACK = 3
} gmem_net_msg_type_t;

#pragma pack(push, 1)
typedef struct {
  char magic[4];       // "GVM!"
  uint32_t auth_token; // New Security Field
  uint32_t type;
  uint32_t seq; // Sequence Number for ARQ
  uint64_t seed;
  union {
    struct {
      uint16_t port;
    } shout;
    struct {
      uint32_t morph_mode;
      float morph_a;
      float morph_b;
      uint32_t archetype;
    } law;
    struct {
      uint64_t addr;
      float val;
    } sync;
    struct {
      uint32_t ack_seq;
    } ack;
  } data;
} gmem_packet_t;
#pragma pack(pop)

#include "../include/gmem_manager.h"
#include "gmem_internal.h"

static volatile uint32_t g_last_ack_received = 0;
static volatile int g_ack_flag = 0;
static uint32_t g_seq_counter = 1;

static void send_ack(struct sockaddr_in *dest, uint32_t seq) {
  gmem_packet_t packet;
  memset(&packet, 0, sizeof(packet));
  memcpy(packet.magic, "GVM!", 4);
  packet.auth_token = g_net_token;
  packet.type = GMEM_NET_MSG_ACK;
  packet.data.ack.ack_seq = seq;
  sendto(shout_sock, (char *)&packet, sizeof(packet), 0,
         (struct sockaddr *)dest, sizeof(*dest));
}

// Portable Thread Routine
static void *listen_thread_func(void *arg) {
  (void)arg;
  struct sockaddr_in si_other;
  socklen_t slen = sizeof(si_other);
  gmem_packet_t packet;

// Set timeout on recv to allow clean exit handling
#ifdef _WIN32
  DWORD timeout = 1000;
  setsockopt(listen_sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout,
             sizeof(timeout));
#else
  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  setsockopt(listen_sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv,
             sizeof(tv));
#endif

  while (net_running) {
    int recv_len = recvfrom(listen_sock, (char *)&packet, sizeof(packet), 0,
                            (struct sockaddr *)&si_other, &slen);
    if (!net_running)
      break;

    if (recv_len == sizeof(packet)) {
      if (memcmp(packet.magic, "GVM!", 4) == 0) {

        // SECURITY CHECK
        if (packet.auth_token != g_net_token) {
          // Drop packet silently
          continue;
        }

        // Handle ACK
        if (packet.type == GMEM_NET_MSG_ACK) {
          g_last_ack_received = packet.data.ack.ack_seq;
          g_ack_flag = 1;
          continue;
        }

        if (packet.type == GMEM_NET_MSG_SHOUT && discovery_callback) {
          char ip_str[INET_ADDRSTRLEN];
          inet_ntop(AF_INET, &si_other.sin_addr, ip_str, INET_ADDRSTRLEN);
          discovery_callback(packet.seed, ip_str, packet.data.shout.port);
        } else if (packet.type == GMEM_NET_MSG_LAW ||
                   packet.type == GMEM_NET_MSG_SYNC) {

          // Send ACK immediately for reliable messages
          send_ack(&si_other, packet.seq);

          // Distributed Sync: Find context by seed and apply
          extern gmem_manager_t g_manager_singleton; // From gmem_manager.c
          if (g_manager_singleton) {
            gmem_ctx_t ctx =
                gmem_manager_get_by_seed(g_manager_singleton, packet.seed);
            if (ctx) {
              if (packet.type == GMEM_NET_MSG_LAW) {
                ctx->morph_mode = (gmem_morph_mode_t)packet.data.law.morph_mode;
                ctx->morph_params.a = packet.data.law.morph_a;
                ctx->morph_params.b = packet.data.law.morph_b;
                ctx->archetype = (gmem_archetype_t)packet.data.law.archetype;
              } else if (packet.type == GMEM_NET_MSG_SYNC) {
                // Apply remote write to local overlay
                extern void gmem_write_internal(
                    gmem_ctx_t ctx, uint64_t virtual_addr, float value);
                GMEM_LOCK(ctx); // Portable Lock
                gmem_write_internal(ctx, packet.data.sync.addr,
                                    packet.data.sync.val);
                GMEM_UNLOCK(ctx);
              }
            }
          }
        }
      }
    }
  }
  return NULL;
}

#ifdef _WIN32
static unsigned __stdcall win_thread_shim(void *arg) {
  listen_thread_func(arg);
  return 0;
}
#endif

int gmem_net_init(uint16_t port) {
#ifdef _WIN32
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    return -1;
#endif

  if (port != 0)
    g_net_port = port;

  shout_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (shout_sock == GMEM_INVALID_SOCKET)
    return -1;
  int broadcast = 1;
  setsockopt(shout_sock, SOL_SOCKET, SO_BROADCAST, (char *)&broadcast,
             sizeof(broadcast));

  listen_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (listen_sock == GMEM_INVALID_SOCKET)
    return -1;
  int reuse = 1;
  setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse,
             sizeof(reuse));

  struct sockaddr_in si_me;
  memset(&si_me, 0, sizeof(si_me));
  si_me.sin_family = AF_INET;
  si_me.sin_port = htons(g_net_port); // Discovery Port
  si_me.sin_addr.s_addr = INADDR_ANY;

  if (bind(listen_sock, (struct sockaddr *)&si_me, sizeof(si_me)) ==
      GMEM_SOCKET_ERROR) {
    return -2;
  }

  return 0;
}

void gmem_net_shutdown() {
  net_running = 0;

  // Close sockets to unblock recvfrom if it's stuck (though we used timeout)
  if (shout_sock != GMEM_INVALID_SOCKET) {
    gmem_close_socket(shout_sock);
    shout_sock = GMEM_INVALID_SOCKET;
  }
  if (listen_sock != GMEM_INVALID_SOCKET) {
    gmem_close_socket(listen_sock);
    listen_sock = GMEM_INVALID_SOCKET;
  }

  if (listen_thread_active) {
#ifdef _WIN32
    WaitForSingleObject(listen_thread, 2000);
    CloseHandle(listen_thread);
#else
    pthread_join(listen_thread, NULL);
#endif
    listen_thread_active = 0;
  }

  gmem_net_cleanup();
}

void gmem_net_listen_start(gmem_discovery_cb callback) {
  if (net_running)
    return;
  discovery_callback = callback;
  net_running = 1;
  listen_thread_active = 1;

#ifdef _WIN32
  unsigned thread_id;
  listen_thread =
      (HANDLE)_beginthreadex(NULL, 0, win_thread_shim, NULL, 0, &thread_id);
#else
  pthread_create(&listen_thread, NULL, listen_thread_func, NULL);
#endif
}

void gmem_net_listen_stop() { gmem_net_shutdown(); }

void gmem_net_shout(uint64_t seed, uint16_t target_port, uint16_t my_port) {
  if (shout_sock == GMEM_INVALID_SOCKET)
    return;

  if (target_port == 0)
    target_port = g_net_port;
  if (my_port == 0)
    my_port = g_net_port;

  struct sockaddr_in si_dest;
  memset(&si_dest, 0, sizeof(si_dest));
  si_dest.sin_family = AF_INET;
  si_dest.sin_port = htons(target_port);
  si_dest.sin_addr.s_addr = INADDR_BROADCAST;

  gmem_packet_t packet;
  memset(&packet, 0, sizeof(packet));
  memcpy(packet.magic, "GVM!", 4);
  packet.auth_token = g_net_token; // AUTH TOKEN
  packet.type = GMEM_NET_MSG_SHOUT;
  packet.seed = seed;
  packet.data.shout.port = my_port;

  sendto(shout_sock, (char *)&packet, sizeof(packet), 0,
         (struct sockaddr *)&si_dest, sizeof(si_dest));
}

void gmem_net_broadcast_law(gmem_ctx_t ctx) {
  if (!ctx || shout_sock == GMEM_INVALID_SOCKET)
    return;

  struct sockaddr_in si_dest;
  memset(&si_dest, 0, sizeof(si_dest));
  si_dest.sin_family = AF_INET;
  si_dest.sin_port = htons(g_net_port);
  si_dest.sin_addr.s_addr = INADDR_BROADCAST;

  gmem_packet_t packet;
  memset(&packet, 0, sizeof(packet));
  memcpy(packet.magic, "GVM!", 4);
  packet.auth_token = g_net_token; // AUTH TOKEN
  packet.type = GMEM_NET_MSG_LAW;
  packet.seed = ctx->seed;
  packet.data.law.morph_mode = ctx->morph_mode;
  packet.data.law.morph_a = ctx->morph_params.a;
  packet.data.law.morph_b = ctx->morph_params.b;
  packet.data.law.archetype = ctx->archetype;

  sendto(shout_sock, (char *)&packet, sizeof(packet), 0,
         (struct sockaddr *)&si_dest, sizeof(si_dest));
}

void gmem_net_broadcast_delta(gmem_ctx_t ctx, uint64_t addr, float val) {
  if (!ctx || shout_sock == GMEM_INVALID_SOCKET)
    return;

  struct sockaddr_in si_dest;
  memset(&si_dest, 0, sizeof(si_dest));
  si_dest.sin_family = AF_INET;
  si_dest.sin_port = htons(g_net_port);
  si_dest.sin_addr.s_addr = INADDR_BROADCAST;

  gmem_packet_t packet;
  memset(&packet, 0, sizeof(packet));
  memcpy(packet.magic, "GVM!", 4);
  packet.auth_token = g_net_token; // AUTH TOKEN
  packet.type = GMEM_NET_MSG_SYNC;
  packet.seq = g_seq_counter++;
  packet.seed = ctx->seed;
  packet.data.sync.addr = addr;
  packet.data.sync.val = val;

  // Redundant Send (Poor Man's Reliability for Broadcast)
  // Real ARQ requires session management per peer
  for (int i = 0; i < 3; i++) {
    sendto(shout_sock, (char *)&packet, sizeof(packet), 0,
           (struct sockaddr *)&si_dest, sizeof(si_dest));
#ifdef _WIN32
    Sleep(1);
#else
    usleep(100);
#endif
  }
}
