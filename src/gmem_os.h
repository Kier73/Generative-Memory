#ifndef GMEM_OS_H
#define GMEM_OS_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef CRITICAL_SECTION gmem_mutex_t;
#define gmem_mutex_init(m) InitializeCriticalSection(m)
#define gmem_mutex_destroy(m) DeleteCriticalSection(m)
#define gmem_mutex_lock(m) EnterCriticalSection(m)
#define gmem_mutex_unlock(m) LeaveCriticalSection(m)

typedef HANDLE gmem_thread_t;

// Windows Socket includes usually handled in source, but we can standardise
// types if needed
#else
#include <pthread.h>
#include <stdlib.h>
typedef pthread_mutex_t gmem_mutex_t;
#define gmem_mutex_init(m) pthread_mutex_init(m, NULL)
#define gmem_mutex_destroy(m) pthread_mutex_destroy(m)
#define gmem_mutex_lock(m) pthread_mutex_lock(m)
#define gmem_mutex_unlock(m) pthread_mutex_unlock(m)

typedef pthread_t gmem_thread_t;
#endif

#endif // GMEM_OS_H
