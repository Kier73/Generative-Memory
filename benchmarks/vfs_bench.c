#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define FILE_PATH "G:\\gmem_100tb.raw"
#define BLOCK_SIZE (1024 * 1024) // 1MB
#define TOTAL_MB 1024            // 1GB

int main() {
  FILE *f = fopen(FILE_PATH, "rb");
  if (!f) {
    printf("Error: Could not open %s\n", FILE_PATH);
    return 1;
  }

  void *buffer = malloc(BLOCK_SIZE);
  if (!buffer)
    return 1;

  printf("Benchmarking Sequential Read: %d MB from %s\n", TOTAL_MB, FILE_PATH);

  LARGE_INTEGER frequency;
  LARGE_INTEGER start, end;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&start);

  for (int i = 0; i < TOTAL_MB; i++) {
    size_t read = fread(buffer, 1, BLOCK_SIZE, f);
    if (read != BLOCK_SIZE) {
      printf("\nRead error at MB %d (read %zu)\n", i, read);
      break;
    }
    if (i % 100 == 0)
      printf(".");
  }

  QueryPerformanceCounter(&end);
  double duration =
      (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
  double mb_per_sec = (double)TOTAL_MB / duration;

  printf("\nCompleted in %.4f seconds\n", duration);
  printf("Read Throughput: %.2f MB/s\n", mb_per_sec);

  fclose(f);
  free(buffer);
  return 0;
}
