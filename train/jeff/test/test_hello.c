// test/test_hello.c — M0.2's proof that kernels and host code are linked and launched: fill 1M floats on the device,
// copy them back, check every one of them is 3.0f.
//
// On the shim's null device nothing computes: the launch is reported, the memory is whatever it was. With -DJT_SHIM the
// value check is therefore skipped unless the hardware path is actually armed (TINYNV_HW and TINYNV_SOCKET, both set;
// design §11.2) and the program says so and exits 0 (README.md's conventions).
#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "misc.h"

#define N ((size_t)1 << 20)

static int has_card(void) {
#if defined(JT_SHIM)
  const char *hw = getenv("TINYNV_HW");
  const char *sock = getenv("TINYNV_SOCKET");
  return hw != NULL && hw[0] != '\0' && sock != NULL && sock[0] != '\0';
#else
  return 1;
#endif
}

int main(void) {
  float *host = malloc(N * sizeof *host);
  float *dev = NULL;
  if (host == NULL) { fprintf(stderr, "test_hello: out of host memory\n"); return 1; }

  for (size_t i = 0; i < N; i++) host[i] = -1.0f;
  CK(cudaMalloc((void **)&dev, N * sizeof *dev));
  CK(cudaMemcpy(dev, host, N * sizeof *host, cudaMemcpyHostToDevice));

  jt_fill(dev, 3.0f, N, 0);
  JT_SYNC_DEBUG;
  CK(cudaMemcpy(host, dev, N * sizeof *host, cudaMemcpyDeviceToHost));
  CK(cudaFree(dev));

  if (!has_card()) {
    printf("hello OK (null device: launch only)\n");
    free(host);
    return 0;
  }
  for (size_t i = 0; i < N; i++) {
    if (host[i] != 3.0f) {
      printf("FAIL test_hello: x[%zu] = %g, want 3\n", i, (double)host[i]);
      free(host);
      return 1;
    }
  }
  printf("hello OK\n");
  free(host);
  return 0;
}
