// src/util.h — the two error macros every GPU call goes through, the debug-sync macro, and a host clock (design §12).
// CK(x) wraps a cudart call, CB(x) wraps a cuBLAS call: both print file:line and the runtime's own error string, then
// exit(1). Nothing is swallowed and nothing is ignored, so a failure names the line that caused it.
#ifndef JT_UTIL_H
#define JT_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef JT_CPU
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x)                                                                     \
  do {                                                                            \
    cudaError_t jt_err_ = (x);                                                    \
    if (jt_err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "%s:%d: CUDA error %d (%s) in %s\n", __FILE__, __LINE__,    \
              (int)jt_err_, cudaGetErrorString(jt_err_), #x);                     \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

#define CB(x)                                                                             \
  do {                                                                                     \
    cublasStatus_t jt_st_ = (x);                                                           \
    if (jt_st_ != CUBLAS_STATUS_SUCCESS) {                                                 \
      fprintf(stderr, "%s:%d: cuBLAS error %d (%s) in %s\n", __FILE__, __LINE__,           \
              (int)jt_st_, cublasGetStatusString(jt_st_), #x);                             \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

// After every kernel launch: JT_SYNC_DEBUG; — a no-op in a normal build, a launch error plus a device sync with
// -DJT_DEBUG=1 (design §12).
#ifdef JT_DEBUG
#define JT_SYNC_DEBUG                                                              \
  do {                                                                             \
    CK(cudaGetLastError());                                                        \
    CK(cudaDeviceSynchronize());                                                   \
  } while (0)
#else
#define JT_SYNC_DEBUG do { } while (0)
#endif
#endif // !JT_CPU

// Seconds on a monotonic host clock. CUDA events are not allowed (the shim has no cudaEventElapsedTime): time on the
// host around a cudaDeviceSynchronize() instead (design §12).
static inline double jt_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif // JT_UTIL_H
