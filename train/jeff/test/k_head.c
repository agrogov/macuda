// test/k_head.c — the `head` group of test_kernels (plan M2.0; the owning milestone is named below).
// Plan M2.5: ce_head with n_opts in {1, 2, 255}, dead options set to -1e9 before the softmax, and a one-option row whose
// loss and dlogits must be 0.
//
// M2.0 stub: it prints NOT IMPLEMENTED and returns JT_NOT_IMPL, so `test_kernels head` exits 2, while
// `test_kernels all` and the no-argument run behave as test/harness.h and the plan's M2.0 say: all fails on a stub, no
// arguments skips it and exits 0. The task that writes this group replaces this whole file with the real group:
//
//   int k_head(void) { ... rnd_fill/CHECK_REL/CHECK_ABS/dev_upload/jt_grad_check ...; return jt_exit_code(); }
//
// The dispatchers are not edited again, so the file name, the function name and the return-code contract are fixed.
#include <stdio.h>

#include "harness.h"

int k_head(void) {
  printf("NOT IMPLEMENTED\n");
  fflush(stdout);
  return JT_NOT_IMPL;
}
