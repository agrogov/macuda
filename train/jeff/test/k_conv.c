// test/k_conv.c — the `conv` group of test_kernels (plan M2.0; the owning milestone is named below).
// Plan M2.4: conv_fwd (pre = conv output, out = silu(pre)) and conv_bwd, wconv[ch*4+k] * x[t-3+k], taps before a row
// start reading zero, plus the ragged check that randomising row 1's last 3 tokens leaves row 2's first 3 outputs alone.
//
// M2.0 stub: it prints NOT IMPLEMENTED and returns JT_NOT_IMPL, so `test_kernels conv` exits 2, while
// `test_kernels all` and the no-argument run behave as test/harness.h and the plan's M2.0 say: all fails on a stub, no
// arguments skips it and exits 0. The task that writes this group replaces this whole file with the real group:
//
//   int k_conv(void) { ... rnd_fill/CHECK_REL/CHECK_ABS/dev_upload/jt_grad_check ...; return jt_exit_code(); }
//
// The dispatchers are not edited again, so the file name, the function name and the return-code contract are fixed.
#include <stdio.h>

#include "harness.h"

int k_conv(void) {
  printf("NOT IMPLEMENTED\n");
  fflush(stdout);
  return JT_NOT_IMPL;
}
