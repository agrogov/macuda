// test/k_adam.c — the `adam` group of test_kernels (plan M2.0; the owning milestone is named below).
// Plan M2.6: sumsq_partial/sumsq_final and clip_scale (deterministic two-pass, no atomics), adamw_step (one step against
// torch's formula, then 10 with random grads, rel 1e-6), lr_at against a table printed by Python, and the flat buffer's
// trainable count 10,484,736.
//
// M2.0 stub: it prints NOT IMPLEMENTED and returns JT_NOT_IMPL, so `test_kernels adam` exits 2, while
// `test_kernels all` and the no-argument run behave as test/harness.h and the plan's M2.0 say: all fails on a stub, no
// arguments skips it and exits 0. The task that writes this group replaces this whole file with the real group:
//
//   int k_adam(void) { ... rnd_fill/CHECK_REL/CHECK_ABS/dev_upload/jt_grad_check ...; return jt_exit_code(); }
//
// The dispatchers are not edited again, so the file name, the function name and the return-code contract are fixed.
#include <stdio.h>

#include "harness.h"

int k_adam(void) {
  printf("NOT IMPLEMENTED\n");
  fflush(stdout);
  return JT_NOT_IMPL;
}
