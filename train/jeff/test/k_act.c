// test/k_act.c — the `act` group of test_kernels (plan M2.0; the owning milestone is named below).
// Plan M2.3: silu_mul_fwd/bwd, sigmoid_gate_fwd/bwd and gdn_gates_fwd/bwd, with softplus as u > 20 ? u : log1p(exp(u)) and
// the inputs kept away from that threshold.
//
// M2.0 stub: it prints NOT IMPLEMENTED and returns JT_NOT_IMPL, so `test_kernels act` exits 2, while
// `test_kernels all` and the no-argument run behave as test/harness.h and the plan's M2.0 say: all fails on a stub, no
// arguments skips it and exits 0. The task that writes this group replaces this whole file with the real group:
//
//   int k_act(void) { ... rnd_fill/CHECK_REL/CHECK_ABS/dev_upload/jt_grad_check ...; return jt_exit_code(); }
//
// The dispatchers are not edited again, so the file name, the function name and the return-code contract are fixed.
#include <stdio.h>

#include "harness.h"

int k_act(void) {
  printf("NOT IMPLEMENTED\n");
  fflush(stdout);
  return JT_NOT_IMPL;
}
