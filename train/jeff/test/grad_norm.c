// test/grad_norm.c — the `norm` group of gradcheck (plan M2.0; the owning milestone is named below).
// Plan M2.2: rmsnorm_bwd, gated_rmsnorm_bwd (the input x and the gate z) and l2norm_bwd.
//
// M2.0 stub: it prints NOT IMPLEMENTED and returns JT_NOT_IMPL, so `gradcheck norm` exits 2, while
// `gradcheck all` and the no-argument run behave as test/harness.h and the plan's M2.0 say: all fails on a stub, no
// arguments skips it and exits 0. The task that writes this group replaces this whole file with the real group:
//
//   int grad_norm(void) { ... rnd_fill/CHECK_REL/CHECK_ABS/dev_upload/jt_grad_check ...; return jt_exit_code(); }
//
// The dispatchers are not edited again, so the file name, the function name and the return-code contract are fixed.
#include <stdio.h>

#include "harness.h"

int grad_norm(void) {
  printf("NOT IMPLEMENTED\n");
  fflush(stdout);
  return JT_NOT_IMPL;
}
