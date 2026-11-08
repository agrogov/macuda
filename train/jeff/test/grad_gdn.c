// test/grad_gdn.c — the `gdn` group of gradcheck (plan M2.0; the owning milestone is named below).
// Plan M3.1, design §3.6/§4.8: ref_gdn_bwd's five input gradients (dq, dk, dv, dg, dbeta) with H=2, Dk=Dv=8, B=2 rows of
// lengths {5,3}, g in [-1,0] and beta in [0,1]. M3.1 owns this group (J16/J17), not M2.0.
//
// M2.0 stub: it prints NOT IMPLEMENTED and returns JT_NOT_IMPL, so `gradcheck gdn` exits 2, while
// `gradcheck all` and the no-argument run behave as test/harness.h and the plan's M2.0 say: all fails on a stub, no
// arguments skips it and exits 0. The task that writes this group replaces this whole file with the real group:
//
//   int grad_gdn(void) { ... rnd_fill/CHECK_REL/CHECK_ABS/dev_upload/jt_grad_check ...; return jt_exit_code(); }
//
// The dispatchers are not edited again, so the file name, the function name and the return-code contract are fixed.
#include <stdio.h>

#include "harness.h"

int grad_gdn(void) {
  printf("NOT IMPLEMENTED\n");
  fflush(stdout);
  return JT_NOT_IMPL;
}
