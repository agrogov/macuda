// test/k_misc.c — the `misc` group of test_kernels (plan M2.0; the owning milestone is named below).
// Plan M2.1: zero, add_inplace, scale, bf16_to_f32, embed_gather(ids, table_bf16, out), qgate_split/qgate_merge,
// kv_expand/kv_reduce and to_head_major/from_head_major; qgate_merge(qgate_split(x)) must be the identity and kv_reduce
// the adjoint of kv_expand.
//
// M2.0 stub: it prints NOT IMPLEMENTED and returns JT_NOT_IMPL, so `test_kernels misc` exits 2, while
// `test_kernels all` and the no-argument run behave as test/harness.h and the plan's M2.0 say: all fails on a stub, no
// arguments skips it and exits 0. The task that writes this group replaces this whole file with the real group:
//
//   int k_misc(void) { ... rnd_fill/CHECK_REL/CHECK_ABS/dev_upload/jt_grad_check ...; return jt_exit_code(); }
//
// The dispatchers are not edited again, so the file name, the function name and the return-code contract are fixed.
#include <stdio.h>

#include "harness.h"

int k_misc(void) {
  printf("NOT IMPLEMENTED\n");
  fflush(stdout);
  return JT_NOT_IMPL;
}
