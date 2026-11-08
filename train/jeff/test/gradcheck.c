// test/gradcheck.c — the finite-difference checks of every backward reference (plan M2.0), dispatched by group.
//
// CPU only and double precision: `make test-cpu` (and every other build) compiles this file and test/grad_*.c with
// -DJT_REAL_DOUBLE and links the refd_*.o build of ref/. The numeric core is in test/harness.h (jt_fd_grad,
// jt_grad_check), so a group file only picks its random input and cotangent, calls ref_*_bwd for the analytic gradient
// and ref_*_fwd inside the closure.
//
//   gradcheck <group>   one group: exit 0 (passed), 1 (a gradient was wrong), 2 (stub: NOT IMPLEMENTED)
//   gradcheck all       every implemented group; a stub prints SKIP and does not fail, so make test-cpu stays green
//                       while the groups are still stubs (plan M2.0) — a real group that fails still fails it
//   gradcheck           (no argument) the same as `all`
//
// The group function names below are the whole interface between this file and test/grad_<group>.c.
#include <stdio.h>
#include <string.h>

#include "harness.h"

#if !JT_REAL_IS_DOUBLE
#error "gradcheck needs -DJT_REAL_DOUBLE (the double-precision build of ref/); the Makefile's gradcheck rule sets it"
#endif

int grad_norm(void); // plan M2.2: rmsnorm_bwd, gated_rmsnorm_bwd, l2norm_bwd (including the gated norm's z)
int grad_act(void);  // plan M2.3: silu_mul_bwd, sigmoid_gate_bwd, gdn_gates_bwd
int grad_rope(void); // plan M2.4: rope_bwd
int grad_conv(void); // plan M2.4: conv_bwd (the transposed conv, which must not cross a row start)
int grad_attn(void); // plan M2.5: attn_softmax_bwd
int grad_head(void); // plan M2.5: ce_head's dlogits, and the readout R
int grad_gdn(void);  // plan M3.1: ref_gdn_bwd, all five input gradients
int grad_lora(void); // plan M2.6/M5: the LoRA A and B gradients (and the readout's, through grad_head)

typedef struct {
  const char *name;
  int (*run)(void);
} jt_group;

static const jt_group GROUPS[] = {
    {"norm", grad_norm}, {"act", grad_act},   {"rope", grad_rope}, {"conv", grad_conv},
    {"attn", grad_attn}, {"head", grad_head}, {"gdn", grad_gdn},   {"lora", grad_lora},
};
#define NGROUPS ((int)(sizeof(GROUPS) / sizeof(GROUPS[0])))

static void usage(FILE *f) {
  fprintf(f, "usage: gradcheck [group|all]\n  groups:");
  for (int i = 0; i < NGROUPS; i++) fprintf(f, " %s", GROUPS[i].name);
  fprintf(f,
          "\n  no argument  the same as all: implemented groups run, stubs are skipped, exit 0\n"
          "  all          every implemented group; a stub prints SKIP and does not fail (make test-cpu)\n");
}

static const jt_group *find_group(const char *name) {
  for (int i = 0; i < NGROUPS; i++)
    if (strcmp(GROUPS[i].name, name) == 0) return &GROUPS[i];
  return NULL;
}

static void add_name(char *buf, size_t cap, const char *name) {
  size_t n = strlen(buf);
  if (n + 1 < cap) buf[n++] = ' ';
  if (n + 1 < cap) snprintf(buf + n, cap - n, "%s", name);
}

// Only the three documented values decide an exit code; anything else is a failure.
static int group_rc(int rc) { return rc == JT_OK ? JT_OK : (rc == JT_NOT_IMPL ? JT_NOT_IMPL : JT_FAIL); }

// `all`/no argument: a stub is skipped with a SKIP line (plan M2.0), an implemented group must pass.
static int run_all(void) {
  int passed = 0, failed = 0, stubs = 0;
  char fail_names[256] = "";

  for (int i = 0; i < NGROUPS; i++) {
    printf("== %s\n", GROUPS[i].name);
    fflush(stdout);
    int rc = group_rc(GROUPS[i].run());
    if (rc == JT_NOT_IMPL) {
      stubs++;
      printf("SKIP %s (not implemented)\n", GROUPS[i].name);
    } else if (rc == JT_OK) {
      passed++;
    } else {
      failed++;
      add_name(fail_names, sizeof fail_names, GROUPS[i].name);
    }
  }

  if (failed) {
    printf("gradcheck FAIL: %d group(s) failed:%s\n", failed, fail_names);
    return JT_FAIL;
  }
  printf("gradcheck OK: %d group(s) passed, %d skipped\n", passed, stubs);
  return JT_OK;
}

int main(int argc, char **argv) {
  printf("gradcheck: double precision, central difference h=%.0e, gate rel<%.0e\n", JT_FD_H, JT_TOL_GRAD);
  if (argc == 1) return run_all();
  if (argc > 2) {
    usage(stderr);
    return JT_USAGE;
  }
  if (strcmp(argv[1], "all") == 0) return run_all();
  if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
    usage(stdout);
    return JT_OK;
  }

  const jt_group *g = find_group(argv[1]);
  if (g == NULL) {
    fprintf(stderr, "gradcheck: unknown group '%s'\n", argv[1]);
    usage(stderr);
    return JT_USAGE;
  }
  printf("== %s\n", g->name);
  fflush(stdout);
  return group_rc(g->run());
}
