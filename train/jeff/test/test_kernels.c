// test/test_kernels.c — the GPU kernel tests, dispatched by group (plan M2.0).
//
// One group per file, so parallel tasks never share a source file:
//
//   test_kernels <group>    one group: exit 0 (passed), 1 (a check failed), 2 (stub: NOT IMPLEMENTED)
//   test_kernels all        every group; a stub counts as a failure, because card checkpoint m2 runs exactly this line
//   test_kernels            (no argument) the implemented groups only, stubs skipped, exit 0 — what make test-null runs
//
// The group list, the return codes and the CHECK_* macros are in test/harness.h; the group function names below are the
// whole interface between this file and test/k_<group>.c, so this dispatcher is never edited again.
#include <stdio.h>
#include <string.h>

#include "harness.h"

// One function per group file (each returns JT_OK, JT_FAIL or JT_NOT_IMPL; see harness.h).
int k_misc(void); // plan M2.1: zero, add_inplace, scale, bf16_to_f32, embed_gather, qgate_split/merge, kv_expand/reduce,
                  // to_head_major/from_head_major
int k_norm(void); // plan M2.2: rmsnorm_fwd/bwd, gated_rmsnorm_fwd/bwd, l2norm_fwd/bwd
int k_act(void);  // plan M2.3: silu_mul_fwd/bwd, sigmoid_gate_fwd/bwd, gdn_gates_fwd/bwd
int k_rope(void); // plan M2.4: rope_tables, rope_fwd/bwd
int k_conv(void); // plan M2.4: conv_fwd/bwd, including the row-boundary check
int k_attn(void); // plan M2.5: attn_softmax_fwd/bwd
int k_head(void); // plan M2.5: ce_head
int k_adam(void); // plan M2.6: sumsq_partial/final, clip_scale, adamw_step, and the flat buffer's size

typedef struct {
  const char *name;
  int (*run)(void);
} jt_group;

static const jt_group GROUPS[] = {
    {"misc", k_misc}, {"norm", k_norm}, {"act", k_act},   {"rope", k_rope},
    {"conv", k_conv}, {"attn", k_attn}, {"head", k_head}, {"adam", k_adam},
};
#define NGROUPS ((int)(sizeof(GROUPS) / sizeof(GROUPS[0])))

static void usage(FILE *f) {
  fprintf(f, "usage: test_kernels [group|all]\n  groups:");
  for (int i = 0; i < NGROUPS; i++) fprintf(f, " %s", GROUPS[i].name);
  fprintf(f,
          "\n  no argument  run the implemented groups, skip the stubs, exit 0 (make test-null)\n"
          "  all          run every group; a stub or a failed check fails (card checkpoint m2)\n");
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

// A group's return value decides the exit code, and only the three documented values do: anything else is a failure.
static int group_rc(int rc) { return rc == JT_OK ? JT_OK : (rc == JT_NOT_IMPL ? JT_NOT_IMPL : JT_FAIL); }

// strict (argv[1] == "all"): a stub fails, because the checkpoint that runs it expects real numbers.
// not strict (no arguments): a stub is skipped, because the null device run only proves that what exists launches.
static int run_all(int strict) {
  int passed = 0, failed = 0, stubs = 0;
  char fail_names[256] = "", stub_names[256] = "";

  for (int i = 0; i < NGROUPS; i++) {
    printf("== %s\n", GROUPS[i].name);
    fflush(stdout);
    int rc = group_rc(GROUPS[i].run());
    if (rc == JT_NOT_IMPL) {
      stubs++;
      add_name(stub_names, sizeof stub_names, GROUPS[i].name);
      if (!strict) printf("skip %s (not implemented)\n", GROUPS[i].name);
    } else if (rc == JT_OK) {
      passed++;
    } else {
      failed++;
      add_name(fail_names, sizeof fail_names, GROUPS[i].name);
    }
  }

  if (failed) printf("kernels FAIL: %d group(s) failed:%s\n", failed, fail_names);
  if (strict && stubs) printf("kernels FAIL all: %d group(s) not implemented:%s\n", stubs, stub_names);
  if (failed || (strict && stubs)) return JT_FAIL;

  if (strict)
    printf("kernels OK all: %d group(s)\n", passed);
  else
    printf("kernels OK: %d implemented group(s) passed, %d stub(s) skipped\n", passed, stubs);
  return JT_OK;
}

int main(int argc, char **argv) {
  if (argc == 1) return run_all(0);
  if (argc > 2) {
    usage(stderr);
    return JT_USAGE;
  }
  if (strcmp(argv[1], "all") == 0) return run_all(1);
  if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
    usage(stdout);
    return JT_OK;
  }

  const jt_group *g = find_group(argv[1]);
  if (g == NULL) {
    fprintf(stderr, "test_kernels: unknown group '%s'\n", argv[1]);
    usage(stderr);
    return JT_USAGE;
  }
  printf("== %s\n", g->name);
  fflush(stdout);
  return group_rc(g->run());
}
