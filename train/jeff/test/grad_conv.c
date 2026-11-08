// test/grad_conv.c — the `conv` group of gradcheck (plan M2.4, and the M2.0 numeric core in test/harness.h).
//
// Double precision, central difference h = 1e-6, gate rel < 1e-6, every element of the input:
//
//   conv_bwd   d/dqkv <dout, silu(conv(qkv))>, which must come out as the transposed causal conv of design §4.8
//
// conv_bwd is the adjoint of conv_fwd, so this is the check that catches a flipped tap order or a tap that reads across
// a row boundary in the backward. The rows are ragged ({2,3,2} and {1,4}) so the first taps of every row exercise the
// zero padding and the last taps of every short row exercise the row end.
//
// wconv is frozen (design §2.1: only LoRA A/B and the readout are trained), so there is no dwconv to check — the weight
// is a constant of the closure, exactly as it is in the trainer. pre is the forward's conv output at the unperturbed
// qkv, which is what the real backward has on hand (design §4.8 keeps it), so silu's derivative is not recomputed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"

#include "../ref/conv.h" // ref_conv_fwd / ref_conv_bwd, double under -DJT_REAL_DOUBLE

enum { JT_G_TOK = 8, JT_G_CH = 4, JT_G_ROWS = 4 };

// Zeroed and never NULL: every allocation below is small and fixed, and a failed one ends the run with a message.
static real *rnew(size_t n) {
  real *p = (real *)calloc(n ? n : 1, sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "grad_conv: out of memory (%zu reals)\n", n);
    exit(1);
  }
  return p;
}

typedef struct {
  int n, channels;
  int row_off[JT_G_ROWS + 1], row_of[JT_G_TOK];
  real wconv[JT_G_CH * JT_CONV_K];
  real pre[JT_G_TOK * JT_G_CH];  // the forward's conv output at the unperturbed input
  real dout[JT_G_TOK * JT_G_CH]; // the cotangent
  real scratch_pre[JT_G_TOK * JT_G_CH];
} conv_ctx;

// y = silu(conv(x)): the forward whose transpose conv_bwd must be.
static void conv_as_qkv(void *v, const real *x, real *y) {
  conv_ctx *c = (conv_ctx *)v;
  ref_conv_fwd(x, c->wconv, c->row_of, c->row_off, c->scratch_pre, y, c->n, c->channels);
}

static void check_conv(const char *tag, const int *lens, int nrows, int channels, uint32_t seed) {
  conv_ctx c;
  char name[96];
  c.channels = channels;
  c.n = 0;
  c.row_off[0] = 0;
  for (int b = 0; b < nrows; b++) {
    for (int t = 0; t < lens[b]; t++) c.row_of[c.n + t] = b;
    c.n += lens[b];
    c.row_off[b + 1] = c.n;
  }
  if (c.n > JT_G_TOK || channels > JT_G_CH || nrows > JT_G_ROWS) {
    printf("grad_conv: %s is too large for the fixture\n", tag);
    jt_fail_inc();
    return;
  }

  const size_t n = (size_t)c.n * (size_t)channels;
  real *qkv = rnew(n), *gx = rnew(n), *out = rnew(n);

  rnd_fill(c.wconv, channels * JT_CONV_K, -1.0, 1.0, seed);
  rnd_fill(qkv, n, -1.5, 1.5, seed ^ 0x5bd1e995u);
  rnd_fill(c.dout, n, -1.0, 1.0, seed ^ 0x1b873593u);
  ref_conv_fwd(qkv, c.wconv, c.row_of, c.row_off, c.pre, out, c.n, channels); // pre at the unperturbed input
  ref_conv_bwd(c.dout, c.pre, c.wconv, c.row_of, c.row_off, gx, c.n, channels);

  snprintf(name, sizeof name, "conv/d_qkv %s n=%d ch=%d", tag, c.n, channels);
  jt_grad_check(name, conv_as_qkv, &c, qkv, n, gx, c.dout, n);

  free(qkv), free(gx), free(out);
}

int grad_conv(void) {
  static const int lens_a[3] = {2, 3, 2};
  static const int lens_b[2] = {1, 4};

  check_conv("ragged", lens_a, 3, 3, 0x2545f491u);
  check_conv("short", lens_b, 2, 2, 0x68e31da4u);
  return jt_exit_code();
}
