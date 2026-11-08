// test/grad_rope.c — the `rope` group of gradcheck (plan M2.4, and the M2.0 numeric core in test/harness.h).
//
// Double precision, central difference h = 1e-6, gate rel < 1e-6, every element of the input:
//
//   rope_bwd   d/dx <w, rope_fwd(x)>, which must come out as the transposed rotation (sin negated)
//
// The rotation has no weights and no second input, so x is the whole story. The closures copy x into the output before
// running the reference, because ref_rope_fwd rotates a buffer in place.
//
// Two shapes: the model's head (head_dim 256, only the first 64 dims rotate, the rest must have gradient exactly w) on a
// 2-row packed batch, and a head_dim 64 variant where the whole head rotates and a row is a single token (position 0,
// where the rotation is the identity — the gradient there must still be w, through sin = 0).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"

#include "../ref/rope.h" // ref_rope_tables / ref_rope_fwd / ref_rope_bwd, double under -DJT_REAL_DOUBLE

enum { JT_G_MAXTOK = 8, JT_G_MAXL = 4, JT_G_MAXH = 4 };

// Zeroed and never NULL: every allocation below is small and fixed, and a failed one ends the run with a message.
static real *rnew(size_t n) {
  real *p = (real *)calloc(n ? n : 1, sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "grad_rope: out of memory (%zu reals)\n", n);
    exit(1);
  }
  return p;
}

typedef struct {
  int n, heads, head_dim;
  int row_off[JT_G_MAXTOK + 1], row_of[JT_G_MAXTOK];
  real cos[(size_t)JT_G_MAXL * JT_ROPE_DIM];
  real sin[(size_t)JT_G_MAXL * JT_ROPE_DIM];
} rope_ctx;

// y = rope_fwd(x) on a fresh copy: the reference rotates in place.
static void rope_as_x(void *v, const real *x, real *y) {
  const rope_ctx *c = (const rope_ctx *)v;
  memcpy(y, x, (size_t)c->n * (size_t)c->heads * (size_t)c->head_dim * sizeof *y);
  ref_rope_fwd(y, c->row_off, c->row_of, c->n, c->heads, c->head_dim, c->cos, c->sin);
}

static void check_rope(const char *tag, const int *lens, int nrows, int heads, int head_dim, uint32_t seed) {
  rope_ctx c;
  char name[96];
  int lmax = 0;
  c.n = 0;
  c.heads = heads;
  c.head_dim = head_dim;
  c.row_off[0] = 0;
  for (int b = 0; b < nrows; b++) {
    if (lens[b] > lmax) lmax = lens[b];
    for (int t = 0; t < lens[b]; t++) c.row_of[c.n + t] = b;
    c.n += lens[b];
    c.row_off[b + 1] = c.n;
  }
  if (c.n > JT_G_MAXTOK || lmax > JT_G_MAXL || heads > JT_G_MAXH) {
    printf("grad_rope: %s is too large for the fixture\n", tag);
    jt_fail_inc();
    return;
  }

  const size_t n = (size_t)c.n * (size_t)heads * (size_t)head_dim;
  real *x = rnew(n), *w = rnew(n), *gx = rnew(n);

  ref_rope_tables(c.cos, c.sin, lmax);
  rnd_fill(x, n, -1.5, 1.5, seed);
  rnd_fill(w, n, -1.5, 1.5, seed ^ 0x9e3779b9u);
  memcpy(gx, w, n * sizeof *gx);
  ref_rope_bwd(gx, c.row_off, c.row_of, c.n, heads, head_dim, c.cos, c.sin); // analytic

  snprintf(name, sizeof name, "rope/d_x %s n=%d heads=%d d=%d", tag, c.n, heads, head_dim);
  jt_grad_check(name, rope_as_x, &c, x, n, gx, w, n);

  free(x), free(w), free(gx);
}

int grad_rope(void) {
  static const int lens_a[2] = {1, 2};
  static const int lens_b[3] = {2, 1, 1};

  check_rope("packed2", lens_a, 2, 2, 256, 0x51ed270bu);
  check_rope("d64", lens_b, 3, 1, 64, 0x1b873593u);
  return jt_exit_code();
}
