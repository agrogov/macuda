// ref/rope.c — the CPU reference of RoPE (plan M2.4; design §3.3, §4.7). The loops of design §3.3 written literally;
// kernels/rope.cu is the same arithmetic with one thread per (token, head, j < 32), so the forward gate is 1e-5 and
// there is no reason for the two to differ by more than a rounding of the last bit.
//
// The table builder is deliberately written in fp32 even in the double build: the tables the model uses are fp32
// (design §3.3: "Compute inv_freq in fp32 as torch does ... then cos/sin in fp32"), so the double build only needs
// tables that are a faithful fp32 table too — the gradcheck differentiates the rotation, not the table.
#include "rope.h"

#include <math.h>

// inv_freq[k] = 1 / theta^(2k/dim), theta = 1e7 (config.json's rope_theta, design §3.3), computed in fp32 in the order
// torch uses: base ** (arange(0, dim, 2).float() / dim), then the reciprocal. ang = p * inv_freq in fp32.
void ref_rope_tables(real *cos, real *sin, int lmax) {
  for (int k = 0; k < JT_ROPE_HALF; k++) {
    const float inv = 1.0f / powf(1e7f, (float)(2 * k) / (float)JT_ROPE_DIM);
    for (int p = 0; p < lmax; p++) {
      const float ang = (float)p * inv;
      const float c = cosf(ang), s = sinf(ang);
      // torch's cat((f, f), dim=-1): slot j and slot j + 32 carry the angle of frequency j.
      cos[(size_t)p * JT_ROPE_DIM + k] = REAL(c);
      cos[(size_t)p * JT_ROPE_DIM + k + JT_ROPE_HALF] = REAL(c);
      sin[(size_t)p * JT_ROPE_DIM + k] = REAL(s);
      sin[(size_t)p * JT_ROPE_DIM + k + JT_ROPE_HALF] = REAL(s);
    }
  }
}

// The rotation of one head, in place. sgn is +1 for the forward (y = R x) and -1 for the backward (y = R^T x), which is
// the same expression with sin negated (design §4.7: "same formula with s -> -s").
static void rope_rot(real *x, int n, int heads, int head_dim, const int *row_off, const int *row_of, const real *cos,
                     const real *sin, real sgn) {
  if (head_dim < JT_ROPE_DIM) return; // the model's head_dim is 256; a head shorter than the rotary width has no RoPE
  const size_t width = (size_t)heads * (size_t)head_dim;
  for (int t = 0; t < n; t++) {
    const int p = t - row_off[row_of[t]]; // position inside the row, never the packed index (design §7)
    const real *c = cos + (size_t)p * JT_ROPE_DIM;
    const real *s = sin + (size_t)p * JT_ROPE_DIM;
    for (int h = 0; h < heads; h++) {
      real *xh = x + (size_t)t * width + (size_t)h * (size_t)head_dim;
      for (int j = 0; j < JT_ROPE_HALF; j++) {
        const real a = xh[j], b = xh[j + JT_ROPE_HALF];
        xh[j] = a * c[j] - b * (s[j] * sgn);
        xh[j + JT_ROPE_HALF] = b * c[j] + a * (s[j] * sgn);
      }
    }
  }
}

void ref_rope_fwd(real *x, const int *row_off, const int *row_of, int n, int heads, int head_dim, const real *cos,
                  const real *sin) {
  rope_rot(x, n, heads, head_dim, row_off, row_of, cos, sin, REAL(1));
}

void ref_rope_bwd(real *dx, const int *row_off, const int *row_of, int n, int heads, int head_dim, const real *cos,
                  const real *sin) {
  rope_rot(dx, n, heads, head_dim, row_off, row_of, cos, sin, REAL(-1));
}
