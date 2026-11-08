// ref/conv.c — the CPU reference of the GDN causal depthwise conv (plan M2.4; design §3.6 forward, §4.8 backward).
// kernels/conv.cu runs the same arithmetic with one thread per (token, channel), so the forward gate is 1e-5 and the
// backward's is 1e-4.
//
// The row bookkeeping is the same in both directions and comes from one place: row_of[t] names the row, row_off[b] and
// row_off[b+1] bracket its tokens, so local index tl = t - row_off[row_of[t]] and the row's length is row_off[b+1] -
// row_off[b]. A tap whose local index is negative (forward) or past the row's end (backward) contributes zero; nothing
// here ever reads a token of another row (design §7).
#include "conv.h"

// sigmoid and the silu pair, in `real`; the device versions use expf and are one ulp away at most.
static real sigmoid(real u) { return REAL(1) / (REAL(1) + r_exp(-u)); }
// silu(u) = u * sigmoid(u) and dsilu(u) = sigmoid(u) * (1 + u * (1 - sigmoid(u))), the derivative of u*sigmoid(u).
static real dsilu(real u) {
  const real sig = sigmoid(u);
  return sig * (REAL(1) + u * (REAL(1) - sig));
}

// dpre at (row-local tl, ch) = dout * dsilu(pre). Shared by the backward's four taps; computing it per tap instead of
// materialising a dpre array keeps the reference allocation-free, which matters because the gradcheck calls it
// thousands of times.
static real dpre_at(const real *dout, const real *pre, int tl, int ch, int channels) {
  const size_t i = (size_t)tl * (size_t)channels + (size_t)ch;
  return dout[i] * dsilu(pre[i]);
}

void ref_conv_fwd(const real *qkv, const real *wconv, const int *row_of, const int *row_off, real *pre, real *out,
                  int n, int channels) {
  for (int t = 0; t < n; t++) {
    const int b = row_of[t];
    const int start = row_off[b];
    const int tl = t - start;
    for (int ch = 0; ch < channels; ch++) {
      const real *w = wconv + (size_t)ch * JT_CONV_K;
      real acc = REAL(0);
      for (int k = 0; k < JT_CONV_K; k++) {
        const int tl2 = tl - (JT_CONV_K - 1) + k; // qkv[tl - 3 + k]
        if (tl2 < 0) continue;                   // the row's own left zero-padding (design §3.6: qkv[<0] = 0)
        acc += w[k] * qkv[(size_t)(start + tl2) * (size_t)channels + (size_t)ch];
      }
      const size_t i = (size_t)t * (size_t)channels + (size_t)ch;
      pre[i] = acc;
      out[i] = acc * sigmoid(acc);
    }
  }
}

void ref_conv_bwd(const real *dout, const real *pre, const real *wconv, const int *row_of, const int *row_off,
                  real *dqkv, int n, int channels) {
  for (int t = 0; t < n; t++) {
    const int b = row_of[t];
    const int start = row_off[b];
    const int len = row_off[b + 1] - start;
    const int tl = t - start;
    for (int ch = 0; ch < channels; ch++) {
      const real *w = wconv + (size_t)ch * JT_CONV_K;
      real acc = REAL(0);
      for (int k = 0; k < JT_CONV_K; k++) {
        const int tl2 = tl + (JT_CONV_K - 1) - k; // dpre[tl + 3 - k]
        if (tl2 >= len) continue;                 // past this row's last token: zero, never the next row (design §4.8)
        acc += w[k] * dpre_at(dout, pre, start + tl2, ch, channels);
      }
      dqkv[(size_t)t * (size_t)channels + (size_t)ch] = acc;
    }
  }
}
