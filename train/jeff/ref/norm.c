// ref/norm.c — the CPU references of the norm kernels (plan M2.2). The loops of design §4.3, §4.4 and §4.8 written out
// literally, in `real` (ref/real.h): float for the GPU comparison, double for the gradcheck.
//
// These are the oracle, not a bit-for-bit copy of the device: row sums accumulate in double and go back through sqrt
// once, so in the float build r differs from the kernel's rsqrtf() by a couple of ulp — three orders of magnitude below
// the 1e-5 forward gate — and in the double build the reference is exact enough for the 1e-6 finite-difference gate.
#include "norm.h"

// inv_rms over a row (mean of squares, design §3.3) and inv_l2 over a row (sum of squares, l.294): one sqrt each.
static real row_inv_rms(const real *x, int d) {
  double ss = 0.0;
  for (int i = 0; i < d; i++) ss += (double)x[i] * (double)x[i];
  return REAL(1.0 / sqrt(ss / (double)d + (double)JT_NORM_EPS));
}

static real row_inv_l2(const real *x, int d) {
  double ss = 0.0;
  for (int i = 0; i < d; i++) ss += (double)x[i] * (double)x[i];
  return REAL(1.0 / sqrt(ss + (double)JT_NORM_EPS));
}

// (1 + w) for the zero-centred norms, w for the GDN gated norm (design §3.3 / §4.4).
static real w_scale(real w, int zero_centred) { return zero_centred ? REAL(1.0 + (double)w) : w; }

// sg = silu(z) = z * sigmoid(z) and ds = its derivative, sigmoid(z) * (1 + z * (1 - sigmoid(z))). The form of silu
// torch uses, so a large negative z gives exactly 0 rather than inf/inf.
static void silu_pair(real z, real *sg, real *ds) {
  double zd = (double)z;
  double sig = 1.0 / (1.0 + exp(-zd));
  if (sg != NULL) *sg = REAL(zd * sig);
  if (ds != NULL) *ds = REAL(sig * (1.0 + zd * (1.0 - sig)));
}

// ---- RMSNorm (design §4.3) ------------------------------------------------------------------------------------------
void ref_rmsnorm_fwd(const real *x, const real *w, real *y, real *r, int rows, int d, int zero_centred) {
  for (int b = 0; b < rows; b++) {
    const real *xr = x + (size_t)b * (size_t)d;
    real *yr = y + (size_t)b * (size_t)d;
    real rb = row_inv_rms(xr, d);
    r[b] = rb;
    for (int i = 0; i < d; i++) yr[i] = REAL((double)xr[i] * (double)rb * (double)w_scale(w[i], zero_centred));
  }
}

void ref_rmsnorm_bwd(const real *x, const real *w, const real *dy, const real *r, real *dx, int rows, int d,
                     int zero_centred) {
  for (int b = 0; b < rows; b++) {
    const real *xr = x + (size_t)b * (size_t)d;
    const real *dr = dy + (size_t)b * (size_t)d;
    real *dxr = dx + (size_t)b * (size_t)d;
    double dot = 0.0; // sum over the row of gy * x, gy = dy * wp
    for (int i = 0; i < d; i++) dot += (double)dr[i] * (double)w_scale(w[i], zero_centred) * (double)xr[i];
    double mean = dot / (double)d;
    double rb = (double)r[b];
    for (int i = 0; i < d; i++) {
      double gy = (double)dr[i] * (double)w_scale(w[i], zero_centred);
      dxr[i] = REAL(rb * (gy - (double)xr[i] * rb * rb * mean));
    }
  }
}

// ---- Gated RMSNorm (design §4.4) ------------------------------------------------------------------------------------
void ref_gated_rmsnorm_fwd(const real *x, const real *w, const real *z, real *y, real *r, int rows, int d,
                           int zero_centred) {
  for (int b = 0; b < rows; b++) {
    const real *xr = x + (size_t)b * (size_t)d;
    const real *zr = z + (size_t)b * (size_t)d;
    real *yr = y + (size_t)b * (size_t)d;
    real rb = row_inv_rms(xr, d);
    r[b] = rb;
    for (int i = 0; i < d; i++) {
      real sg;
      silu_pair(zr[i], &sg, NULL);
      double n = (double)xr[i] * (double)rb;
      yr[i] = REAL((double)w_scale(w[i], zero_centred) * n * (double)sg);
    }
  }
}

void ref_gated_rmsnorm_bwd(const real *x, const real *w, const real *z, const real *dy, const real *r, real *dx,
                           real *dz, int rows, int d, int zero_centred) {
  for (int b = 0; b < rows; b++) {
    const real *xr = x + (size_t)b * (size_t)d;
    const real *zr = z + (size_t)b * (size_t)d;
    const real *dr = dy + (size_t)b * (size_t)d;
    real *dxr = dx + (size_t)b * (size_t)d;
    real *dzr = dz + (size_t)b * (size_t)d;
    double rb = (double)r[b];
    double dot = 0.0; // sum of gn * n over the row, gn = dy * wp * sg, n = x * r
    for (int i = 0; i < d; i++) {
      real sg;
      silu_pair(zr[i], &sg, NULL);
      double n = (double)xr[i] * rb;
      double gn = (double)dr[i] * (double)w_scale(w[i], zero_centred) * (double)sg;
      dot += gn * n;
    }
    double mean = dot / (double)d;
    for (int i = 0; i < d; i++) {
      real sg, ds;
      silu_pair(zr[i], &sg, &ds);
      double n = (double)xr[i] * rb;
      double wp = (double)w_scale(w[i], zero_centred);
      double gn = (double)dr[i] * wp * (double)sg;
      dxr[i] = REAL(rb * (gn - n * mean));
      dzr[i] = REAL((double)dr[i] * wp * n * (double)ds);
    }
  }
}

// ---- L2 norm (design §3.3 l.294, §4.8) ------------------------------------------------------------------------------
void ref_l2norm_fwd(const real *x, real post_scale, real *y, real *r, int rows, int d) {
  for (int b = 0; b < rows; b++) {
    const real *xr = x + (size_t)b * (size_t)d;
    real *yr = y + (size_t)b * (size_t)d;
    real rb = row_inv_l2(xr, d);
    r[b] = rb;
    for (int i = 0; i < d; i++) yr[i] = REAL((double)xr[i] * (double)rb * (double)post_scale);
  }
}

void ref_l2norm_bwd(const real *x, const real *dy, const real *r, real post_scale, real *dx, int rows, int d) {
  for (int b = 0; b < rows; b++) {
    const real *xr = x + (size_t)b * (size_t)d;
    const real *dr = dy + (size_t)b * (size_t)d;
    real *dxr = dx + (size_t)b * (size_t)d;
    double rb = (double)r[b];
    double dot = 0.0; // qn . dqn, qn = x * r, dqn = dy * post_scale
    for (int i = 0; i < d; i++) dot += (double)xr[i] * rb * ((double)dr[i] * (double)post_scale);
    for (int i = 0; i < d; i++) {
      double qn = (double)xr[i] * rb;
      double dqn = (double)dr[i] * (double)post_scale;
      dxr[i] = REAL(rb * (dqn - qn * dot));
    }
  }
}
