// ref/real.h — the one type every CPU reference (ref/*.c) and every gradcheck works in (plan M2.0).
//
// Default is fp32 (`real` = float), which is what the GPU tests compare against. `make test-cpu` builds ref/ a second
// time with -DJT_REAL_DOUBLE for the gradchecks: the finite-difference gate there is rel < 1e-6, and in fp32 the
// difference of two nearby function values only has about seven digits. Nothing else changes between the two builds, so
// every reference is written once, in terms of `real` and the r_* wrappers below.
//
// The r_* wrappers evaluate in double and cast back. A reference is the oracle, not a bit-for-bit copy of the GPU: in
// the float build sqrt/exp/log... then differ from the device's float versions by at most one ulp, which is three
// orders of magnitude below the 1e-5 forward gate. In the double build they are the double functions themselves.
#ifndef JT_REAL_H
#define JT_REAL_H

#include <math.h>

#ifdef JT_REAL_DOUBLE
typedef double real;
#define JT_REAL_IS_DOUBLE 1
// %.17g round-trips a double; %.9g round-trips a float.
#define JT_RFMT "%.17g"
#else
typedef float real;
#define JT_REAL_IS_DOUBLE 0
#define JT_RFMT "%.9g"
#endif

// A literal or an int/double expression as `real`: REAL(1), REAL(1e-6), REAL(0.5 * x).
#define REAL(x) ((real)(x))

// The C library functions a reference needs, in `real`: always computed in double, rounded back once.
static inline real r_fabs(real x) { return REAL(fabs((double)x)); }
static inline real r_sqrt(real x) { return REAL(sqrt((double)x)); }
static inline real r_exp(real x) { return REAL(exp((double)x)); }
static inline real r_expm1(real x) { return REAL(expm1((double)x)); }
static inline real r_log(real x) { return REAL(log((double)x)); }
static inline real r_log1p(real x) { return REAL(log1p((double)x)); }
static inline real r_tanh(real x) { return REAL(tanh((double)x)); }
static inline real r_pow(real x, real y) { return REAL(pow((double)x, (double)y)); }
static inline real r_floor(real x) { return REAL(floor((double)x)); }
static inline real r_ceil(real x) { return REAL(ceil((double)x)); }
static inline real r_fmin(real a, real b) { return REAL(fmin((double)a, (double)b)); }
static inline real r_fmax(real a, real b) { return REAL(fmax((double)a, (double)b)); }

#endif // JT_REAL_H
