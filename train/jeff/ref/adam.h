// ref/adam.h — the CPU oracle for the optimizer kernels (plan M2.6, design §2.2 and §8.1).
//
// Written before the kernels (plan rule 2): k_sumsq_partial/k_sumsq_final, k_clip_scale and k_adamw_step are compared
// against these, and test/k_adam.c is the only caller. `real` is float by default and double under -DJT_REAL_DOUBLE
// (ref/real.h); nothing here has to know which build it is in.
//
// Design §2.2, word for word (weight_decay 0, so it is Adam):
//
//   m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2
//   mhat = m/(1-b1^t);    vhat = v/(1-b2^t)              t starts at 1
//   p -= lr * mhat / (sqrt(vhat) + eps)
//
// with one lr per element: index < lr_group_split is LoRA (both groups' A and B), the rest is the readout, which sits at
// the end of the same flat buffer (design §8.1). The bias corrections are computed on the host in double and handed over
// as floats (plan M2.6), so ref_adam_bias_corr exists to check that conversion, not to be called inside the step.
//
// The gradient norm is torch's clip_grad_norm_ (train.py:334): the global L2 norm over every trainable tensor, then
//   coef = 1.0 / (norm + 1e-6);  if coef < 1: g *= coef
// summed as two deterministic passes with a fixed block count and no atomics (design §8.1): partial sums per block, then
// one pass over the partials. Float addition is not associative, so an atomic reduction would give a different value
// from run to run and could not be compared with a reference at all.
#ifndef JT_REF_ADAM_H
#define JT_REF_ADAM_H

#include <stddef.h>

#include "real.h"

// torch's clipping constant (design §2.2 / train.py's clip_grad_norm_(params, 1.0) with its default eps).
#define REF_ADAM_CLIP_EPS 1e-6
#define REF_ADAM_MAX_NORM 1.0

// The default block count of the two-pass reduction; the kernel's K_ADAM_BLOCKS is the same number, and every reference
// below takes the count as a parameter so a test can also check a small, ragged one.
#define REF_ADAM_BLOCKS 1024

// First pass: partials[b] = sum of g[i]^2 over i = b, b + nblocks, b + 2*nblocks, ... < n (the slices a grid-stride
// block of nblocks blocks owns when block b starts at element b). partials has nblocks entries.
void ref_sumsq_partial(const real *g, size_t n, real *partials, int nblocks);
// Second pass: *out = partials[0] + partials[1] + ... in index order.
void ref_sumsq_final(const real *partials, int nblocks, real *out);
// Both passes in one call: *out = sum_i g[i]^2. (The kernel needs two launches because one block cannot see another's
// result; this is the same arithmetic.)
void ref_sumsq(const real *g, size_t n, int nblocks, real *out);

// coef = min(1, 1 / (norm + 1e-6)): design §2.2's clipping coefficient for a global norm of `norm`.
real ref_clip_coef(real norm);
// g[i] *= coef, i < n.
void ref_clip_scale(real *g, size_t n, real coef);

// bias1 = 1 - b1^t, bias2 = 1 - b2^t, computed in double (t = step_t, the step about to be taken, starting at 1) and
// rounded to `real` — the host's job in design §2.2, and the value the kernel is handed as a float.
void ref_adam_bias_corr(double b1, double b2, int step_t, real *bias1, real *bias2);

// One Adam step (design §2.2) over a flat buffer of n elements:
//   p[i], m[i], v[i] are updated in place; g[i] is read only.
//   i < lr_group_split: lr = lr_lora (every LoRA A and B); else lr = lr_readout (the readout, at the end).
//   bias1/bias2 are the pre-computed 1-b^t values (ref_adam_bias_corr), b1/b2 the betas themselves.
void ref_adamw_step(real *p, const real *g, real *m, real *v, size_t n, size_t lr_group_split, real lr_lora,
                    real lr_readout, real b1, real b2, real eps, real bias1, real bias2);

#endif // JT_REF_ADAM_H
