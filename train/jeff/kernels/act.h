// kernels/act.h — activation and gate kernels (plan M2.3, design §8.1). Host C includes this; every wrapper is
// extern "C" and takes plain pointers, sizes and a stream (0 = the default stream), so the C side never sees CUDA
// syntax (design §11.1). The CPU references the GPU test compares against are in ref/act.h.
//
//   silu_mul_fwd/bwd          act = silu(h1) * h2, the MLP's gate*up (D§4.5)
//   sigmoid_gate_fwd/bwd      out = o * sigmoid(gate), the attention output gate (D§4.7)
//   gdn_gates_fwd/bwd         beta = sigmoid(bb), g = -exp(A_log) * softplus(aa + dt_bias) (D§3.6, D§4.8)
//
// All three are elementwise, one thread per element, grid-stride. n is the number of elements and is an int: the
// largest call is 2516 tokens x 3584 (the packed full batch's MLP width), 9.0M, far below 2^31. gdn_gates indexes its
// per-head tables with i % heads (entry i is token i/heads, head i%heads); heads is 16 in the model.
#ifndef JT_ACT_H
#define JT_ACT_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// act[i] = silu(h1[i]) * h2[i],  i < n.
void k_silu_mul_fwd(const float *h1, const float *h2, float *act, int n, cudaStream_t s);
// dh1[i] = dact[i] * h2[i] * dsilu(h1[i]);  dh2[i] = dact[i] * silu(h1[i]). dh1 and dh2 may not alias dact.
void k_silu_mul_bwd(const float *dact, const float *h1, const float *h2, float *dh1, float *dh2, int n, cudaStream_t s);

// out[i] = o[i] * sigmoid(gate[i]),  i < n.
void k_sigmoid_gate_fwd(const float *o, const float *gate, float *out, int n, cudaStream_t s);
// d_o[i] = dout[i] * sigmoid(gate[i]);  dgate[i] = dout[i] * o[i] * sigmoid(gate[i]) * (1 - sigmoid(gate[i])).
void k_sigmoid_gate_bwd(const float *dout, const float *o, const float *gate, float *d_o, float *dgate, int n,
                        cudaStream_t s);

// beta[i] = sigmoid(bb[i]);  g[i] = -exp(A_log[i % heads]) * softplus(aa[i] + dt_bias[i % heads]).
// beta and g are written; A_log and dt_bias are [heads] (the checkpoint's bf16 values, converted to fp32 at load).
void k_gdn_gates_fwd(const float *bb, const float *aa, const float *A_log, const float *dt_bias, float *beta, float *g,
                     int n, int heads, cudaStream_t s);
// dbb[i] = dbeta[i] * beta[i] * (1 - beta[i]);  daa[i] = dg[i] * (-exp(A_log[i % heads])) * dsigmoid(aa[i] + dt_bias).
// beta is the forward's saved value, as in torch's sigmoid backward.
void k_gdn_gates_bwd(const float *dbeta, const float *dg, const float *beta, const float *aa, const float *A_log,
                     const float *dt_bias, float *dbb, float *daa, int n, int heads, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_ACT_H
