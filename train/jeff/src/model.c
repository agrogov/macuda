// src/model.c — the model's weights, the load (plan M4.2) and the layer driver (design §3.4, §6.2), plus the context and
// activation arena every layer call takes (src/layer.h).
//
// The load is one walk of src/model_table.h's table with the tensor's dtype and every dimension handed to st_getv, which
// is the only way a shape gets into this program: a checkpoint whose layout differs fails at load, naming the tensor and
// both shapes (src/st.h). Nothing here invents a name or a number — the table has them, and the numbers it has are the
// config's (design §3.1).
//
// Two things the design singles out (design §6.1, plan M4.2):
//   * every frozen text weight is widened from the checkpoint's bf16 to fp32 once, into one arena, on the device; the
//     embedding table is the exception and stays bf16, because only the rows a batch uses are ever wanted
//     (kernels/misc.h's jt_embed_gather widens those);
//   * A_log and dt_bias are converted on the host, and A_log is precomputed as negA = -exp(A_log) — the gates work in
//     fp32 (design §3.6) and the exp is done once.
//
// The layer driver is design §3.4 verbatim, split at the post-mixer residual that dump_ref.py's keys and test_layer use
// (design §10.2): the mixer writes x_out = x_in + Mixer(RMSNorm(x_in)), the MLP adds its own path on top. Neither piece
// keeps anything during the forward sweep; the backward recomputes each layer from its saved input (design §6.2).
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "model.h"
#include "layer.h"
#include "st.h"
#include "util.h"

#include "gdn.h"  // kernels/gdn.h: the checkpoint and backward-scratch size helpers
#include "misc.h" // kernels/misc.h: jt_bf16_to_f32_n, jt_embed_gather
#include "rope.h" // kernels/rope.h: rope_tables, JT_ROPE_DIM

// The Linear.stem strings: the 150 adapted weights, the 36 frozen GDN in_proj_a/in_proj_b and the readout need less than
// 12 KB; the pool is checked as it fills rather than trusted.
#define MODEL_STEM_POOL ((size_t)16384)

static void die(const char *fmt, ...) {
  va_list ap;
  fflush(stdout);
  fputs("model: ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

// bf16 -> fp32 is `bits << 16`, nothing else (design §8.1). The host-side twin of kernels/misc.h's jt_bf16_to_f32_n.
static float bf16_to_f32(unsigned short bits) {
  const unsigned int u = (unsigned int)bits << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

static const char *path_join(char *buf, size_t cap, const char *dir, const char *name) {
  const size_t dl = strlen(dir);
  const char *slash = (dl > 0 && dir[dl - 1] == '/') ? "" : "/";
  if (snprintf(buf, cap, "%s%s%s", dir, slash, name) >= (int)cap) die("path too long: %s/%s", dir, name);
  return buf;
}

// `<dir>/readout.safetensors` for a checkpoint directory, `<model.safetensors' directory>/readout.safetensors` for a file.
static const char *readout_path(char *buf, size_t cap, const char *path) {
  struct stat sb;
  if (stat(path, &sb) != 0) die("%s: %s", path, strerror(errno));
  if (!S_ISDIR(sb.st_mode)) {
    char dir[4096];
    if (strlen(path) >= sizeof dir) die("path too long: %s", path);
    strcpy(dir, path);
    char *slash = strrchr(dir, '/');
    if (slash == NULL) {
      strcpy(dir, ".");
    } else if (slash == dir) {
      slash[1] = '\0';
    } else {
      *slash = '\0';
    }
    return path_join(buf, cap, dir, mt_readout_file);
  }
  return path_join(buf, cap, path, mt_readout_file);
}

// ---- the load's two passes ---------------------------------------------------------------------------------------------
// Pass 0 checks every tensor and adds up the arena, pass 1 widens each one into it. Both walk the same code, so the sizes
// cannot disagree with the fills (the end of model_load asserts it anyway). The bf16 staging buffer is the largest single
// tensor: widening straight from host mmap to device fp32 is what jt_bf16_to_f32_n is for, and it needs device bf16 input.
typedef struct {
  Model *m;              // m->w is NULL on pass 0
  const Lora *lora;      // for the A/B offsets of the adapted linears
  St *st;
  int pass;              // 0 = size and check, 1 = widen
  size_t at;             // floats placed so far
  size_t ntensor;        // tensors checked with st_get
  size_t max_el;         // the largest tensor's element count (sizes the staging buffer)
  size_t embed_el;       // the embedding table's element count
  unsigned short *stage; // device bf16 staging (pass 1 only)
  size_t stage_el;
  char *pool;            // Linear.stem strings, filled in pass 1
  size_t pool_cap, pool_at;
} Loader;

// The adapter key of a checkpoint weight: its name with ".weight" removed (design §9.2, the same rule src/lora.c uses).
static const char *stem(Loader *L, const char *name) {
  const size_t n = strlen(name);
  if (n <= 7 || strcmp(name + n - 7, ".weight") != 0) die("model table: '%s' is not a .weight tensor", name);
  const size_t len = n - 7;
  if (L->pool_at + len + 1 > L->pool_cap) die("the stem pool (%zu bytes) is too small for '%s'", L->pool_cap, name);
  char *dst = L->pool + L->pool_at;
  memcpy(dst, name, len);
  dst[len] = '\0';
  L->pool_at += len + 1;
  return dst;
}

static const LoraEntry *find_lora(const Lora *l, const char *stem_name) {
  for (int k = 0; k < MT_LORA_PAIRS; k++)
    if (strcmp(l->ent[k].stem, stem_name) == 0) return &l->ent[k];
  return NULL;
}

static float *reserve(Loader *L, size_t n) {
  float *p = L->pass ? L->m->w + L->at : NULL;
  L->at += n;
  return p;
}

// One bf16 checkpoint tensor: dtype and every dimension checked by st_getv, then (pass 1) widened into the arena.
static float *put(Loader *L, const char *name, int ndim, const int *shape) {
  const void *src = st_getv(L->st, name, ST_DTYPE_BF16, ndim, shape);
  size_t n = 1;
  for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
  if (n > L->max_el) L->max_el = n;
  float *dst = reserve(L, n);
  if (L->pass) {
    if (n > L->stage_el) die("'%s' has %zu elements, more than the %zu-element staging buffer", name, n, L->stage_el);
    CK(cudaMemcpy(L->stage, src, n * sizeof(unsigned short), cudaMemcpyHostToDevice));
    jt_bf16_to_f32_n(L->stage, dst, n, 0);
  }
  if (L->pass) L->ntensor++;
  return dst;
}

// A_log [16] and dt_bias [16] as fp32, plus negA = -exp(A_log) (plan M4.2). The exp is the host's, in double, once.
static void put_gates(Loader *L, const MtLayer *ml, LayerWeights *lw) {
  const int shape[1] = {MT_GDN_HEADS};
  const void *al = st_getv(L->st, ml->a_log, ST_DTYPE_BF16, 1, shape);
  const void *dt = st_getv(L->st, ml->dt_bias, ST_DTYPE_BF16, 1, shape);
  float *dst = reserve(L, 3 * (size_t)MT_GDN_HEADS); // a_log | neg_a | dt_bias
  if (L->pass) {
    float a[MT_GDN_HEADS], neg[MT_GDN_HEADS], d[MT_GDN_HEADS];
    for (int i = 0; i < MT_GDN_HEADS; i++) {
      a[i] = bf16_to_f32(((const unsigned short *)al)[i]);
      neg[i] = (float)-exp((double)a[i]);
      d[i] = bf16_to_f32(((const unsigned short *)dt)[i]);
    }
    CK(cudaMemcpy(dst, a, sizeof a, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dst + MT_GDN_HEADS, neg, sizeof neg, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dst + 2 * MT_GDN_HEADS, d, sizeof d, cudaMemcpyHostToDevice));
    lw->a_log = dst;
    lw->neg_a = dst + MT_GDN_HEADS;
    lw->dt_bias = dst + 2 * MT_GDN_HEADS;
  }
  if (L->pass) L->ntensor += 2;
}

// One [out,in] checkpoint matrix and the Linear that reads it. The structural fields (in/out/lora/W/stem, and the A/B
// offsets found by adapter key in src/lora.c's table) are pass 1's; pass 0 only places and counts the tensor.
static void put_linear(Loader *L, const MtLinear *t, Linear *out) {
  const int shape[2] = {t->out, t->in};
  float *W = put(L, t->name, 2, shape);
  if (!L->pass) return;
  out->W = W;
  out->in = t->in;
  out->out = t->out;
  out->lora = (t->lora == MT_LORA);
  out->a_off = 0;
  out->b_off = 0;
  out->stem = stem(L, t->name);
  if (out->lora) {
    const LoraEntry *e = find_lora(L->lora, out->stem);
    if (e == NULL) die("no LoRA entry for '%s' in the flat parameter buffer", out->stem);
    if (e->in != t->in || e->out != t->out) {
      die("'%s': the flat buffer has it [%d,%d], the model table [%d,%d]", out->stem, e->out, e->in, t->out, t->in);
    }
    out->a_off = e->off;
    out->b_off = e->b_off;
  }
}

static void walk_layer(Loader *L, const MtLayer *ml, LayerWeights *lw) {
  const int hidden[1] = {MT_HIDDEN};
  lw->input_norm = put(L, ml->input_norm, 1, hidden);
  lw->post_attn_norm = put(L, ml->post_attention_norm, 1, hidden);
  put_linear(L, &ml->gate, &lw->gate);
  put_linear(L, &ml->up, &lw->up);
  put_linear(L, &ml->down, &lw->down);
  if (ml->full_attention) {
    const int hd[1] = {MT_HEAD_DIM};
    put_linear(L, &ml->q_proj, &lw->q_proj);
    put_linear(L, &ml->k_proj, &lw->k_proj);
    put_linear(L, &ml->v_proj, &lw->v_proj);
    put_linear(L, &ml->o_proj, &lw->o_proj);
    lw->q_norm = put(L, ml->q_norm, 1, hd);
    lw->k_norm = put(L, ml->k_norm, 1, hd);
  } else {
    const int conv[3] = {MT_GDN_QKV_OUT, 1, MT_GDN_CONV_K};
    const int gd[1] = {MT_GDN_HEAD_DIM};
    // in_proj_a / in_proj_b are [16,1024] and frozen (design §2.1, §3.2). The model table keeps only their names, so
    // their shape is written here from the config's constants and st_get checks it against the checkpoint.
    const MtLinear ia = {.name = ml->in_proj_a, .in = MT_HIDDEN, .out = MT_GDN_HEADS, .lora = MT_FROZEN};
    const MtLinear ib = {.name = ml->in_proj_b, .in = MT_HIDDEN, .out = MT_GDN_HEADS, .lora = MT_FROZEN};
    put_linear(L, &ml->in_proj_qkv, &lw->in_proj_qkv);
    put_linear(L, &ml->in_proj_z, &lw->in_proj_z);
    put_linear(L, &ia, &lw->in_proj_a);
    put_linear(L, &ib, &lw->in_proj_b);
    put_linear(L, &ml->out_proj, &lw->out_proj);
    lw->conv1d = put(L, ml->conv1d, 3, conv); // [6144,1,4] flattening to [6144][4] is just its storage order
    put_gates(L, ml, lw);
    lw->gdn_norm = put(L, ml->gdn_norm, 1, gd); // NOT zero-centred (design §3.3)
  }
}

static void model_walk(Loader *L) {
  Model *m = L->m;
  const int eshape[2] = {MT_VOCAB, MT_HIDDEN};
  const int fshape[1] = {MT_HIDDEN};
  const void *esrc = st_getv(L->st, mt_embed_name, ST_DTYPE_BF16, 2, eshape); // stays bf16 (design §6.1)
  L->embed_el = (size_t)MT_VOCAB * (size_t)MT_HIDDEN;
  if (L->pass) {
    CK(cudaMemcpy(m->embed, esrc, L->embed_el * sizeof(unsigned short), cudaMemcpyHostToDevice));
    m->embed_el = L->embed_el;
    L->ntensor++; // a tensor checked by st_get like the rest, just not widened
  }
  m->final_norm = put(L, mt_final_norm_name, 1, fshape);
  for (int i = 0; i < MT_LAYERS; i++) walk_layer(L, &mt_layers[i], &m->layers[i]);
}

// The readout is not in model.safetensors: it is a separate BF16 file next to the checkpoint (design §2.1, §3.2) and it
// belongs to the flat parameter buffer as fp32 (plan M4.2). A later lora_load_adapter() overwrites it, which is what
// train.py's --init-adapter does.
static void load_readout(Model *m, const char *path, unsigned short *stage, size_t stage_el) {
  char buf[4096];
  const char *file = readout_path(buf, sizeof buf, path);
  const int shape[2] = {MT_READOUT_OUT, MT_READOUT_IN};
  St *st = st_open(file);
  const void *src = st_getv(st, mt_readout_name, ST_DTYPE_BF16, 2, shape);
  const size_t n = (size_t)MT_READOUT_OUT * (size_t)MT_READOUT_IN;
  if (n > stage_el) die("the readout has %zu elements, more than the %zu-element staging buffer", n, stage_el);
  CK(cudaMemcpy(stage, src, n * sizeof(unsigned short), cudaMemcpyHostToDevice));
  jt_bf16_to_f32_n(stage, m->lora->param + m->lora->readout_off, n, 0);
  st_close(st);
}

Model *model_load(const char *path, Lora *lora) {
  if (path == NULL || path[0] == '\0') die("model_load: no checkpoint path");
  if (lora == NULL) die("model_load: the readout lives in the flat parameter buffer, so a Lora is required");
  Model *m = (Model *)calloc(1, sizeof *m);
  if (m == NULL) die("out of memory (Model)");
  m->lora = lora;
  for (int i = 0; i < MT_LAYERS; i++) {
    if (mt_layers[i].full_attention)
      m->n_full++;
    else
      m->n_gdn++;
  }

  Loader L;
  memset(&L, 0, sizeof L);
  L.m = m;
  L.lora = lora;
  L.st = st_open(path);
  L.pool_cap = MODEL_STEM_POOL;
  L.pool = (char *)malloc(L.pool_cap);
  if (L.pool == NULL) die("out of memory (stem pool)");

  model_walk(&L); // pass 0: every shape checked, the arena sized
  if (L.at == 0 || L.max_el == 0 || L.embed_el == 0) die("nothing to load in %s", path);
  const size_t w_floats = L.at;
  CK(cudaMalloc((void **)&m->w, w_floats * sizeof(float)));
  CK(cudaMalloc((void **)&m->embed, L.embed_el * sizeof(unsigned short)));
  unsigned short *stage = NULL;
  CK(cudaMalloc((void **)&stage, L.max_el * sizeof(unsigned short)));

  L.pass = 1;
  L.at = 0;
  L.stage = stage;
  L.stage_el = L.max_el;
  model_walk(&L); // pass 1: widen every tensor into the arena
  if (L.at != w_floats) die("the sizing pass and the load pass disagree (%zu vs %zu floats)", w_floats, L.at);
  m->w_floats = w_floats;
  load_readout(m, path, stage, L.max_el);
  const size_t ntensor = L.ntensor + 1; // + the readout, which is st_get-checked in its own file

  CK(cudaFree(stage));
  st_close(L.st);
  CK(cudaDeviceSynchronize()); // the load ran on the null stream; the layers run on ctx->stream (a non-blocking one)
  free(L.pool);

  printf("model: %d layers (%d gdn, %d attn), weights %.2f GB fp32 + embed %.2f GB bf16\n", MT_LAYERS, m->n_gdn, m->n_full,
         (double)w_floats * sizeof(float) / 1e9, (double)m->embed_el * sizeof(unsigned short) / 1e9);
  printf("model: %zu frozen tensors checked by st_get (dtype and every dimension), readout %.2f MB fp32 in the flat buffer\n",
         ntensor, (double)((size_t)MT_READOUT_OUT * MT_READOUT_IN * sizeof(float)) / 1e6);
  fflush(stdout);
  return m;
}

void model_free(Model *m) {
  if (m == NULL) return;
  if (m->w != NULL) CK(cudaFree(m->w));
  if (m->embed != NULL) CK(cudaFree(m->embed));
  free(m);
}

// ---- a batch's row geometry (design §7) -------------------------------------------------------------------------------
void jt_batch_set(JtBatch *b, int rows, int n, const int *off) {
  if (b == NULL || off == NULL) die("jt_batch_set: no batch or no row table");
  if (rows < 1 || rows > JT_MAX_ROWS) die("jt_batch_set: %d rows, want 1..%d", rows, JT_MAX_ROWS);
  if (n < 1) die("jt_batch_set: %d tokens", n);
  memset(b, 0, sizeof *b);
  b->rows = rows;
  b->n = n;
  for (int i = 0; i <= rows; i++) {
    if (off[i] < 0 || off[i] > n) die("jt_batch_set: off[%d] = %d, outside 0..%d", i, off[i], n);
    if (i > 0 && off[i] < off[i - 1]) die("jt_batch_set: off is not a prefix sum (off[%d] = %d < off[%d] = %d)", i, off[i],
                                          i - 1, off[i - 1]);
    b->off[i] = off[i];
  }
  if (b->off[0] != 0) die("jt_batch_set: off[0] = %d, want 0", b->off[0]);
  if (b->off[rows] != n) die("jt_batch_set: off[%d] = %d, want n = %d", rows, b->off[rows], n);

  int *row_of = (int *)malloc((size_t)n * sizeof *row_of);
  if (row_of == NULL) die("out of memory (row_of, %d tokens)", n);
  for (int r = 0; r < rows; r++)
    for (int t = b->off[r]; t < b->off[r + 1]; t++) row_of[t] = r;
  CK(cudaMalloc((void **)&b->off_dev, (size_t)(rows + 1) * sizeof(int)));
  CK(cudaMemcpy(b->off_dev, b->off, (size_t)(rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMalloc((void **)&b->row_of, (size_t)n * sizeof(int)));
  CK(cudaMemcpy(b->row_of, row_of, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
  free(row_of);
}

void jt_batch_free(JtBatch *b) {
  if (b == NULL) return;
  if (b->off_dev != NULL) CK(cudaFree(b->off_dev));
  if (b->row_of != NULL) CK(cudaFree(b->row_of));
  b->off_dev = NULL;
  b->row_of = NULL;
}

// ---- the context and the activation arena (design §6.2) ---------------------------------------------------------------
JtCtx *jt_ctx_alloc(Model *m, Lora *l, int cap_tokens, int cap_rows, int lmax) {
  if (m == NULL || l == NULL) die("jt_ctx_alloc: the model and the flat parameter buffer are both required");
  if (cap_tokens < 1) die("jt_ctx_alloc: cap_tokens %d", cap_tokens);
  if (cap_rows < 1 || cap_rows > JT_MAX_ROWS) die("jt_ctx_alloc: cap_rows %d, want 1..%d", cap_rows, JT_MAX_ROWS);
  if (lmax < 1) die("jt_ctx_alloc: lmax %d", lmax);
  JtCtx *c = (JtCtx *)calloc(1, sizeof *c);
  if (c == NULL) die("out of memory (JtCtx)");
  c->model = m;
  c->lora = l;
  c->cap_tokens = cap_tokens;
  c->cap_rows = cap_rows;
  c->lmax = lmax;
  CB(cublasCreate(&c->cublas));
  CK(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking)); // design §12: no cudaStreamCreate
  CB(cublasSetStream(c->cublas, c->stream));

  const size_t rope_n = (size_t)lmax * (size_t)JT_ROPE_DIM;
  float *hc = (float *)malloc(rope_n * sizeof(float));
  float *hs = (float *)malloc(rope_n * sizeof(float));
  if (hc == NULL || hs == NULL) die("out of memory (rope tables, %zu rows)", (size_t)lmax);
  rope_tables(hc, hs, lmax); // design §3.3, built once and passed as pointers (the shim has no cudaMemcpyToSymbol)
  CK(cudaMalloc((void **)&c->rope_cos, rope_n * sizeof(float)));
  CK(cudaMalloc((void **)&c->rope_sin, rope_n * sizeof(float)));
  CK(cudaMemcpy(c->rope_cos, hc, rope_n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(c->rope_sin, hs, rope_n * sizeof(float), cudaMemcpyHostToDevice));
  free(hc);
  free(hs);

  c->arena_n = (size_t)cap_tokens * JT_ARENA_FLOATS_PER_TOKEN +
               2 * (size_t)MT_FULL_HEADS * (size_t)lmax * (size_t)lmax; // one row's P and dP (design §6.2)
  CK(cudaMalloc((void **)&c->arena, c->arena_n * sizeof(float)));
  c->save_at = 0;
  c->work_at = c->arena_n;

  c->gdn_nseg = (lmax + JT_GDN_SEG - 1) / JT_GDN_SEG;
  if (c->gdn_nseg < 1) c->gdn_nseg = 1;
  const size_t ckpt = gdn_ckpt_floats(cap_rows, MT_GDN_HEADS, c->gdn_nseg);
  const size_t sp = gdn_bwd_sp_floats(cap_rows, MT_GDN_HEADS);
  const size_t part = gdn_bwd_part_floats((size_t)cap_tokens, MT_GDN_HEADS);
  const size_t gate = gdn_bwd_gate_floats((size_t)cap_tokens, MT_GDN_HEADS);
  CK(cudaMalloc((void **)&c->gdn_ckpt, ckpt * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.sp, sp * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dq_part, part * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dk_part, part * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dbeta_part, gate * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dg_part, gate * sizeof(float)));

  printf("ctx: lmax %d, arena %.2f GB fp32 (%d tokens x %zu floats + 2 x %d x %d^2), gdn nseg %d, ckpt %.2f GB, bwd scratch %.2f GB\n",
         lmax, (double)c->arena_n * sizeof(float) / 1e9, cap_tokens, JT_ARENA_FLOATS_PER_TOKEN, MT_FULL_HEADS, lmax,
         c->gdn_nseg, (double)ckpt * sizeof(float) / 1e9, (double)(sp + 2 * part + 2 * gate) * sizeof(float) / 1e9);
  fflush(stdout);
  // One device-wide barrier, once, after the load: everything before this point uploaded with plain cudaMemcpy on the
  // null stream (model_load, lora_load_adapter, jt_batch_set), and ctx->stream is a *non-blocking* one (design §12), which
  // the driver does not order against the legacy stream. Without the barrier the first kernel on ctx->stream can read a
  // weight before its upload has landed — seen on gpubox as a weight arena of zeros.
  CK(cudaDeviceSynchronize());
  return c;
}

void jt_ctx_free(JtCtx *c) {
  if (c == NULL) return;
  if (c->gdn_work.dg_part != NULL) CK(cudaFree(c->gdn_work.dg_part));
  if (c->gdn_work.dbeta_part != NULL) CK(cudaFree(c->gdn_work.dbeta_part));
  if (c->gdn_work.dk_part != NULL) CK(cudaFree(c->gdn_work.dk_part));
  if (c->gdn_work.dq_part != NULL) CK(cudaFree(c->gdn_work.dq_part));
  if (c->gdn_work.sp != NULL) CK(cudaFree(c->gdn_work.sp));
  if (c->gdn_ckpt != NULL) CK(cudaFree(c->gdn_ckpt));
  if (c->arena != NULL) CK(cudaFree(c->arena));
  if (c->rope_cos != NULL) CK(cudaFree(c->rope_cos));
  if (c->rope_sin != NULL) CK(cudaFree(c->rope_sin));
  if (c->stream != NULL) CK(cudaStreamDestroy(c->stream));
  if (c->cublas != NULL) CB(cublasDestroy(c->cublas));
  free(c);
}

// ---- the layer driver (design §3.4, §6.2) -----------------------------------------------------------------------------
int layer_fwd(JtCtx *c, int i, const float *x_in, int n, const JtBatch *b, float *x_out, float *post_mixer, int mode) {
  const MtLayer *ml = mt_layer(i);
  if (c == NULL || ml == NULL || x_in == NULL || x_out == NULL || b == NULL) die("layer_fwd: bad arguments (layer %d)", i);
  if (n != b->n) die("layer_fwd: n %d but the batch has %d tokens", n, b->n);
  if (mode != JT_FWD_ONLY && mode != JT_FWD_SAVE) die("layer_fwd: mode %d, want JT_FWD_ONLY or JT_FWD_SAVE", mode);
  if (mode == JT_FWD_SAVE) {
    if (post_mixer == NULL) die("layer_fwd: a JT_FWD_SAVE pass must be given post_mixer (layer_bwd takes it)");
    if (post_mixer == x_out) die("layer_fwd: post_mixer and x_out may not be the same buffer");
    jt_saves_reset(c); // this layer's saves start here; the previous layer's are gone (design §6.2)
  }

  const LayerWeights *lw = &c->model->layers[i];
  if (x_out != x_in) CK(cudaMemcpyAsync(x_out, x_in, (size_t)n * MT_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
  int rc = ml->full_attention ? layer_attn_mixer_fwd(c, lw, x_in, n, b, x_out, mode)
                              : layer_gdn_mixer_fwd(c, lw, x_in, n, b, x_out, mode);
  if (rc != JT_LAYER_OK) return rc;
  if (post_mixer != NULL) { // the residual right after the mixer, i.e. the MLP's input (dump_ref's post_mixer key)
    CK(cudaMemcpyAsync(post_mixer, x_out, (size_t)n * MT_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
  }
  return layer_mlp_fwd(c, lw, post_mixer != NULL ? post_mixer : x_out, n, b, x_out, mode);
}

int layer_bwd(JtCtx *c, int i, const float *x_in, const float *post_mixer, float *dres, int n, const JtBatch *b) {
  const MtLayer *ml = mt_layer(i);
  if (c == NULL || ml == NULL || x_in == NULL || post_mixer == NULL || dres == NULL || b == NULL)
    die("layer_bwd: bad arguments (layer %d)", i);
  if (n != b->n) die("layer_bwd: n %d but the batch has %d tokens", n, b->n);
  const LayerWeights *lw = &c->model->layers[i];
  int rc = layer_mlp_bwd(c, lw, post_mixer, dres, n, b); // the MLP is the last thing the forward did
  if (rc != JT_LAYER_OK) return rc;
  return ml->full_attention ? layer_attn_mixer_bwd(c, lw, x_in, dres, n, b) : layer_gdn_mixer_bwd(c, lw, x_in, dres, n, b);
}

int model_forward(JtCtx *c, const int *ids, int n, const JtBatch *b, float *saved_in, float *post_mixer) {
  if (c == NULL || ids == NULL || b == NULL || saved_in == NULL) die("model_forward: bad arguments");
  if (n != b->n) die("model_forward: n %d but the batch has %d tokens", n, b->n);
  const size_t stride = (size_t)n * MT_HIDDEN;
  jt_embed_gather(ids, c->model->embed, saved_in, n, MT_HIDDEN, c->stream); // design §8.1: only the used rows widen
  for (int i = 0; i < MT_LAYERS; i++) {
    const float *in = saved_in + (size_t)i * stride;
    float *out = saved_in + (size_t)(i + 1) * stride;
    int rc = layer_fwd(c, i, in, n, b, out, post_mixer != NULL ? post_mixer + (size_t)i * stride : NULL, JT_FWD_ONLY);
    if (rc != JT_LAYER_OK) return rc;
  }
  return JT_LAYER_OK;
}
