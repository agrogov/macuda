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
// keeps anything during the forward sweep; the backward recomputes each layer from its saved input (design §6.2, the
// layer-level recompute — model_backward below).
//
// Plan M8 item 4 adds the other side of that trade: with --no-recompute (JT_NO_RECOMPUTE=1) the forward sweep keeps every
// layer's saves in a "store" appended to the arena, so the backward reads them where the forward left them and re-runs
// nothing — one forward less per step, at 3.2 MB of device memory per packed token. The section before jt_ctx_alloc has
// the layout and the reasons. Since task J51 the mode needs no flag when the device has room for it: it is the default
// when the store (plus a 1 GB margin) fits, because J39 measured its trace byte-identical to the recompute path's at -31%
// s/step; JT_NO_RECOMPUTE=0 is how a run asks for the recompute path back.
//
// Plan M8.1b (JT_BF16_WEIGHTS=1) is the third memory story here: the frozen Linear weights stay bf16 in a second arena
// instead of being widened into the fp32 one, which halves them (1.99 -> 1.00 GB) and gives their GEMMs bf16 operands.
// It changes no layer, no kernel and no arithmetic outside those two GEMMs; model_load's walk places each tensor in one
// arena or the other, and exact mode refuses the toggle so M6.2's exact trace cannot move.
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

#include "gemm.h" // src/gemm.h: gemm_is_exact, so exact mode can ignore M8.1b's bf16 weights everywhere

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
  int bf16;              // 1 = the frozen Linear weights stay bf16, in m->wb (plan M8.1b)
  size_t at;             // floats placed so far
  size_t atb;            // bf16 elements placed so far
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

// One bf16 checkpoint tensor kept as bf16 (plan M8.1b): the checkpoint's own bits, copied straight from the host mapping
// into the bf16 arena, so this path needs neither the staging buffer nor the widening kernel — and loses nothing, because
// every frozen tensor in the checkpoint is already BF16 (the load's listing is in notes/J47.md). Slots are rounded up to
// 8 elements (16 bytes) so that design §6.1's alignment rule holds for every operand base, not just by luck of the
// tensor sizes; the arena itself comes from cudaMalloc and is 256-byte aligned.
static unsigned short *reserve_bf16(Loader *L, size_t n) {
  L->atb = (L->atb + 7u) & ~(size_t)7u;
  unsigned short *p = L->pass ? L->m->wb + L->atb : NULL;
  L->atb += n;
  return p;
}

static unsigned short *put_bf16(Loader *L, const char *name, int ndim, const int *shape) {
  const void *src = st_getv(L->st, name, ST_DTYPE_BF16, ndim, shape);
  size_t n = 1;
  for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
  unsigned short *dst = reserve_bf16(L, n);
  if (L->pass) {
    CK(cudaMemcpy(dst, src, n * sizeof(unsigned short), cudaMemcpyHostToDevice));
    L->ntensor++;
  }
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

// One [out,in] checkpoint matrix and the Linear that reads it. The structural fields (in/out/lora/W/Wb/stem, and the A/B
// offsets found by adapter key in src/lora.c's table) are pass 1's; pass 0 only places and counts the tensor. With
// plan M8.1b's bf16 storage the matrix goes to the bf16 arena and `W` gets the shared marker pointer (src/linear.h):
// eight zeroed words, not this linear's weight, so a "is it loaded?" check that only tests W for NULL still passes.
static void put_linear(Loader *L, const MtLinear *t, Linear *out) {
  const int shape[2] = {t->out, t->in};
  float *W = L->bf16 ? NULL : put(L, t->name, 2, shape);
  unsigned short *Wb = L->bf16 ? put_bf16(L, t->name, 2, shape) : NULL;
  if (!L->pass) return;
  out->W = L->bf16 ? L->m->w_mark : W;
  out->Wb = Wb;
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

// Plan M8.1b's switch, in the environment as JT_EXACT and JT_NO_RECOMPUTE are: JT_BF16_WEIGHTS unset (or "0") is the
// fp32 storage, byte for byte what this file did before; any other value keeps the frozen Linear weights bf16 and runs
// their GEMMs with bf16 operands (src/linear.c's weight_gemm).
//
// Exact mode ignores it, and says so: every activation and gradient gate (design §10.2, plan M6.2) is measured in exact
// mode, and the reason that mode is bit-reproducible is that it never changes — a bf16 weight GEMM would round the
// weights' 8-bit mantissas against nothing and move M6.2's trace. So JT_EXACT=1 JT_BF16_WEIGHTS=1 is the fp32 storage
// plus one line saying the toggle was seen and dropped, which is what `ab.sh --same --env-b JT_BF16_WEIGHTS=1` reads.
static int bf16_weights_toggle(void) {
  const char *s = getenv("JT_BF16_WEIGHTS");
  if (s == NULL || s[0] == '\0' || strcmp(s, "0") == 0) return 0;
  if (gemm_is_exact()) {
    printf("weights: bf16 ignored (JT_EXACT=1: exact mode keeps the fp32 path byte for byte)\n");
    fflush(stdout);
    return 0;
  }
  return 1;
}

Model *model_load(const char *path, Lora *lora) {
  if (path == NULL || path[0] == '\0') die("model_load: no checkpoint path");
  if (lora == NULL) die("model_load: the readout lives in the flat parameter buffer, so a Lora is required");
  const int bf16 = bf16_weights_toggle(); // before the walk: it decides which arena a Linear's weight goes to
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
  L.bf16 = bf16;
  L.pool_cap = MODEL_STEM_POOL;
  L.pool = (char *)malloc(L.pool_cap);
  if (L.pool == NULL) die("out of memory (stem pool)");

  model_walk(&L); // pass 0: every shape checked, the arena sized
  if (L.at == 0 || L.max_el == 0 || L.embed_el == 0) die("nothing to load in %s", path);
  if (bf16 && L.atb == 0) die("JT_BF16_WEIGHTS: the bf16 arena would be empty, the model table has no linear");
  const size_t w_floats = L.at;
  const size_t wb_el = L.atb;
  // The staging buffer holds the largest tensor this load copies through it: a model tensor, or the readout, which lives
  // in its own file and is checked after the walk (with bf16 weights no model tensor is that big any more — the frozen
  // linears no longer go through `put` — so the readout's size is the one that decides).
  const size_t readout_el = (size_t)MT_READOUT_OUT * (size_t)MT_READOUT_IN;
  const size_t stage_el = L.max_el > readout_el ? L.max_el : readout_el;
  CK(cudaMalloc((void **)&m->w, (w_floats + (bf16 ? 8u : 0u)) * sizeof(float)));
  if (bf16) {
    float *mark = m->w + w_floats; // + 8 words above the arena: the bf16 linears' presence marker (src/linear.h)
    CK(cudaMemset(mark, 0, 8 * sizeof(float)));
    m->w_mark = mark;
  }
  if (bf16) CK(cudaMalloc((void **)&m->wb, wb_el * sizeof(unsigned short)));
  CK(cudaMalloc((void **)&m->embed, L.embed_el * sizeof(unsigned short)));
  unsigned short *stage = NULL;
  CK(cudaMalloc((void **)&stage, stage_el * sizeof(unsigned short)));

  L.pass = 1;
  L.at = 0;
  L.atb = 0;
  L.stage = stage;
  L.stage_el = stage_el;
  model_walk(&L); // pass 1: widen (or copy bf16) every tensor into its arena
  if (L.at != w_floats) die("the sizing pass and the load pass disagree (%zu vs %zu floats)", w_floats, L.at);
  if (L.atb != wb_el) die("the sizing pass and the load pass disagree (%zu vs %zu bf16 elements)", wb_el, L.atb);
  m->w_floats = w_floats;
  m->wb_el = wb_el;
  load_readout(m, path, stage, stage_el);
  const size_t ntensor = L.ntensor + 1; // + the readout, which is st_get-checked in its own file

  CK(cudaFree(stage));
  st_close(L.st);
  CK(cudaDeviceSynchronize()); // the load ran on the null stream; the layers run on ctx->stream (a non-blocking one)
  free(L.pool);

  if (bf16) {
    printf("model: %d layers (%d gdn, %d attn), weights %.2f GB bf16 + %.2f GB fp32 + embed %.2f GB bf16\n", MT_LAYERS,
           m->n_gdn, m->n_full, (double)wb_el * sizeof(unsigned short) / 1e9,
           (double)w_floats * sizeof(float) / 1e9, (double)m->embed_el * sizeof(unsigned short) / 1e9);
    printf("weights: bf16 (%.2f GB)\n",
           ((double)wb_el * sizeof(unsigned short) + (double)w_floats * sizeof(float)) / 1e9);
  } else {
    printf("model: %d layers (%d gdn, %d attn), weights %.2f GB fp32 + embed %.2f GB bf16\n", MT_LAYERS, m->n_gdn,
           m->n_full, (double)w_floats * sizeof(float) / 1e9, (double)m->embed_el * sizeof(unsigned short) / 1e9);
  }
  printf("model: %zu frozen tensors checked by st_get (dtype and every dimension), readout %.2f MB fp32 in the flat buffer\n",
         ntensor, (double)((size_t)MT_READOUT_OUT * MT_READOUT_IN * sizeof(float)) / 1e6);
  fflush(stdout);
  return m;
}

void model_free(Model *m) {
  if (m == NULL) return;
  if (m->wb != NULL) CK(cudaFree(m->wb));
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

// ---- --no-recompute: keep every layer's forward saves (plan M8 item 4) -----------------------------------------------
// Design §6.2's layer recompute re-runs each layer's forward in front of its backward: one extra forward per step (about
// +33% time, D§6.2) and only one layer's scratch on the device. `--no-recompute` (JT_NO_RECOMPUTE=1, the environment the
// parity scripts can be re-run with; or, since task J51, automatic when the store fits — store_auto_mode below) trades
// memory for that time: the forward sweep runs every layer in JT_FWD_SAVE mode
// and leaves its saves where its backward reads them, so the backward re-runs nothing: plan M8 budgets −25% time, and the
// measured pair on gpubox's 5090 is 1.127 -> 0.778 s/step (−31%, the recomputed forward's 0.349 s) for 29.56 GB of peak
// against 7.84 (J39).
//
// The saves cannot live in the one arena: 24 layers of them are 0.81 M floats per packed token (below), eight times the
// per-token budget the arena has, so the context allocates a second region, the "store", appended to the arena **in the
// same cudaMalloc**. For a batch of n tokens it is laid out as
//
//     [ pm_0 .. pm_23 ][ saves_0 | saves_1 | ... | saves_23 ][ gdn ckpt_0 .. ckpt_17 ][ one layer's work ]
//       n * HIDDEN each       n * (this layer's width)          cap_rows sized          the arena's own width
//
// (`pm_i` is layer i's post-mixer residual, which its MLP backward takes as its input x; the GDN checkpoint slots exist
// because c->gdn_ckpt is one buffer that every GDN layer's forward would otherwise overwrite.) Two properties keep the
// rest of this file free of new state:
//
//   * a backward's saves always start at the arena's base (the three pieces' convention, src/layer.h), so the layer loop
//     moves c->arena to the layer's slot and c->arena_n to "the store's top minus that slot". Work then comes from the
//     region at the top — jt_work_alloc returns c->arena + work_at = top - n, which is the same address for every layer —
//     and the arena's own guard, work_at - save_at, is exactly the room between this layer's saves and that same top.
//   * the store needs no field and no global to be found: it sits directly above the arena in the same allocation, so its
//     size and its base are pure functions of the capacities and `no_recompute(c)` is the accounting identity
//     arena_n == base_arena_floats(c) + store_floats(c). Nothing outside this file has to know the mode is on.
//
// The work region is shared by all 24 layers because only one layer is ever live, and it is one arena wide — saves plus
// work for the widest layer — so a piece's temporaries can only leave it if the arena itself is too small, which the
// recompute path refuses first.

// One layer's saved words per packed token, from the pieces' own layouts: src/layer_mlp.c's saves_floats() (11,825),
// src/layer_gdn.c's walk() (23,697) and src/layer_attn.c's attn_save_ptrs() (12,875). Written as the same sums of the
// model's widths so they can be checked against those functions line by line, and `store_check` compares the save cursor
// with them after every SAVE pass of every step: a piece that grows a saved buffer stops this mode with both numbers
// instead of letting one layer's slots overlap the next.
#define JT_MLP_SAVES_PER_TOKEN ((size_t)MT_HIDDEN + 1 + 3 * (size_t)MT_MLP_INTER + 3 * (size_t)MT_LORA_RANK)
#define JT_GDN_SAVES_PER_TOKEN                                                                                          \
  ((size_t)MT_HIDDEN + 1 + 3 * (size_t)MT_LORA_RANK + 3 * (size_t)MT_GDN_HEADS + (size_t)MT_GDN_QKV_OUT +              \
   3 * (size_t)MT_GDN_Z_OUT + 2 * (size_t)MT_GDN_Z_OUT + 2 * (size_t)MT_GDN_HEADS + 3 * (size_t)MT_GDN_Z_OUT +         \
   (size_t)MT_GDN_HEADS)
#define JT_ATTN_SAVES_PER_TOKEN                                                                                         \
  (1 + (size_t)MT_HIDDEN + (size_t)MT_Q_OUT + 3 * (size_t)MT_O_IN + (size_t)MT_FULL_HEADS + (size_t)MT_KV_HEADS +      \
   3 * (size_t)MT_KV_OUT + 4 * (size_t)MT_LORA_RANK)
_Static_assert(JT_MLP_SAVES_PER_TOKEN == 11825, "src/layer_mlp.c's saves_floats() no longer agrees with this file");
_Static_assert(JT_GDN_SAVES_PER_TOKEN == 23697, "src/layer_gdn.c's walk() no longer agrees with this file");
_Static_assert(JT_ATTN_SAVES_PER_TOKEN == 12875, "src/layer_attn.c's attn_save_ptrs() no longer agrees with this file");

static size_t mixer_saves_per_token(const MtLayer *ml) {
  return ml->full_attention ? JT_ATTN_SAVES_PER_TOKEN : JT_GDN_SAVES_PER_TOKEN;
}
static size_t layer_saves_per_token(const MtLayer *ml) { return mixer_saves_per_token(ml) + JT_MLP_SAVES_PER_TOKEN; }

// The arena the context has without a store (plan M0.1/M4.3 unchanged): one layer's saves and work for the widest batch,
// plus one row's P and dP (design §6.2).
static size_t base_arena_floats(const JtCtx *c) {
  return (size_t)c->cap_tokens * JT_ARENA_FLOATS_PER_TOKEN +
         2 * (size_t)MT_FULL_HEADS * (size_t)c->lmax * (size_t)c->lmax;
}

// The store's words per packed token, all 24 layers' saves and pm slabs: 0.81 M, i.e. 3.25 MB, which is what the mode
// costs and what the start-up line prints (plan rule 6: memory is the reason this one is opt-in).
static size_t store_floats_per_token(void) {
  size_t n = (size_t)MT_LAYERS * MT_HIDDEN;
  for (int i = 0; i < MT_LAYERS; i++) n += layer_saves_per_token(mt_layer(i));
  return n;
}

// How many packed tokens the store was sized for. The context's cap_tokens is the widest batch of *any* split, and a run's
// widest is usually a dev check's (train.py's dev batches are twice the training batch) — a forward-only pass that has no
// backward to feed. So main.c hands over the train split's max_tokens_per_batch as JT_NO_RECOMPUTE_TOKENS and that is the
// number the store is sized for; a batch bigger than it takes the recompute path in *both* sweeps (model_forward and
// model_backward ask this same question), which is exactly the pre-M8 behaviour and needs no store at all.
static int store_tokens(const JtCtx *c) {
  const char *s = getenv("JT_NO_RECOMPUTE_TOKENS");
  if (s == NULL) return c->cap_tokens;
  char *end = NULL;
  const long v = strtol(s, &end, 10);
  if (end == s || v < 1) return c->cap_tokens;
  return (int)(v < (long)c->cap_tokens ? v : (long)c->cap_tokens);
}

// One GDN checkpoint slot per GDN layer (c->gdn_ckpt is one buffer, and every GDN layer's forward would overwrite it).
static size_t store_ckpt_floats(const JtCtx *c) {
  return (size_t)c->model->n_gdn * gdn_ckpt_floats(c->cap_rows, MT_GDN_HEADS, c->gdn_nseg);
}

// The shared work region: one arena's worth at the store's token count, because a piece's temporaries are bounded by
// `JT_ARENA_FLOATS_PER_TOKEN` per token and by one row's P and dP (design §6.2's per-token budget), and the arena itself
// reserves those bytes below its own saves.
static size_t store_work_floats(const JtCtx *c) {
  const size_t t = (size_t)store_tokens(c);
  return t * JT_ARENA_FLOATS_PER_TOKEN + 2 * (size_t)MT_FULL_HEADS * (size_t)c->lmax * (size_t)c->lmax;
}

// The store's whole size: the pm slabs and save slots for `store_tokens` tokens, the GDN checkpoint slots, the work region.
static size_t store_floats(const JtCtx *c) {
  return store_floats_per_token() * (size_t)store_tokens(c) + store_ckpt_floats(c) + store_work_floats(c);
}

// The mode's switch, in the environment as JT_EXACT and JT_GDN_CHUNKED are: main.c's --no-recompute sets it before the
// context is allocated, and the parity scripts can re-gate the mode without a flag. Three answers, because task J51 makes
// the mode the *default* when the memory is there: 1 = on (any value but "0"), 0 = off (JT_NO_RECOMPUTE=0 forces the
// recompute path), -1 = nobody said, so jt_ctx_alloc's auto rule decides. A toggle has to be able to say "nothing was
// asked" to leave room for a default; "set but empty" keeps its old meaning, off.
static int no_recompute_env(void) {
  const char *s = getenv("JT_NO_RECOMPUTE");
  if (s == NULL) return -1;
  return s[0] != '\0' && strcmp(s, "0") != 0;
}

// The 1 GB margin task J51's auto rule asks for: the room the store may not eat into, because what the run allocates after
// the context (main.c's step buffers) is on the device before the context is asked, but the driver's own context and its
// cubin copies are not something any API here counts.
#define JT_STORE_MARGIN ((size_t)1 << 30)

// Is the store on with nobody having asked for it? (JT_NO_RECOMPUTE unset, the store fits: see store_auto_mode below.)
// Only the start-up line needs to know, so the answer is recomputed from the environment rather than kept in the context.
static int no_recompute_auto(void) { return no_recompute_env() < 0; }

// J51 rule 1: with nothing said in the environment, no-recompute is the default when the device has room for it — J39
// measured the mode's 20-step trace byte-identical to the recompute path's at -31% s/step, so its only cost is memory. The
// rule is "free device memory covers the store plus a 1 GB margin", and the amount the margin is added to is everything
// the run still has on the device once the context is up: the arena with the store on top (one cudaMalloc), the GDN
// checkpoint slots, the backward scratch and main.c's step buffers (which is why main.c allocates those first). Sizing
// the question that way is what makes the answer safe rather than optimistic, and J51 measured why: asking for arena +
// store + margin alone admitted an 11,052-token batch whose real footprint was 32.6 GB of a 31.84 GB card — every
// allocation "fit", peak came out 31.82 GB, and the step time went from 0.475 to **28.84 s/step** as the driver paged.
// Counting the whole footprint refuses that batch by 3.6 GB (30.17 wanted against 26.55 free) and admits the 5,750-token
// shape (23.35 against 27.08), where the mode is 0.345 vs 0.470 s/step and the trace is byte-identical.
//
// Which of the two numbers to trust is a backend question, and both answers were measured:
//   * real CUDA (the nv build): cudaMemGetInfo's free count, because a probe allocation is a lie there — this driver
//     over-commits, and on gpubox the 29.8 GB probe "succeeded" in 28.4 GB of free memory (J51: the run then crawled);
//   * the shim (JT_SHIM): the allocator, because libtinycudart has no allocation accounting ("TODO: subtract live
//     cudaMalloc bytes", cudart_api.c:211) and answers cudaMemGetInfo with the card's total, while tinynv_malloc maps
//     video memory and refuses a request that does not fit.
// main.c allocates the step's buffers before the context for the same reason: they are part of the footprint asked about.
//
// A run with no backward has nothing to feed the store: main.c sets JT_NO_RECOMPUTE_TOKENS only when a backward will read
// it (--eval-only and --dump-forward do not), and without it the recompute path is what those runs get.
static int store_auto_mode(const JtCtx *c, size_t arena_base, size_t rest_work_floats) {
  if (getenv("JT_NO_RECOMPUTE_TOKENS") == NULL) return 0;
  const size_t store_bytes = store_floats(c) * sizeof(float);
  const size_t rest_bytes = rest_work_floats * sizeof(float);
  const size_t need = (arena_base + store_floats(c) + rest_work_floats) * sizeof(float) + JT_STORE_MARGIN;
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));
#ifdef JT_SHIM
  void *probe = NULL;
  const int fits = cudaMalloc(&probe, need) == cudaSuccess;
  if (fits) CK(cudaFree(probe));
  else (void)cudaGetLastError(); // the refused probe is an answer, not an error to carry
#else
  const int fits = free_b >= need;
#endif
  if (fits) {
    printf("ctx: no-recompute (auto): the store (%.2f GB) plus this context's %.2f GB of checkpoint/scratch and a %zu MB "
           "margin fit in %.2f GB free, so every layer's forward saves are kept (JT_NO_RECOMPUTE=0 forces the recompute "
           "path)\n",
           (double)store_bytes / (double)(1u << 30), (double)rest_bytes / (double)(1u << 30),
           (size_t)(JT_STORE_MARGIN >> 20), (double)free_b / (double)(1u << 30));
    fflush(stdout);
    return 1;
  }
  printf("ctx: recompute (auto): the store (%.2f GB) plus this context's %.2f GB of checkpoint/scratch and a %zu MB "
         "margin want %.2f GB and %.2f of %.2f GB is free, so each layer's forward is re-run by its backward "
         "(JT_NO_RECOMPUTE=1 forces the store)\n",
         (double)store_bytes / (double)(1u << 30), (double)rest_bytes / (double)(1u << 30),
         (size_t)(JT_STORE_MARGIN >> 20), (double)need / (double)(1u << 30), (double)free_b / (double)(1u << 30),
         (double)total_b / (double)(1u << 30));
  fflush(stdout);
  return 0;
}

// Is the store there? c->arena_n is base + store when it is (jt_ctx_alloc), so this is the allocation's own bookkeeping and
// not a flag someone has to keep in step. Call it before moving the arena (store_bind changes arena_n).
static int no_recompute(const JtCtx *c) { return c->arena_n > base_arena_floats(c); }

// Does this batch go through the store? Both sweeps ask this one question, so the forward cannot keep saves that the
// backward then recomputes over (or the other way round): a batch above the store's token count — a dev check's — is the
// recompute path in both.
static int store_serves(const JtCtx *c, int n) { return no_recompute(c) && n <= store_tokens(c); }

// A batch's view of the store: its first word and the top every layer's work grows down from. `n` is this batch's token
// count, and the save slots follow it (they are n * a per-token width), so a smaller batch than cap_tokens simply leaves
// the tail of the region unused.
typedef struct {
  float *base; // the store's first word
  float *top;  // one past its last word (the work half's ceiling)
  int n;
} Store;

static size_t store_pm_off(int i, int n) { return (size_t)i * (size_t)n * MT_HIDDEN; }
static size_t store_slot_off(int i, int n) {
  size_t off = (size_t)MT_LAYERS * (size_t)n * MT_HIDDEN;
  for (int j = 0; j < i; j++) off += layer_saves_per_token(mt_layer(j)) * (size_t)n;
  return off;
}
static size_t store_ckpt_off(const JtCtx *c, int gdn_index, int n) {
  size_t off = (size_t)MT_LAYERS * (size_t)n * MT_HIDDEN;
  for (int j = 0; j < MT_LAYERS; j++) off += layer_saves_per_token(mt_layer(j)) * (size_t)n;
  return off + (size_t)gdn_index * gdn_ckpt_floats(c->cap_rows, MT_GDN_HEADS, c->gdn_nseg);
}

// Everything store_bind and the GDN checkpoint swap move, so a sweep can put the context back the way it found it (the head
// uses the arena's base between the sweeps, src/head.c).
typedef struct {
  float *arena, *gdn_ckpt;
  size_t arena_n, save_at, work_at;
} ArenaKeep;

static void arena_keep(const JtCtx *c, ArenaKeep *k) {
  k->arena = c->arena, k->gdn_ckpt = c->gdn_ckpt;
  k->arena_n = c->arena_n, k->save_at = c->save_at, k->work_at = c->work_at;
}

static void arena_restore(JtCtx *c, const ArenaKeep *k) {
  c->arena = k->arena, c->gdn_ckpt = k->gdn_ckpt;
  c->arena_n = k->arena_n, c->save_at = k->save_at, c->work_at = k->work_at;
}

// Point the arena at `off` words into the store with the save cursor at `save_at` words (0 for a fresh block, the mixer's
// width when the block is the MLP's, which sits above it in the same slot). Work keeps coming from the store's top.
static void store_bind(JtCtx *c, const Store *p, size_t off, size_t save_at) {
  c->arena = p->base + off;
  c->arena_n = (size_t)(p->top - c->arena);
  c->save_at = save_at;
  c->work_at = c->arena_n;
}

// Does the batch fill what jt_ctx_alloc sized? The save slots are sized for store_tokens(c), the ckpt slots for cap_rows,
// and the work region for one arena at store_tokens(c). Both sweeps ask store_tokens first (a batch above it takes the
// recompute path), so the checks here only catch a bookkeeping mistake, and they say which number was wrong.
static void store_open(const JtCtx *c, Store *p, const JtBatch *b, int n) {
  if (n != b->n) die("store: %d tokens but the batch has %d", n, b->n);
  if (!no_recompute(c)) die("store: opened without a store (was jt_ctx_alloc asked for --no-recompute?)");
  if (n > store_tokens(c))
    die("store: %d tokens, the store was sized for %d (JT_NO_RECOMPUTE_TOKENS: main.c passes the train split's "
        "max_tokens_per_batch)",
        n, store_tokens(c));
  if (b->rows > c->cap_rows) die("store: %d rows, the context was sized for %d", b->rows, c->cap_rows);
  const size_t saves_pm = store_floats_per_token() * (size_t)n;
  if (saves_pm + store_ckpt_floats(c) > store_floats(c) - store_work_floats(c))
    die("store: %zu floats of slots and %zu of checkpoints do not fit under the %zu-word work region",
        saves_pm, store_ckpt_floats(c), store_work_floats(c));
  // The store proper ends where jt_ctx_alloc put it; anything above that is the guard (JT_NO_RECOMPUTE_GUARD), which no
  // piece may touch. `c->arena_n` is base + store + guard, so the guard's size is the difference.
  if (c->arena_n < base_arena_floats(c) + store_floats(c))
    die("store: the arena holds %zu floats, the store needs %zu", c->arena_n,
        base_arena_floats(c) + store_floats(c));
  p->base = c->arena + base_arena_floats(c);
  p->top = p->base + store_floats(c);
  p->n = n;
}

// ---- the guard above the work region (diagnostic: JT_NO_RECOMPUTE_GUARD=<KB>) ------------------------------------------
// The work region ends at the last word the store owns, and above it is either padding the driver mapped (a few hundred KB
// at most) or nothing: so a piece that writes past its own top buffer does not corrupt anything, it kills the run with an
// MMU fault that names no kernel - which is exactly the state the card was in on 2026-11-11 (fault 0x1592e01000, 1 MB past
// the end of the arena+store). With a guard the same write lands in memory nobody reads, the sweep checks the guard after
// every piece, and the message names the layer, the piece and how far past the end the write went.
static size_t guard_floats(void) {
  const char *s = getenv("JT_NO_RECOMPUTE_GUARD");
  if (s == NULL || s[0] == '\0') return 0;
  char *end = NULL;
  const long kb = strtol(s, &end, 10);
  if (end == s || kb <= 0) return 0;
  return (size_t)kb * 1024 / sizeof(float);
}
#define JT_GUARD_BYTE 0xA5

static void guard_scan(const Store *p, const char *what) {
  const size_t g = guard_floats();
  if (g == 0) return;
  // `p->top` is the store's first word past the end, i.e. the guard's first word, and it does not move as the layer loop
  // re-binds c->arena to each layer's slot (c->arena + base + store would).
  const float *base = p->top;
  float *host = (float *)malloc(g * sizeof(float));
  if (host == NULL) die("out of host memory (the %zu-float guard)", g);
  CK(cudaMemcpy(host, base, g * sizeof(float), cudaMemcpyDeviceToHost));
  const unsigned char *b = (const unsigned char *)host;
  const size_t bytes = g * sizeof(float);
  size_t first = bytes, last = 0, hits = 0;
  for (size_t i = 0; i < bytes; i++) {
    if (b[i] == JT_GUARD_BYTE) continue;
    if (first == bytes) first = i;
    last = i;
    hits++;
  }
  if (hits == 0) { free(host); return; }
  fprintf(stderr, "guard: %s touched the %zu bytes above the store's work region: %zu byte(s) not the poison, "
                  "first at +%zu (float %zu), last at +%zu (float %zu) - a %zu-byte span\n",
          what, bytes, hits, first, first / sizeof(float), last, last / sizeof(float), last - first + 1);
  const size_t show = first + 64 < bytes ? first + 64 : bytes;
  for (size_t i = first; i < show; i += 16) {
    fprintf(stderr, "guard:   +%zu:", i);
    for (size_t j = i; j < i + 16 && j < show; j++) fprintf(stderr, " %02x%s", b[j], b[j] == JT_GUARD_BYTE ? "" : "*");
    fprintf(stderr, "\n");
  }
  free(host);
  die("%s wrote past the end of the store's work region (first byte +%zu, span %zu): that is the step to look at", what,
      first, last - first + 1);
}

// The sweep's own checks: one per piece, naming the layer.
static void guard_check(const Store *p, const char *piece, int layer) {
  if (guard_floats() == 0) return;
  char what[96];
  snprintf(what, sizeof what, "layer %d's %s", layer, piece);
  guard_scan(p, what);
}

// The save cursor after a SAVE pass must be the width this file budgets for that piece: if a piece's layout changed, stop
// here, at the first step, with both numbers instead of letting a slot overlap its neighbour.
static void store_check(const char *piece, int layer, size_t got, size_t want) {
  if (got == want) return;
  die("layer %d's %s saves are %zu floats, --no-recompute budgets %zu (that piece's save layout changed: update the "
      "*_SAVES_PER_TOKEN sums above)",
      layer, piece, got, want);
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

  c->gdn_nseg = (lmax + JT_GDN_SEG - 1) / JT_GDN_SEG;
  if (c->gdn_nseg < 1) c->gdn_nseg = 1; // the store's ckpt slots are sized from it, so it comes before the arena

  // The context's other allocations, sized before the arena because the store's auto rule (below) asks the device for one
  // allocation that covers the arena *with* the store, and the answer has to be for the run's real footprint.
  const size_t ckpt = gdn_ckpt_floats(cap_rows, MT_GDN_HEADS, c->gdn_nseg);
  const size_t sp = gdn_bwd_sp_floats(cap_rows, MT_GDN_HEADS);
  const size_t part = gdn_bwd_part_floats((size_t)cap_tokens, MT_GDN_HEADS);
  const size_t gate = gdn_bwd_gate_floats((size_t)cap_tokens, MT_GDN_HEADS);

  // One allocation holds the arena and, with --no-recompute, the store directly above it (the section above). c->arena_n is
  // the whole allocation's size in that mode, which is how every later call knows the store is there. With the toggle
  // unset, J51 rule 1 decides: the mode is the default when the store fits (store_auto_mode above).
  const size_t arena_base = base_arena_floats(c);
  int store = no_recompute_env(); // -1 = nothing asked, so the memory decides
  if (store < 0) store = store_auto_mode(c, arena_base, ckpt + sp + 2 * part + 2 * gate);
  const size_t guard = store ? guard_floats() : 0; // diagnostic only (JT_NO_RECOMPUTE_GUARD), never part of the store
  c->arena_n = arena_base;
  if (store) c->arena_n += store_floats(c) + guard;
  CK(cudaMalloc((void **)&c->arena, c->arena_n * sizeof(float)));
  c->work_at = arena_base;
  c->save_at = 0;
  const size_t store_n = c->arena_n - arena_base - guard;
  if (guard) { // the poison the sweeps check: a byte pattern, so a partial write is as visible as a whole float. A plain
    // host copy rather than a memset kernel, so the null device (where kernels do not run) poisons it too and the check
    // stays testable without the card.
    const size_t chunk = 1u << 20;
    unsigned char *pat = (unsigned char *)malloc(chunk);
    if (pat == NULL) die("out of host memory (the guard poison)");
    memset(pat, JT_GUARD_BYTE, chunk);
    unsigned char *dst = (unsigned char *)(c->arena + arena_base + store_floats(c));
    for (size_t done = 0; done < guard * sizeof(float); done += chunk) {
      const size_t n = guard * sizeof(float) - done < chunk ? guard * sizeof(float) - done : chunk;
      CK(cudaMemcpy(dst + done, pat, n, cudaMemcpyHostToDevice));
    }
    free(pat);
    CK(cudaStreamSynchronize(c->stream));
    printf("ctx: no-recompute guard: %.2f MB of poison above the work region (JT_NO_RECOMPUTE_GUARD); a write past the "
           "store ends the run at the piece that made it instead of faulting the card\n",
           (double)guard * sizeof(float) / 1e6);
  }

  CK(cudaMalloc((void **)&c->gdn_ckpt, ckpt * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.sp, sp * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dq_part, part * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dk_part, part * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dbeta_part, gate * sizeof(float)));
  CK(cudaMalloc((void **)&c->gdn_work.dg_part, gate * sizeof(float)));

  printf("ctx: lmax %d, arena %.2f GB fp32 (%d tokens x %zu floats + 2 x %d x %d^2), gdn nseg %d, ckpt %.2f GB, bwd scratch %.2f GB\n",
         lmax, (double)arena_base * sizeof(float) / 1e9, cap_tokens, JT_ARENA_FLOATS_PER_TOKEN, MT_FULL_HEADS, lmax,
         c->gdn_nseg, (double)ckpt * sizeof(float) / 1e9, (double)(sp + 2 * part + 2 * gate) * sizeof(float) / 1e9);
  if (store)
    printf("ctx: no-recompute (%s): every layer's forward saves are kept, store %.2f GB on top of the arena "
           "for batches of up to %d tokens (%.2f KB per packed token: %d layers' saves + their post-mixer residuals, %d "
           "GDN checkpoint slots, one shared work region); a bigger batch takes the recompute path\n",
           no_recompute_auto() ? "auto" : "JT_NO_RECOMPUTE", (double)store_n * sizeof(float) / 1e9, store_tokens(c),
           (double)store_floats_per_token() * sizeof(float) / 1024.0, MT_LAYERS, m->n_gdn);
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

// ---- one layer's backward (design §4.6, §6.2, plan M5.1 item 3) ----------------------------------------------------------
// The two halves of a layer each have a backward that reads the saves of ITS OWN forward pass from the arena's base:
// layer_mlp_bwd binds its block at c->arena (src/layer_mlp.c's saves_bind), layer_gdn_mixer_bwd replays its walk from
// c->arena (src/layer_gdn.c's saves_replay) and layer_attn_mixer_bwd indexes its block from c->arena too (src/layer_attn.c).
// One layer_fwd(SAVE) puts the mixer's block at the base and the MLP's above it, so a single save half cannot serve both
// backwards — each half is recomputed with its saves at the base, in the order the layer's maths requires: the MLP adds its
// path to dres first (it is the last thing the forward did, design §4.6), then the mixer's dy is the accumulated value.
//
// `post_mixer` is the fwd's saved residual after the mixer (the MLP's input). `x_in` is the layer's input, which the
// mixer's fwd(SAVE) is re-run from. The two recomputes' outputs are work temporaries: only the saves matter here.
int layer_bwd(JtCtx *c, int i, const float *x_in, const float *post_mixer, float *dres, int n, const JtBatch *b) {
  const MtLayer *ml = mt_layer(i);
  if (c == NULL || ml == NULL || x_in == NULL || post_mixer == NULL || dres == NULL || b == NULL)
    die("layer_bwd: bad arguments (layer %d)", i);
  if (n != b->n) die("layer_bwd: n %d but the batch has %d tokens", n, b->n);
  const LayerWeights *lw = &c->model->layers[i];
  const size_t mark = jt_work_mark(c);
  float *x_out = jt_work_alloc(c, (size_t)n * MT_HIDDEN); // the recomputes' output; nothing reads it

  // 1. the MLP: recompute it from its residual input into the base, then its backward (dres += the MLP's path).
  jt_saves_reset(c);
  int rc = layer_mlp_fwd(c, lw, post_mixer, n, b, x_out, JT_FWD_SAVE);
  if (rc == JT_LAYER_OK) rc = layer_mlp_bwd(c, lw, post_mixer, dres, n, b);

  // 2. the mixer: recompute it from the layer's input into the base, then its backward (dres += the mixer's path).
  if (rc == JT_LAYER_OK) {
    jt_saves_reset(c);
    rc = ml->full_attention ? layer_attn_mixer_fwd(c, lw, x_in, n, b, x_out, JT_FWD_SAVE)
                            : layer_gdn_mixer_fwd(c, lw, x_in, n, b, x_out, JT_FWD_SAVE);
  }
  if (rc == JT_LAYER_OK) {
    rc = ml->full_attention ? layer_attn_mixer_bwd(c, lw, x_in, dres, n, b)
                            : layer_gdn_mixer_bwd(c, lw, x_in, dres, n, b);
  }
  jt_work_restore(c, mark);
  return rc;
}

// ---- the two sweeps with the store (plan M8 item 4) ---------------------------------------------------------------------
// --no-recompute's forward: every layer runs in JT_FWD_SAVE and its saves land in its own slot, at the offsets its backward
// will bind them from (the mixer's half at the slot's base, the MLP's above it). Nothing is copied afterwards and nothing
// is recomputed later, which is the whole of the mode. The mixer's output IS the layer's post-mixer residual, so it is
// written straight into `pm`: the caller's `post_mixer` slab when there is one (--dump-backward's key) and the store's
// otherwise, and the MLP then continues from it exactly as layer_fwd does.
static int store_forward(JtCtx *c, const Store *st, const JtBatch *b, float *saved_in, float *post_mixer) {
  const int n = st->n;
  const size_t stride = (size_t)n * MT_HIDDEN;
  int gdn_index = 0;
  for (int i = 0; i < MT_LAYERS; i++) {
    const MtLayer *ml = mt_layer(i);
    const LayerWeights *lw = &c->model->layers[i];
    const float *x_in = saved_in + (size_t)i * stride;
    float *pm = post_mixer != NULL ? post_mixer + (size_t)i * stride : st->base + store_pm_off(i, n);
    const size_t slot = store_slot_off(i, n);
    const size_t mixer = mixer_saves_per_token(ml) * (size_t)n;

    store_bind(c, st, slot, 0); // the mixer's saves at the slot's base, its work from the store's top
    if (!ml->full_attention) c->gdn_ckpt = st->base + store_ckpt_off(c, gdn_index++, n);
    int rc = ml->full_attention ? layer_attn_mixer_fwd(c, lw, x_in, n, b, pm, JT_FWD_SAVE)
                                : layer_gdn_mixer_fwd(c, lw, x_in, n, b, pm, JT_FWD_SAVE);
    if (rc != JT_LAYER_OK) return rc;
    store_check("mixer", i, c->save_at, mixer);
    guard_check(st, "mixer forward", i);

    store_bind(c, st, slot + mixer, 0); // the MLP's saves directly above the mixer's, its work from the same top
    rc = layer_mlp_fwd(c, lw, pm, n, b, saved_in + (size_t)(i + 1) * stride, JT_FWD_SAVE);
    if (rc != JT_LAYER_OK) return rc;
    store_check("MLP", i, c->save_at, JT_MLP_SAVES_PER_TOKEN * (size_t)n);
    guard_check(st, "MLP forward", i);
  }
  return JT_LAYER_OK;
}

// --no-recompute's backward: the same loop as model_backward's, with the recompute replaced by a jump to the layer's slot.
// The two halves keep the order the maths needs (the MLP's path first, then the mixer's, since the mixer's dy is the sum of
// both) and each binds its saves where store_forward left them, so nothing here recomputes anything.
static int store_backward(JtCtx *c, const Store *st, const JtBatch *b, const float *saved_in, float *post_mixer,
                          float *grad_out, float *dres) {
  const int n = st->n;
  const size_t stride = (size_t)n * MT_HIDDEN;
  int gdn_index = c->model->n_gdn;
  for (int i = MT_LAYERS - 1; i >= 0; i--) {
    const MtLayer *ml = mt_layer(i);
    const LayerWeights *lw = &c->model->layers[i];
    const float *x_in = saved_in + (size_t)i * stride;
    float *pm = post_mixer != NULL ? post_mixer + (size_t)i * stride : st->base + store_pm_off(i, n);
    const size_t slot = store_slot_off(i, n);
    const size_t mixer = mixer_saves_per_token(ml) * (size_t)n;
    if (!ml->full_attention) gdn_index--;
    if (grad_out != NULL) // grad.layer.{i}.out: the entry value, before any backward writes dres
      CK(cudaMemcpyAsync(grad_out + (size_t)i * stride, dres, stride * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));

    store_bind(c, st, slot + mixer, 0); // the MLP's block, at its own base (jt_saves_reset + the block's own width)
    int rc = layer_mlp_bwd(c, lw, pm, dres, n, b);
    if (rc != JT_LAYER_OK) return rc;
    guard_check(st, "MLP backward", i);

    store_bind(c, st, slot, mixer); // the mixer's block at the slot's base, the cursor where its replay expects it
    if (!ml->full_attention) c->gdn_ckpt = st->base + store_ckpt_off(c, gdn_index, n);
    rc = ml->full_attention ? layer_attn_mixer_bwd(c, lw, x_in, dres, n, b)
                            : layer_gdn_mixer_bwd(c, lw, x_in, dres, n, b);
    if (rc != JT_LAYER_OK) return rc;
    guard_check(st, "mixer backward", i);
  }
  return JT_LAYER_OK;
}

int model_forward(JtCtx *c, const int *ids, int n, const JtBatch *b, float *saved_in, float *post_mixer) {
  if (c == NULL || ids == NULL || b == NULL || saved_in == NULL) die("model_forward: bad arguments");
  if (n != b->n) die("model_forward: n %d but the batch has %d tokens", n, b->n);
  const size_t stride = (size_t)n * MT_HIDDEN;
  jt_embed_gather(ids, c->model->embed, saved_in, n, MT_HIDDEN, c->stream); // design §8.1: only the used rows widen
  if (store_serves(c, n)) { // plan M8 item 4: keep every layer's saves for the backward (the section above)
    Store st;
    store_open(c, &st, b, n);
    ArenaKeep keep;
    arena_keep(c, &keep);
    const int rc = store_forward(c, &st, b, saved_in, post_mixer);
    arena_restore(c, &keep);
    return rc;
  }
  for (int i = 0; i < MT_LAYERS; i++) {
    const float *in = saved_in + (size_t)i * stride;
    float *out = saved_in + (size_t)(i + 1) * stride;
    int rc = layer_fwd(c, i, in, n, b, out, post_mixer != NULL ? post_mixer + (size_t)i * stride : NULL, JT_FWD_ONLY);
    if (rc != JT_LAYER_OK) return rc;
  }
  return JT_LAYER_OK;
}

// ---- the backward sweep (design §6.2, plan M5.1 item 4) -----------------------------------------------------------------
// The forward kept only each layer's input (model_forward's saved_in), so the backward walks the layers in reverse and
// re-runs each one from its saved input before differentiating it:
//
//   for i = 23 .. 0:   the recompute of layer i from saved_in[i], then its backward
//                      dres:  dL/d(layer i's out)  ->  dL/d(layer i's in)
//
// The layer's dx is the next layer's dy: layer i's output gradient (dres on entry) becomes layer i-1's input gradient
// (dres on return), which is why the loop needs one gradient buffer and not 24 (design §6.2's "The dx of layer i is the
// dy of layer i-1"). Design §6.2's cost is one extra forward, and its memory is one layer's scratch, reused.
//
// Why this does not simply call layer_fwd(SAVE) + layer_bwd: those two are correct (layer_bwd recomputes both halves itself,
// above) but they cost a third forward, because the mixer's save block is thrown away when the MLP's fwd(SAVE) takes the
// arena's base. Here the mixer's block is parked in the work half instead — its size is the save cursor right after the
// mixer's fwd(SAVE), so nothing here knows any piece's layout — and copied back before the mixer's backward, which makes
// the sweep's recompute exactly one forward, as design §6.2 budgets. Two device-to-device copies per layer (the mixer's
// ~24 K floats per token each way) against a saved forward pass.
//
// `saved_in` is model_forward's buffer (25 slabs of n * MT_HIDDEN). `post_mixer`, when not NULL, is MT_LAYERS slabs and
// receives each layer's post-mixer residual: the mixer's fwd(SAVE) writes its output there, which IS post_mixer, so the
// recompute produces dump_ref's `layer.{i}.post_mixer` key at no extra cost. `grad_out`, when not NULL, is MT_LAYERS slabs
// and receives dL/d(layer i's output) as it stood on entry to layer i's backward — dump_ref's `grad.layer.{i}.out`, which
// the backward overwrites in place and so must be copied out first. `dres` is n * MT_HIDDEN, in as dL/d(layer 23's output)
// (head_bwd's result) and out as dL/d(the embedding's output). Both optional buffers exist for --dump-backward's keys; a
// training step passes NULL and saves nothing (design §6.2).
//
// The scratch (the recompute's output, one slab for post_mixer when the caller has no buffer, and the park) comes from the
// arena's work half and is given back before returning; each half's own saves and temporaries are bracketed by the piece
// that takes them. No allocation happens here.
//
// With --no-recompute (plan M8 item 4) there is no recompute: model_forward kept every layer's saves in the store, so this
// is store_backward's loop instead — the same order and the same backwards, one forward cheaper per layer.
int model_backward(JtCtx *c, const float *saved_in, int n, const JtBatch *b, float *post_mixer, float *grad_out,
                   float *dres) {
  if (c == NULL || c->model == NULL || saved_in == NULL || b == NULL || dres == NULL)
    die("model_backward: bad arguments");
  if (n != b->n) die("model_backward: n %d but the batch has %d tokens", n, b->n);
  if (store_serves(c, n)) {
    Store st;
    store_open(c, &st, b, n);
    ArenaKeep keep;
    arena_keep(c, &keep);
    const int rc = store_backward(c, &st, b, saved_in, post_mixer, grad_out, dres);
    arena_restore(c, &keep);
    return rc;
  }
  const size_t stride = (size_t)n * MT_HIDDEN;
  const size_t mark = jt_work_mark(c);
  float *x_out = jt_work_alloc(c, stride);                       // the recompute's output; nothing reads it
  float *one_pm = post_mixer == NULL ? jt_work_alloc(c, stride) : NULL; // one post_mixer slab, reused, when nobody wants them
  int rc = JT_LAYER_OK;
  for (int i = MT_LAYERS - 1; i >= 0 && rc == JT_LAYER_OK; i--) {
    const MtLayer *ml = mt_layer(i);
    const LayerWeights *lw = &c->model->layers[i];
    const float *x_in = saved_in + (size_t)i * stride;
    float *lp = one_pm != NULL ? one_pm : post_mixer + (size_t)i * stride; // this layer's post-mixer residual
    const size_t at = jt_work_mark(c);

    // 1. the mixer's recompute from the layer's input, its saves at the arena's base (lp receives its output = post_mixer).
    jt_saves_reset(c);
    rc = ml->full_attention ? layer_attn_mixer_fwd(c, lw, x_in, n, b, lp, JT_FWD_SAVE)
                            : layer_gdn_mixer_fwd(c, lw, x_in, n, b, lp, JT_FWD_SAVE);
    if (rc != JT_LAYER_OK) break;

    // 2. park those saves (their size is the save cursor, so no piece's layout is assumed) and hand the base to the MLP.
    const size_t mixer_saves = c->save_at;
    float *park = jt_work_alloc(c, mixer_saves);
    CK(cudaMemcpyAsync(park, c->arena, mixer_saves * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
    if (grad_out != NULL) { // grad.layer.{i}.out: the entry value, before any backward writes dres
      CK(cudaMemcpyAsync(grad_out + (size_t)i * stride, dres, stride * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
    }
    jt_saves_reset(c);
    rc = layer_mlp_fwd(c, lw, lp, n, b, x_out, JT_FWD_SAVE);
    if (rc == JT_LAYER_OK) rc = layer_mlp_bwd(c, lw, lp, dres, n, b); // the MLP is the last thing the forward did
    if (rc != JT_LAYER_OK) break;

    // 3. the mixer's saves back at the base (and the cursor with them, which its replay checks), then its backward.
    CK(cudaMemcpyAsync(c->arena, park, mixer_saves * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
    jt_saves_reset(c);
    (void)jt_save_alloc(c, mixer_saves);
    rc = ml->full_attention ? layer_attn_mixer_bwd(c, lw, x_in, dres, n, b)
                            : layer_gdn_mixer_bwd(c, lw, x_in, dres, n, b);
    jt_work_restore(c, at);
  }
  jt_work_restore(c, mark);
  return rc;
}
