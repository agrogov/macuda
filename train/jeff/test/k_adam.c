// test/k_adam.c — the `adam` group of test_kernels (plan M2.6): the optimizer kernels against ref/adam.c, the schedule
// against a table Python printed, and the flat parameter buffer of src/lora.c (its size, its layout, and a real
// save/load round trip of an adapter directory).
//
// What is checked
//   1. sumsq_partial/sumsq_final, chained: the global sum of squares of a random gradient, block-sliced into exactly
//      K_ADAM_BLOCKS partials with no atomics. The partials are compared one by one as well, which is what pins the
//      slicing (a wrong slice map still sums to the same total); the sum and sqrt(sum) are the numbers torch's
//      clip_grad_norm_ uses.
//   2. clip_scale with a coefficient from the reference's norm (coef < 1 and coef == 1), plus the clipped norm.
//   3. adamw_step: one step, then ten steps with fresh random gradients, against ref_adamw_step, with the two learning
//      rate groups (LoRA below the split, the readout at and above it). Gate 1e-6 (plan M2.6). The ten-step case runs on
//      the real flat buffer, so the real size (10,484,736) and the real split (10,223,616) are exercised.
//   4. lr_at against values Python printed with train.py's own lambda (the snippet is in the comment below), at
//      total = 100 for every step 0..100, plus the base-lr scaling and the total = 0/1 edge.
//   5. lora_alloc: the flat buffer is 151 entries and 10,484,736 floats (plan M2.6's Done-when number; lora_alloc prints
//      "trainable 10,484,736"), the entries are contiguous and cover the buffer exactly, every A/B is [rank,in]+[out,rank]
//      with rank 16, the readout is last at offset 10,223,616, and the stems are the model table's, each once.
//   6. lora_load_adapter/lora_save_adapter: a synthetic adapter directory written the way CLM-8B's lora.py writes one
//      (fp32 factors, readout fp32 and then bf16), loaded and compared value by value; then the buffer is saved, the
//      file's key set and dtype checked against the table and design §9.2, zeroed, loaded back and compared again.
//
// The schedule's Python (plan M2.6 asks for the snippet that prints f(s) for total = 100 and s = 0..100):
//
//   import math
//   total = 100
//   def f(s):
//       return min(1.0, (s + 1) / max(1, int(total * .03))) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
//   print([repr(f(s)) for s in range(total + 1)])
//
// It is train.py's LambdaLR lambda (train.py:307-308) with the number of optimizer steps already taken as s; lr_at
// applies it to a group's base lr.
//
// The adapter files are synthetic: random factors from rnd_fill, no real data (plan rule 5). They are written under
// build/ (git-ignored) and are not cleaned up, so a failed run leaves its input behind. On the shim's null device every
// CHECK_* prints `(launch only)` and counts nothing; the host-side checks (jt_check_*1) and the file I/O still run.
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "harness.h"
#include "adam.h" // kernels/adam.h
#include "lora.h" // src/lora.h
#include "st.h"   // src/st.h: the round trip's files
#include "../ref/adam.h"

// The optimizer's constants: train.py's defaults and design §2.2.
#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1e-8f
#define ADAM_LR_LORA 2e-4f
#define ADAM_LR_READOUT 5e-6f

#define ADAPTER_DIR "build/lora-test"

// f(s) for total = 100, s = 0..100, printed by the Python in the header comment.
static const double LR_TABLE_100[101] = {
    0.3333333333333333, 0.6665186347763861, 0.9991120277927222, 0.998002884071386,  0.9964516155915151,
    0.994459753267812,  0.9920292628279099, 0.9891625428724363, 0.985862422507884,  0.9821321585546243,
    0.9779754323328191, 0.9733963460294015, 0.9683994186497131, 0.9629895815577915, 0.9571721736097089,
    0.9509529358847655, 0.9443380060197386, 0.9373339121517746, 0.9299475664759068, 0.9221862584235527,
    0.9140576474687263, 0.9055697555690607, 0.8967309592491052, 0.8875499813337068, 0.8780358823396351,
    0.8681980515339464, 0.8580461976679099, 0.8475903393956433, 0.8368407953869104, 0.8258081741438394,
    0.814503363531613,  0.8029375200334589, 0.7911220577405484, 0.779068637087667,  0.7667891533457718,
    0.7542957248827961, 0.7416006812042827, 0.7287165507856512, 0.7156560487081052, 0.7024320641103812,
    0.6890576474687263, 0.6755459977176532, 0.6619104492241846, 0.6481644586284442, 0.6343215915635761,
    0.6203955092681039, 0.606399955103937,  0.5923487409933315, 0.5782557337881911, 0.5641348415851577,
    0.55,               0.5358651584148423, 0.521744266211809,  0.5076512590066685, 0.49360004489606313,
    0.47960449073189615, 0.46567840843642394, 0.451835541371556, 0.43808955077581535, 0.42445400228234687,
    0.41094235253127376, 0.3975679358896189, 0.384343951291895, 0.37128344921434864, 0.3583993187957173,
    0.345704275117204,  0.33321084665422807, 0.32093136291233293, 0.3088779422594514, 0.2970624799665412,
    0.2854966364683872, 0.27419182585616053, 0.2631592046130896, 0.25240966060435677, 0.24195380233209007,
    0.23180194846605365, 0.2219641176603649, 0.2124500186662932, 0.2032690407508949, 0.19443024443093931,
    0.18594235253127372, 0.17781374157644725, 0.17005243352409333, 0.16266608784822534, 0.1556619939802614,
    0.1490470641152345, 0.1428278263902913, 0.1370104184422086, 0.1316005813502869, 0.12660365397059856,
    0.12202456766718092, 0.11786784144537568, 0.11413757749211607, 0.11083745712756371, 0.10797073717209008,
    0.10554024673218806, 0.10354838440848502, 0.101997115928614, 0.1008879722072778, 0.10022204783542078, 0.1,
};

static float *f_alloc(size_t n) {
  float *p = (float *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "k_adam: out of memory (%zu floats)\n", n);
    exit(1);
  }
  return p;
}

// ---- 1: the two-pass gradient norm ------------------------------------------------------------------------------------
static void case_sumsq(int n, uint32_t seed, double tol) {
  char name[96];
  float *g = f_alloc((size_t)n);
  float *partials_want = f_alloc(K_ADAM_BLOCKS);
  rnd_fill(g, (size_t)n, -1.0, 1.0, seed);

  ref_sumsq_partial(g, (size_t)n, partials_want, K_ADAM_BLOCKS);
  float want;
  ref_sumsq_final(partials_want, K_ADAM_BLOCKS, &want);

  float *gd = dev_upload(g, (size_t)n);
  float *pd = dev_alloc(K_ADAM_BLOCKS * sizeof *pd);
  float *od = dev_alloc(sizeof *od);
  k_sumsq_partial(gd, n, pd, K_ADAM_BLOCKS, 0);
  k_sumsq_final(pd, K_ADAM_BLOCKS, od, 0);
  JT_SYNC_DEBUG;

  float got, got_partials[K_ADAM_BLOCKS];
  dev_download(od, &got, 1);
  dev_download(pd, got_partials, K_ADAM_BLOCKS);

  snprintf(name, sizeof name, "sumsq/n%d sum", n);
  CHECK_REL1(name, got, want, tol);
  snprintf(name, sizeof name, "sumsq/n%d norm", n);
  CHECK_REL1(name, sqrt((double)got), sqrt((double)want), tol);
  snprintf(name, sizeof name, "sumsq/n%d partials", n);
  CHECK_REL(name, got_partials, partials_want, K_ADAM_BLOCKS, JT_TOL_FWD);

  dev_free(gd);
  dev_free(pd);
  dev_free(od);
  free(g);
  free(partials_want);
}

// g, clipped by the reference's own coefficient, against the device's clip_scale.
static void case_clip(int n, float amplitude, uint32_t seed) {
  char name[96];
  float *g = f_alloc((size_t)n), *want = f_alloc((size_t)n);
  rnd_fill(g, (size_t)n, -amplitude, amplitude, seed);
  memcpy(want, g, (size_t)n * sizeof *want);

  float sq;
  ref_sumsq(g, (size_t)n, K_ADAM_BLOCKS, &sq);
  const float coef = ref_clip_coef(sqrtf(sq));
  ref_clip_scale(want, (size_t)n, coef);

  float *gd = dev_upload(g, (size_t)n);
  k_clip_scale(gd, n, coef, 0);
  JT_SYNC_DEBUG;
  float *got = f_alloc((size_t)n);
  dev_download(gd, got, (size_t)n);

  snprintf(name, sizeof name, "clip/n%d coef=%.3g", n, (double)coef);
  CHECK_REL(name, got, want, (size_t)n, 1e-6);
  // The clipped buffer's norm is the old one times the coefficient: the check the plan asks for, to 1e-6.
  float sq_clipped;
  ref_sumsq(want, (size_t)n, K_ADAM_BLOCKS, &sq_clipped);
  snprintf(name, sizeof name, "clip/n%d clipped_norm", n);
  CHECK_REL1(name, sqrt((double)sq_clipped), (double)coef * sqrt((double)sq), 1e-6);

  dev_free(gd);
  free(g);
  free(want);
  free(got);
}

// ---- 2: the Adam step ------------------------------------------------------------------------------------------------
// `flat` (optional): run on that Lora's four buffers instead of fresh ones, when n is its size. The ten-step case does
// that, so the real buffer and the real lr split are what is exercised (and 168 MB are not allocated twice).
static void case_adam(int n, int split, int steps, uint32_t seed, Lora *flat) {
  char name[128];
  const size_t N = (size_t)n;
  const size_t bytes = N * sizeof(float);
  float *p_ref = f_alloc(N), *m_ref = f_alloc(N), *v_ref = f_alloc(N), *g = f_alloc(N);
  float *p_got = f_alloc(N), *m_got = f_alloc(N), *v_got = f_alloc(N);

  rnd_fill(p_ref, N, -1.0, 1.0, seed);
  rnd_fill(m_ref, N, -0.1, 0.1, seed + 1u);
  rnd_fill(v_ref, N, 1e-8, 1e-4, seed + 2u); // Adam's second moment is non-negative: sqrt() of a negative would be NaN

  float *pd, *md, *vd, *gd;
  const int own = flat == NULL || (size_t)n != flat->n_param;
  if (own) {
    pd = dev_upload(p_ref, N);
    md = dev_upload(m_ref, N);
    vd = dev_upload(v_ref, N);
    gd = dev_alloc(bytes);
  } else {
    pd = flat->param;
    md = flat->m;
    vd = flat->v;
    gd = flat->grad;
    CK(cudaMemcpy(pd, p_ref, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(md, m_ref, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(vd, v_ref, bytes, cudaMemcpyHostToDevice));
  }

  for (int t = 1; t <= steps; t++) {
    rnd_fill(g, N, -1.0, 1.0, seed + 100u * (uint32_t)t);
    float bias1, bias2;
    ref_adam_bias_corr(ADAM_B1, ADAM_B2, t, &bias1, &bias2);
    ref_adamw_step(p_ref, g, m_ref, v_ref, N, (size_t)split, ADAM_LR_LORA, ADAM_LR_READOUT, ADAM_B1, ADAM_B2, ADAM_EPS,
                   bias1, bias2);
    CK(cudaMemcpy(gd, g, bytes, cudaMemcpyHostToDevice));
    k_adamw_step(pd, gd, md, vd, n, split, ADAM_LR_LORA, ADAM_LR_READOUT, ADAM_B1, ADAM_B2, ADAM_EPS, bias1, bias2, 0);
    JT_SYNC_DEBUG;
    if (t == 1 || t == steps) { // the plan's two comparisons: one step, then the whole run
      dev_download(pd, p_got, N);
      snprintf(name, sizeof name, "adam/n%d_split%d_step%d p", n, split, t);
      CHECK_REL(name, p_got, p_ref, N, 1e-6);
    }
  }

  dev_download(md, m_got, N);
  dev_download(vd, v_got, N);
  snprintf(name, sizeof name, "adam/n%d_split%d m", n, split);
  CHECK_REL(name, m_got, m_ref, N, 1e-6);
  snprintf(name, sizeof name, "adam/n%d_split%d v", n, split);
  CHECK_REL(name, v_got, v_ref, N, 1e-6);

  if (own) {
    dev_free(pd);
    dev_free(md);
    dev_free(vd);
    dev_free(gd);
  }
  free(p_ref);
  free(m_ref);
  free(v_ref);
  free(g);
  free(p_got);
  free(m_got);
  free(v_got);
}

// ---- 3: the schedule -------------------------------------------------------------------------------------------------
static void case_lr_at(void) {
  int bad = 0;
  for (int s = 0; s <= 100; s++) {
    const double got = lr_at(s, 100, 1.0), want = LR_TABLE_100[s];
    if (fabs(got - want) > 1e-12 * fabs(want)) {
      printf("FAIL lr_at total=100 s=%d got %.17g want %.17g\n", s, got, want);
      jt_fail_inc();
      bad++;
    }
  }
  if (!bad) printf("ok  lr_at total=100 s=0..100 (101 values, rel<1e-12)\n");

  // The base lr is a plain multiplier (torch's LambdaLR), for both of train.py's groups.
  jt_check_rel1("lr_at lora_lr s=7", lr_at(7, 100, (double)ADAM_LR_LORA), (double)ADAM_LR_LORA * LR_TABLE_100[7], 1e-12);
  jt_check_rel1("lr_at readout_lr s=99", lr_at(99, 100, (double)ADAM_LR_READOUT),
                (double)ADAM_LR_READOUT * LR_TABLE_100[99], 1e-12);
  // total < 1 is clamped to 1 (train.py never sees it): warmup is done and the cosine is at s = total.
  jt_check_rel1("lr_at total=0 s=0", lr_at(0, 0, 1.0), 1.0, 1e-12);
  jt_check_rel1("lr_at total=1 s=0", lr_at(0, 1, 1.0), 1.0, 1e-12);
  jt_check_rel1("lr_at total=1 s=1", lr_at(1, 1, 1.0), 0.1, 1e-12);
}

// ---- 4: the flat buffer's layout --------------------------------------------------------------------------------------
static void case_lora_table(Lora *l) {
  const LoraEntry *ro = &l->ent[LORA_READOUT_ENTRY];

  jt_check_rel1("lora/entries 151", (double)l->n_ent, (double)LORA_ENTRIES, 1e-12);
  jt_check_rel1("lora/n_param 10,484,736", (double)l->n_param, 10484736.0, 1e-12);
  jt_check_rel1("lora/readout_off 10,223,616", (double)l->readout_off, 10223616.0, 1e-12);
  jt_check_rel1("lora/readout floats 261,120", (double)ro->count, 261120.0, 1e-12);
  jt_check_abs1("lora/readout rows 255", (double)ro->out, (double)MT_READOUT_OUT, 0.5);
  jt_check_abs1("lora/readout cols 1024", (double)ro->in, (double)MT_READOUT_IN, 0.5);
  jt_check_abs1("lora/readout rank 0", (double)ro->rank, 0.0, 0.5);
  jt_check_abs1("lora/readout stem", (double)strcmp(ro->stem, "readout"), 0.0, 0.5);
  jt_check_abs1("lora/readout offset", (double)(ro->off == l->readout_off), 1.0, 0.5);

  // Every entry owns exactly the floats its shape says, and the entries tile the buffer with no gap and no overlap.
  size_t off = 0;
  int shape_bad = 0, pairs = 0, rank_bad = 0, dup = 0;
  for (int k = 0; k < l->n_ent; k++) {
    const LoraEntry *e = &l->ent[k];
    const size_t want = e->rank > 0 ? (size_t)e->rank * (size_t)e->in + (size_t)e->out * (size_t)e->rank
                                    : (size_t)e->in * (size_t)e->out;
    if (e->off != off || e->count != want) shape_bad++;
    if (e->rank > 0) {
      pairs++;
      if (e->rank != MT_LORA_RANK || e->b_off != e->off + (size_t)e->rank * (size_t)e->in) rank_bad++;
    }
    for (int j = 0; j < k; j++)
      if (strcmp(l->ent[j].stem, e->stem) == 0) dup++;
    off += e->count;
  }
  jt_check_abs1("lora/entries contiguous and shaped", (double)shape_bad, 0.0, 0.5);
  jt_check_abs1("lora/entries cover the buffer", (double)(off == l->n_param), 1.0, 0.5);
  jt_check_abs1("lora/150 LoRA pairs", (double)pairs, (double)MT_LORA_PAIRS, 0.5);
  jt_check_abs1("lora/rank 16 and A then B", (double)rank_bad, 0.0, 0.5);
  jt_check_abs1("lora/stems unique", (double)dup, 0.0, 0.5);

  // The stems and shapes are the model table's (design §9.2's key set is built from them).
  int stem_bad = 0;
  for (int k = 0; k < MT_LORA_PAIRS; k++) {
    const MtLinear *m = mt_lora_pair(k);
    char stem[LORA_STEM_MAX];
    const size_t len = strlen(m->name);
    if (len <= 7 || strcmp(m->name + len - 7, ".weight") != 0) {
      stem_bad++;
      continue;
    }
    memcpy(stem, m->name, len - 7);
    stem[len - 7] = '\0';
    if (strcmp(stem, l->ent[k].stem) != 0 || l->ent[k].in != m->in || l->ent[k].out != m->out) stem_bad++;
  }
  jt_check_abs1("lora/stems and shapes from the model table", (double)stem_bad, 0.0, 0.5);
}

// ---- 5: adapter files ------------------------------------------------------------------------------------------------
static void ensure_dir(const char *path) {
  if (mkdir(path, 0777) != 0 && errno != EEXIST) {
    fprintf(stderr, "k_adam: mkdir %s: %s\n", path, strerror(errno));
    exit(1);
  }
}

// One BF16 tensor in a safetensors file, by hand: the header of design §3.2/§9.2, padded to a multiple of 8, then the
// data. st.c writes F32 only, and a checkpoint's readout is bf16, so the test builds that file itself.
static void write_bf16_tensor(const char *path, const char *name, int rows, int cols, const float *v) {
  const size_t n = (size_t)rows * (size_t)cols;
  char header[512];
  const int len = snprintf(header, sizeof header, "{\"%s\":{\"dtype\":\"BF16\",\"shape\":[%d,%d],\"data_offsets\":[0,%zu]}}",
                           name, rows, cols, n * sizeof(unsigned short));
  size_t hlen = len > 0 ? (size_t)len : 0;
  while (hlen % 8 != 0) hlen++; // pad with spaces
  unsigned char pad[8] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
  const unsigned long long hlen64 = (unsigned long long)hlen;
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    fprintf(stderr, "k_adam: cannot write %s: %s\n", path, strerror(errno));
    exit(1);
  }
  fwrite(&hlen64, 8, 1, f); // little-endian host: both x86-64 and arm64 here
  fwrite(header, 1, (size_t)len, f);
  fwrite(pad, 1, hlen - (size_t)len, f);
  for (size_t i = 0; i < n; i++) {
    const unsigned short b = jt_f32_to_bf16(v[i]);
    fwrite(&b, sizeof b, 1, f);
  }
  fclose(f);
}

// The 300 factors plus a readout, laid out as `flat` (the flat buffer's element order), written the way CLM-8B's
// lora.py writes an adapter: `{stem}.lora_A` / `{stem}.lora_B` fp32, and readout.safetensors `weight` fp32 or bf16.
static void write_adapter_dir(const Lora *l, const char *dir, const float *flat, int readout_bf16) {
  ensure_dir(dir);
  const int nfac = 2 * MT_LORA_PAIRS;
  char(*names)[LORA_STEM_MAX + 16] = (char(*)[LORA_STEM_MAX + 16])calloc((size_t)nfac, sizeof *names);
  const char **namep = (const char **)calloc((size_t)nfac, sizeof *namep);
  int(*shapes)[ST_MAX_DIMS] = (int(*)[ST_MAX_DIMS])calloc((size_t)nfac, sizeof *shapes);
  int *ndims = (int *)calloc((size_t)nfac, sizeof *ndims);
  const float **data = (const float **)calloc((size_t)nfac, sizeof *data);
  if (names == NULL || namep == NULL || shapes == NULL || ndims == NULL || data == NULL) {
    fprintf(stderr, "k_adam: out of memory (adapter fixture)\n");
    exit(1);
  }
  for (int k = 0; k < MT_LORA_PAIRS; k++) {
    const LoraEntry *e = &l->ent[k];
    snprintf(names[2 * k], sizeof names[2 * k], "%s.lora_A", e->stem);
    shapes[2 * k][0] = e->rank;
    shapes[2 * k][1] = e->in;
    data[2 * k] = flat + e->off;
    ndims[2 * k] = 2;
    namep[2 * k] = names[2 * k];
    snprintf(names[2 * k + 1], sizeof names[2 * k + 1], "%s.lora_B", e->stem);
    shapes[2 * k + 1][0] = e->out;
    shapes[2 * k + 1][1] = e->rank;
    data[2 * k + 1] = flat + e->b_off;
    ndims[2 * k + 1] = 2;
    namep[2 * k + 1] = names[2 * k + 1];
  }
  char path[512];
  snprintf(path, sizeof path, "%s/adapter_model.safetensors", dir);
  st_write(path, nfac, namep, (const int(*)[ST_MAX_DIMS])shapes, ndims, data);

  const LoraEntry *ro = &l->ent[LORA_READOUT_ENTRY];
  snprintf(path, sizeof path, "%s/readout.safetensors", dir);
  if (readout_bf16) {
    write_bf16_tensor(path, mt_readout_name, ro->out, ro->in, flat + ro->off);
  } else {
    const int shape[ST_MAX_DIMS] = {ro->out, ro->in};
    const float *d = flat + ro->off;
    const char *nm = mt_readout_name;
    st_write(path, 1, &nm, &shape, (const int[]){2}, &d);
  }
  free(names);
  free(namep);
  free(shapes);
  free(ndims);
  free((void *)data);
}

static int file_contains(const char *path, const char *needle) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) return 0;
  char buf[8192];
  const size_t got = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  buf[got] = '\0';
  return strstr(buf, needle) != NULL;
}

static void case_adapter(Lora *l) {
  const size_t N = l->n_param;
  char path[512], name[160];
  float *expect = f_alloc(N), *expect_bf16 = f_alloc(N), *got = f_alloc(N);

  // Random but deterministic factor values in the flat layout: distinguishable per entry, small like real factors.
  for (int k = 0; k < l->n_ent; k++)
    rnd_fill(expect + l->ent[k].off, l->ent[k].count, -0.05, 0.05, 500u + (uint32_t)k);
  memcpy(expect_bf16, expect, N * sizeof *expect_bf16);
  {
    const LoraEntry *ro = &l->ent[LORA_READOUT_ENTRY];
    for (size_t i = 0; i < ro->count; i++) // what a bf16 readout file can hold
      expect_bf16[ro->off + i] = jt_bf16_to_f32(jt_f32_to_bf16(expect[ro->off + i]));
  }

  // 1. A Python-written adapter with an fp32 readout loads exactly.
  write_adapter_dir(l, ADAPTER_DIR, expect, 0);
  lora_load_adapter(l, ADAPTER_DIR);
  JT_SYNC_DEBUG;
  dev_download(l->param, got, N);
  CHECK_REL("adapter/load fp32", got, expect, N, 1e-6);

  // 2. Saving it again and loading it back gives the same buffer, and the file is the format design §9.2 asks for.
  const LoraMeta meta = {"~/models/jeff-base", "soft", "none"};
  snprintf(path, sizeof path, "%s/saved", ADAPTER_DIR);
  lora_save_adapter(l, path, 7, 0.5123456, &meta);

  St *st = st_open(ADAPTER_DIR "/saved/adapter_model.safetensors");
  jt_check_abs1("adapter/saved 300 factors", (double)st_count(st), 300.0, 0.5);
  int missing = 0, nonf32 = 0;
  for (int k = 0; k < l->n_ent; k++) {
    const LoraEntry *e = &l->ent[k];
    if (e->rank > 0) {
      for (int half = 0; half < 2; half++) {
        snprintf(name, sizeof name, "%s.lora_%c", e->stem, half == 0 ? 'A' : 'B');
        const StTensor *t = st_find(st, name);
        if (t == NULL) {
          missing++;
        } else if (t->dtype != ST_DTYPE_F32) {
          nonf32++;
        } else if (t->ndim != 2 || (half == 0 ? (t->shape[0] != e->rank || t->shape[1] != e->in)
                                             : (t->shape[0] != e->out || t->shape[1] != e->rank))) {
          missing++;
        }
      }
    }
  }
  snprintf(name, sizeof name, "%s.lora_A", l->ent[LORA_READOUT_ENTRY].stem); // the readout has no factors in this file
  if (st_find(st, name) != NULL) missing++;
  st_close(st);
  jt_check_abs1("adapter/saved keys and shapes", (double)missing, 0.0, 0.5);
  jt_check_abs1("adapter/saved factors are F32", (double)nonf32, 0.0, 0.5);

  st = st_open(ADAPTER_DIR "/saved/readout.safetensors");
  const StTensor *rt = st_count(st) == 1 ? st_tensor(st, 0) : NULL;
  jt_check_abs1("adapter/saved readout is one tensor", (double)st_count(st), 1.0, 0.5);
  jt_check_abs1("adapter/saved readout is F32", (double)(rt != NULL && rt->dtype == ST_DTYPE_F32), 1.0, 0.5);
  jt_check_abs1("adapter/saved readout shape",
                (double)(rt != NULL && rt->ndim == 2 && rt->shape[0] == MT_READOUT_OUT && rt->shape[1] == MT_READOUT_IN),
                1.0, 0.5);
  st_close(st);
  jt_check_abs1("adapter/config format", (double)file_contains(ADAPTER_DIR "/saved/adapter_config.json",
                                                              "\"format\": \"anemll-jeff-lora-1\""),
                1.0, 0.5);
  jt_check_abs1("adapter/config step", (double)file_contains(ADAPTER_DIR "/saved/adapter_config.json", "\"step\": 7"),
                1.0, 0.5);
  jt_check_abs1("adapter/config layers", (double)file_contains(ADAPTER_DIR "/saved/adapter_config.json", "\"layers\": 150"),
                1.0, 0.5);

  CK(cudaMemset(l->param, 0, N * sizeof(float)));
  lora_load_adapter(l, ADAPTER_DIR "/saved");
  JT_SYNC_DEBUG;
  dev_download(l->param, got, N);
  CHECK_REL("adapter/save then load", got, expect, N, 1e-6);

  // 3. A bf16 readout (a checkpoint's own readout.safetensors) is converted, not misread.
  write_adapter_dir(l, ADAPTER_DIR, expect, 1);
  lora_load_adapter(l, ADAPTER_DIR);
  JT_SYNC_DEBUG;
  dev_download(l->param, got, N);
  CHECK_REL("adapter/load bf16 readout", got, expect_bf16, N, 1e-6);

  free(expect);
  free(expect_bf16);
  free(got);
}

int k_adam(void) {
  case_sumsq(1023, 11u, 1e-6);  // fewer elements than blocks: most partials are exactly 0
  case_sumsq(4099, 12u, 1e-6);  // ragged: 4099 = 4 * 1024 + 3
  case_sumsq(5003, 13u, 1e-6);  // prime: no block gets a different count by luck
  case_sumsq(10484736, 14u, JT_TOL_FWD); // the real flat buffer; 10M terms is the plan's forward gate

  case_clip(4099, 1.0f, 21u);   // norm 37 > 1: the coefficient clips
  case_clip(4099, 1e-4f, 22u);  // norm 0.0037 < 1: the coefficient is 1 and nothing changes
  case_clip(10484736, 1.0f, 23u);

  case_lr_at();

  case_adam(257, 200, 10, 31u, NULL); // a small ragged split, fresh buffers
  Lora *l = lora_alloc();             // prints "trainable 10,484,736" (plan M2.6's Done-when)
  case_lora_table(l);
  case_adam((int)l->n_param, (int)l->readout_off, 10, 32u, l);
  case_adapter(l);
  lora_free(l);
  return jt_exit_code();
}
