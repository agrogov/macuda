// src/main.c — jefftrain's driver: the flags, the model load, the training loop, the dev check, the eval and the outputs
// (plan M5.2, design §2.2, §2.3, §9.2, §13).
//
// One optimizer step is plan M5.2's list, in its order: zero the flat gradient buffer, run the accumulation group's
// batches (upload, forward sweep keeping each layer's input, head, backward sweep), clip the global norm to 1, apply the
// schedule and AdamW, advance the step. The log line, the dev check, the best-adapter save and reload, the `--eval`
// scoring and report.json's keys follow CLM-8B's tools/jeff_lora/train.py line by line (design §1.1: train.py is the
// spec), so `tools/jeff_lora/cuda/timing.sh` parses this log as is and CLM-8B's scoring tools read the outputs unchanged:
//
//     step {step}/{total}  {elapsed:.2f}s  loss {mean CE:.4f}  {rows/s:.1f} rows/s  peak {GB:.2f} GB  lr {lr:.2e}
//       dev @ step {step}: soft CE {ce:.4f}  agree {agree:.3f}[  *saved]  ({s}s)
//   eval {name}: soft CE {ce:.4f}  agree {agree:.3f}  ({s}s)
//
// Outputs (design §9.2), all under --output: adapter/ (adapter_model.safetensors, readout.safetensors,
// adapter_config.json — written by src/lora.c), report.json (through src/json.c) and pred-<name>.jsonl per --eval split.
// The evals score the best dev check's adapter, not the last step's: the save happens at the check, the reload after the
// last step.
//
// Also here, unchanged from plan M4.3/M5.1: --dump-forward OUT and --dump-backward OUT write py/dump_ref.py's keys for the
// .jtb's batch 0 (the M4.3/M5.1 gates and the card checkpoints M4.4/M5.4). Those two flags still run alone.
//
// --trace-loss FILE is plan M6.2's per-row trace: one JSON line per optimizer step, {"step": k, "row": [...], "ce":
// [...]}, the rows in batch order (the accumulation group first, then each batch's rows in the file's order) with each
// row's own unweighted soft CE, so a comparison against train.py's traced `per` values lines up step for step.
//
// Memory (design §6.2): one JtCtx sized for the largest packed batch of any of the run's splits, plus 25 activation slabs
// [N,1024] (the forward's saved layer inputs), the sweep's one gradient slab, the logits and the head's device-side
// inputs. Nothing is allocated inside the step loop.
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "adam.h" // kernels/adam.h: the gradient norm, the clip and the Adam step
#include "gemm.h"
#include "json.h"
#include "jtb.h"
#include "layer.h"
#include "lora.h"
#include "model.h"
#include "st.h"
#include "util.h"

// train.py's AdamW settings (design §2.2: betas, eps; weight decay 0, so it is plain Adam) and its clip norm.
#define JT_BETA1 0.9
#define JT_BETA2 0.999
#define JT_ADAM_EPS 1e-8
#define JT_CLIP_NORM 1.0
#define JT_EPS_CLIP 1e-6f // train.py's clip_coef = max_norm / (total_norm + 1e-6)

#define JT_MAX_EVALS 8
#define JT_PATH_MAX 4096
#define JT_TAG_MAX 24

typedef struct {
  const char *model;    // --model DIR: the checkpoint directory (or a model.safetensors file)
  const char *adapter;  // --init-adapter DIR: an adapter to load over the checkpoint's readout
  const char *train;    // --train FILE.jtb: the epoch-ordered train split (design §2.3)
  const char *dev;      // --dev FILE.jtb: the selection split, soft CE at T=1
  const char *output;   // --output DIR: adapter/, report.json, pred-<name>.jsonl
  const char *dump_fwd; // --dump-forward OUT: write the whole forward there (plan M4.3)
  const char *dump_bwd; // --dump-backward OUT: write the forward and every gradient there (plan M5.1 item 4)
  const char *trace;    // --trace-loss FILE: one JSON line per step with its per-row CE (plan M6.2)
  const char *eval_name[JT_MAX_EVALS];
  const char *eval_path[JT_MAX_EVALS];
  int n_eval;
  const char *temperature; // --eval-temperature: a number, or "checkpoint" (decision_config.json)
  const char *targets;     // --targets soft|hard: what prep.py baked into the targets (recorded only)
  const char *weights;     // --weights none|balanced: what prep.py baked into the row weights (recorded only)
  double lr, readout_lr;   // design §2.2's two lr groups
  double weight_cap;
  int epochs, batch, accum, eval_every, max_steps, log_every, eval_only;
} Args;

// src/model.c's design §6.2 backward driver. Declared here, not in src/model.h (another task's frozen file): the dumps
// and the training step are its two callers. `post_mixer` and `grad_out` are the optional dump buffers.
int model_backward(JtCtx *c, const float *saved_in, int n, const JtBatch *b, float *post_mixer, float *grad_out,
                   float *dres);

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char *fmt, ...) {
  va_list ap;
  fflush(stdout);
  fputs("jefftrain: ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

static void usage(FILE *f) {
  fprintf(f,
          "usage: jefftrain --model DIR --train FILE.jtb --dev FILE.jtb --output DIR [options]\n"
          "       jefftrain --model DIR --eval-only --eval NAME=FILE.jtb --output DIR [options]\n"
          "       jefftrain --model DIR --train FILE.jtb --dump-forward|--dump-backward OUT\n"
          "\n"
          "  --model DIR             the jeff-base checkpoint directory (model.safetensors + readout.safetensors)\n"
          "  --train FILE.jtb        the train split: py/prep.py's epoch-ordered batches (design §2.3, §9.1)\n"
          "  --dev FILE.jtb          the selection split: soft CE against the teacher at T=1\n"
          "  --eval NAME=FILE.jtb    a split scored once at the end, with pred-NAME.jsonl written (repeatable)\n"
          "  --output DIR            adapter/, report.json and pred-<name>.jsonl go here (new or empty)\n"
          "  --lr F                  LoRA learning rate (default 2e-4)\n"
          "  --readout-lr F          the readout's learning rate (default 5e-6)\n"
          "  --epochs N              epochs (default 1; must match the .jtb's batch list)\n"
          "  --batch N               rows per batch (default 8; must match the .jtb's batches)\n"
          "  --accum N               gradient accumulation: batches per optimizer step (default 1)\n"
          "  --eval-every N          dev checks every N steps (0, the default: at each epoch's end)\n"
          "  --max-steps N           stop after N optimizer steps (0, the default: the whole schedule)\n"
          "  --eval-temperature T    temperature for the written probabilities: a number, or 'checkpoint'\n"
          "  --eval-only             score the --eval splits, do not train\n"
          "  --init-adapter DIR      load a Python-written adapter over the checkpoint's readout\n"
          "  --targets soft|hard     recorded in report.json/adapter_config.json; default soft\n"
          "  --weights none|balanced   recorded likewise; default none\n"
          "  --weight-cap F          recorded likewise (default 4.0)\n"
          "  --log-every N           print the step line every N steps (default: train.py's rule — steps 1-3, then\n"
          "                          every 25th; --log-every 1 prints every step, which plan M6 needs)\n"
          "  --trace-loss FILE       each step's per-row soft CE as JSON lines (plan M6.2)\n"
          "  --dump-forward OUT      one batch's forward in py/dump_ref.py's format, then exit (plan M4.3)\n"
          "  --dump-backward OUT     one batch's forward plus every gradient, then exit (plan M5.1 item 4)\n"
          "  (no arguments)          print this and exit 0\n");
}

// ---- arguments --------------------------------------------------------------------------------------------------------
static double num_arg(const char *opt, const char *s, double lo, double hi) {
  char *end = NULL;
  errno = 0;
  const double v = strtod(s, &end);
  if (end == s || *end != '\0' || errno != 0 || !(v >= lo) || !(v <= hi))
    die("%s wants a number in [%g, %g], got '%s'", opt, lo, hi, s);
  return v;
}

static int int_arg(const char *opt, const char *s, int lo, int hi) {
  char *end = NULL;
  errno = 0;
  const long v = strtol(s, &end, 10);
  if (end == s || *end != '\0' || errno != 0 || v < lo || v > hi)
    die("%s wants an integer in [%d, %d], got '%s'", opt, lo, hi, s);
  return (int)v;
}

// NAME=PATH, the --eval spelling train.py uses.
static void eval_arg(Args *a, const char *spec) {
  if (a->n_eval >= JT_MAX_EVALS) die("at most %d --eval splits", JT_MAX_EVALS);
  const char *eq = strchr(spec, '=');
  if (eq == NULL || eq == spec || eq[1] == '\0') die("--eval wants NAME=PATH, got '%s'", spec);
  const size_t n = (size_t)(eq - spec);
  char *name = (char *)malloc(n + 1);
  if (name == NULL) die("out of memory (an --eval name)");
  memcpy(name, spec, n);
  name[n] = '\0';
  a->eval_name[a->n_eval] = name;
  a->eval_path[a->n_eval] = eq + 1;
  a->n_eval++;
}

static void parse_args(int argc, char **argv, Args *a) {
  memset(a, 0, sizeof *a);
  a->lr = 2e-4;
  a->readout_lr = 5e-6;
  a->weight_cap = 4.0;
  a->epochs = 1;
  a->batch = 8;
  a->accum = 1;
  a->log_every = -1; // train.py's rule
  a->targets = NULL;
  a->weights = NULL;
  a->temperature = "1";
  for (int k = 1; k < argc; k++) {
    const char *opt = argv[k];
    const int has = k + 1 < argc;
    const char *val = has ? argv[k + 1] : NULL;
    if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0) {
      usage(stdout);
      exit(0);
    } else if (strcmp(opt, "--model") == 0 && has) {
      a->model = val, k++;
    } else if (strcmp(opt, "--init-adapter") == 0 && has) {
      a->adapter = val, k++;
    } else if (strcmp(opt, "--train") == 0 && has) {
      a->train = val, k++;
    } else if (strcmp(opt, "--dev") == 0 && has) {
      a->dev = val, k++;
    } else if (strcmp(opt, "--output") == 0 && has) {
      a->output = val, k++;
    } else if (strcmp(opt, "--eval") == 0 && has) {
      eval_arg(a, val), k++;
    } else if (strcmp(opt, "--dump-forward") == 0 && has) {
      a->dump_fwd = val, k++;
    } else if (strcmp(opt, "--dump-backward") == 0 && has) {
      a->dump_bwd = val, k++;
    } else if (strcmp(opt, "--trace-loss") == 0 && has) {
      a->trace = val, k++;
    } else if (strcmp(opt, "--eval-temperature") == 0 && has) {
      a->temperature = val, k++;
    } else if (strcmp(opt, "--targets") == 0 && has) {
      if (strcmp(val, "soft") != 0 && strcmp(val, "hard") != 0) die("--targets wants soft or hard, got '%s'", val);
      a->targets = val, k++;
    } else if (strcmp(opt, "--weights") == 0 && has) {
      if (strcmp(val, "none") != 0 && strcmp(val, "balanced") != 0)
        die("--weights wants none or balanced, got '%s'", val);
      a->weights = val, k++;
    } else if (strcmp(opt, "--lr") == 0 && has) {
      a->lr = num_arg(opt, val, 0.0, 1.0), k++;
    } else if (strcmp(opt, "--readout-lr") == 0 && has) {
      a->readout_lr = num_arg(opt, val, 0.0, 1.0), k++;
    } else if (strcmp(opt, "--weight-cap") == 0 && has) {
      a->weight_cap = num_arg(opt, val, 0.0, 1e6), k++;
    } else if (strcmp(opt, "--epochs") == 0 && has) {
      a->epochs = int_arg(opt, val, 1, 100000), k++;
    } else if (strcmp(opt, "--batch") == 0 && has) {
      a->batch = int_arg(opt, val, 1, JT_MAX_ROWS), k++;
    } else if (strcmp(opt, "--accum") == 0 && has) {
      a->accum = int_arg(opt, val, 1, 100000), k++;
    } else if (strcmp(opt, "--eval-every") == 0 && has) {
      a->eval_every = int_arg(opt, val, 0, 1000000), k++;
    } else if (strcmp(opt, "--max-steps") == 0 && has) {
      a->max_steps = int_arg(opt, val, 0, 1000000), k++;
    } else if (strcmp(opt, "--log-every") == 0 && has) {
      a->log_every = int_arg(opt, val, 1, 1000000), k++;
    } else if (strcmp(opt, "--eval-only") == 0) {
      a->eval_only = 1;
    } else {
      die("unknown or incomplete option '%s' (--help for the usage)", opt);
    }
  }
}

// ---- device and host buffers -------------------------------------------------------------------------------------------
static float *dev_alloc(size_t n_floats) {
  float *p = NULL;
  CK(cudaMalloc((void **)&p, (n_floats ? n_floats : 1) * sizeof(float)));
  return p;
}

static void dev_free(float **p) {
  if (p != NULL && *p != NULL) CK(cudaFree(*p));
  if (p != NULL) *p = NULL;
}

static float *host_alloc(size_t n_floats) {
  float *p = (float *)malloc((n_floats ? n_floats : 1) * sizeof(float));
  if (p == NULL) die("out of host memory (%zu floats)", n_floats);
  return p;
}

// Copy `n` floats back to a fresh host array, ordered on the context's stream (a non-blocking one, design §12: a plain
// cudaMemcpy on the null stream would race the kernels).
static float *dev_pull(JtCtx *c, const float *dev, size_t n) {
  float *h = host_alloc(n);
  CK(cudaMemcpyAsync(h, dev, n * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
  CK(cudaStreamSynchronize(c->stream));
  return h;
}

// train.py's mem_gb(): what the process holds on the device now, in GiB. The step loop keeps the maximum.
static double dev_mem_gb(void) {
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));
  return (double)(total_b - free_b) / (double)(1u << 30);
}

// mkdir -p (lora.c's own helper is static; the output directory must exist before report.json is written).
static void mkdirs(const char *path) {
  char buf[JT_PATH_MAX];
  const size_t n = path != NULL ? strlen(path) : 0;
  if (n == 0 || n >= sizeof buf) die("bad output directory '%s'", path != NULL ? path : "(null)");
  memcpy(buf, path, n + 1);
  for (size_t i = 1; i <= n; i++) {
    if (buf[i] != '/' && buf[i] != '\0') continue;
    const char saved = buf[i];
    buf[i] = '\0';
    if (mkdir(buf, 0777) != 0 && errno != EEXIST) die("mkdir %s: %s", buf, strerror(errno));
    buf[i] = saved;
  }
}

// "<dir>/<name>"; a truncated path is a bug, not something to carry on from.
static const char *path_join(char *buf, size_t cap, const char *dir, const char *name) {
  const size_t dl = dir != NULL ? strlen(dir) : 0;
  const char *slash = (dl > 0 && dir[dl - 1] == '/') ? "" : "/";
  if (snprintf(buf, cap, "%s%s%s", dir != NULL ? dir : "", slash, name) >= (int)cap)
    die("path too long: %s/%s", dir != NULL ? dir : "", name);
  return buf;
}

// train.py rejects a used --output for a training run; an empty or absent one is fine.
static void check_output_dir(const char *dir, int must_be_empty) {
  DIR *d = opendir(dir);
  if (d == NULL) {
    if (errno == ENOENT) return; // mkdirs/lora_save_adapter creates it
    die("cannot read --output %s: %s", dir, strerror(errno));
  }
  int nonempty = 0;
  struct dirent *e = NULL;
  while ((e = readdir(d)) != NULL) {
    if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
    nonempty = 1;
    break;
  }
  closedir(d);
  if (nonempty && must_be_empty)
    die("--output %s is not empty; use a new or empty directory (train.py's rule)", dir);
}

// The checkpoint directory a --model path names: the path itself when it is a directory, else its parent.
static const char *model_dir_of(const char *path, char *buf, size_t cap) {
  struct stat st;
  if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
    snprintf(buf, cap, "%s", path);
    return buf;
  }
  const char *slash = strrchr(path, '/');
  if (slash == NULL) snprintf(buf, cap, ".");
  else if (slash == path) snprintf(buf, cap, "/");
  else snprintf(buf, cap, "%.*s", (int)(slash - path), path);
  return buf;
}

// ---- the .jtb splits (design §9.1, §2.3) ------------------------------------------------------------------------------
// One batch of a split: the rows' real token ids packed back to back (design §7), each row's live option count and weight,
// and the targets zero-padded past the live count (train.py's collate pads the same way).
typedef struct {
  int rows, n, n_opts;
  int row[JT_MAX_ROWS];     // the file's row indices
  int off[JT_MAX_ROWS + 1]; // prefix sums of the lengths
  int live[JT_MAX_ROWS];    // each row's live option count
  float wrow[JT_MAX_ROWS];  // each row's weight
  float wsum;               // the batch's summed weight
  int *ids;                 // host [cap_ids]
  int cap_ids;
  float *targets; // host [JT_MAX_ROWS * JTB_MAX_OPTS], row-major [rows][n_opts]
} Batch;

typedef struct {
  const char *name; // "train", "dev", or the --eval name
  const char *path;
  Jtb *j;
  char *sidecar;   // the `<split>.prep.json` text py/prep.py wrote next to it, or NULL
  int batch;       // the rows per batch the file must have been written with
  int epochs;      // 1 for dev and eval
  char targets[16]; // --targets/--weights as this split's sidecar records them ("" when it does not)
  char weights[16];
  uint32_t n_rows, n_batches, per_epoch;
  Batch b;
} Split;

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// "<split>.prep.json" beside "<split>.jtb" (py/prep.py, plan M5.3). NULL when there is none, which is what the synthetic
// fixtures have: the structural check below is then the only one.
static char *sidecar_of(const char *path) {
  static const char suffix[] = ".jtb";
  static const char wanted[] = ".prep.json";
  const size_t n = strlen(path);
  char buf[JT_PATH_MAX];
  if (n < sizeof suffix - 1 || strcmp(path + n - 4, suffix) != 0) return NULL;
  if (n - 4 + sizeof wanted > sizeof buf) return NULL;
  memcpy(buf, path, n - 4);
  memcpy(buf + n - 4, wanted, sizeof wanted);
  return json_file(buf);
}

static const char *skip_ws(const char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  return p;
}

// The batch list the file holds, as py/prep.py records it: `<split>.prep.json`'s `rows_per_batch`, one row count per
// batch, every epoch's list in file order (prep.py:193). The check cannot predict that list — train.py's batches()
// length-buckets, cuts and then shuffles the batch order (design §2.3), so the one short batch a split may have lands
// anywhere — so it is read from the sidecar instead. Returns a malloc'd array and stores its length in *n; NULL when
// the sidecar is absent (the synthetic fixtures) or does not record the list.
static uint32_t *sidecar_rows_per_batch(const Split *s, uint32_t *n) {
  *n = 0;
  if (s->sidecar == NULL) return NULL;
  const char *p = strstr(s->sidecar, "\"rows_per_batch\"");
  if (p == NULL) return NULL;
  p = skip_ws(p + sizeof "\"rows_per_batch\"" - 1);
  if (*p != ':') return NULL;
  p = skip_ws(p + 1);
  if (*p != '[') return NULL;
  size_t cap = 64, len = 0;
  uint32_t *out = (uint32_t *)malloc(cap * sizeof *out);
  if (out == NULL) die("out of host memory (a split's batch list)");
  int bad = 0;
  for (p = skip_ws(p + 1); *p != ']';) {
    char *end = NULL;
    errno = 0;
    const long v = strtol(p, &end, 10);
    if (end == p || errno != 0 || v < 0) {
      bad = 1;
      break;
    }
    if (len == cap) {
      cap *= 2;
      uint32_t *bigger = (uint32_t *)realloc(out, cap * sizeof *out);
      if (bigger == NULL) die("out of host memory (a split's batch list)");
      out = bigger;
    }
    out[len++] = (uint32_t)v;
    p = skip_ws(end);
    if (*p == ',') p = skip_ws(p + 1);
    else if (*p != ']') {
      bad = 1;
      break;
    }
  }
  if (bad) {
    free(out); // not the integer list prep.py writes; fall back to the count-and-sum check below
    return NULL;
  }
  *n = (uint32_t)len;
  return out;
}

// Reject a flag combination the .jtb disagrees with (plan M5.2). --batch and --epochs decide how many batches the file
// must have and how many rows each holds, and both are in its header's n_rows/n_batches and its batch table.
//
// Which rows each batch holds is prep.py's call, not this check's (design §2.3): the split's row counts are whatever its
// sidecar's `rows_per_batch` list says, compared batch by batch. Without a sidecar the list is unknown, so every batch
// must then hold 1..--batch rows and each epoch's batch sizes must add up to the split's rows.
static void split_check_structure(Split *s) {
  s->n_rows = jtb_n_rows(s->j);
  s->n_batches = jtb_n_batches(s->j);
  if (s->n_rows < 1) die("%s: the split has no rows", s->path);
  if (s->n_batches < 1) die("%s: the split has no batches", s->path);
  s->per_epoch = ceil_div(s->n_rows, (uint32_t)s->batch);
  const uint32_t want_batches = s->per_epoch * (uint32_t)s->epochs;
  if (s->n_batches != want_batches)
    die("%s: %u row(s) in %u batch(es) is not what --batch %d --epochs %d implies (%u batches); the file was written "
        "with a different --batch/--epochs",
        s->path, s->n_rows, s->n_batches, s->batch, s->epochs, want_batches);

  uint32_t listed = 0;
  uint32_t *per_batch = sidecar_rows_per_batch(s, &listed);
  if (per_batch != NULL && listed != s->n_batches)
    die("%s: its .prep.json lists %u row count(s) per batch, but the file has %u batches; the sidecar is not this "
        "file's",
        s->path, listed, s->n_batches);

  for (uint32_t e = 0; e < (uint32_t)s->epochs; e++) {
    uint64_t epoch_rows = 0;
    for (uint32_t i = 0; i < s->per_epoch; i++) {
      const uint32_t k = e * s->per_epoch + i;
      uint32_t n = 0;
      const uint32_t *rows = jtb_batch(s->j, k, &n);
      if (n > (uint32_t)JT_MAX_ROWS)
        die("%s: batch %u has %u rows, at most %d are supported", s->path, k, n, JT_MAX_ROWS);
      if (per_batch != NULL) {
        if (per_batch[k] < 1 || per_batch[k] > (uint32_t)s->batch)
          die("%s: its .prep.json says batch %u holds %u row(s), but --batch %d allows 1..%d; the sidecar was written "
              "with a different --batch",
              s->path, k, per_batch[k], s->batch, s->batch);
        if (n != per_batch[k])
          die("%s: batch %u holds %u row(s), but its .prep.json says %u; the sidecar and the file's batch table "
              "disagree",
              s->path, k, n, per_batch[k]);
      } else if (n < 1 || n > (uint32_t)s->batch) {
        die("%s: batch %u holds %u row(s), but --batch %d --epochs %d allows 1..%d; the file was written with a "
            "different --batch",
            s->path, k, n, s->batch, s->epochs, s->batch);
      }
      epoch_rows += n;
      for (uint32_t r = 0; r < n; r++)
        if (rows[r] >= s->n_rows) die("%s: batch %u names row %u, which the file does not have", s->path, k, rows[r]);
    }
    if (epoch_rows != (uint64_t)s->n_rows)
      die("%s: epoch %u's batches hold %llu row(s) in all, but the split has %u; the file was written with a different "
          "--batch",
          s->path, e, (unsigned long long)epoch_rows, s->n_rows);
  }
  free(per_batch);
}

// The sidecar's own record of the flags, when py/prep.py wrote one: a mismatch is rejected the same way, and --targets/
// --weights (which only prep.py acts on) are adopted from it when this run did not name them.
static void split_check_sidecar(Split *s, Args *a) {
  if (s->sidecar == NULL) return;
  long long v = 0;
  if (json_int(s->sidecar, "batch", &v) && v != a->batch)
    die("%s: its .prep.json says --batch %lld, this run has --batch %d", s->path, v, a->batch);
  if (json_int(s->sidecar, "accum", &v) && v != a->accum)
    die("%s: its .prep.json says --accum %lld, this run has --accum %d", s->path, v, a->accum);
  if (json_int(s->sidecar, "epochs", &v) && v != s->epochs)
    die("%s: its .prep.json says %lld epoch(s), this run has %d", s->path, v, s->epochs);
  if (json_int(s->sidecar, "eval_batch", &v) && v != s->batch)
    die("%s: its .prep.json says eval_batch %lld, but --batch %d makes this split's batches %d rows", s->path, v,
        a->batch, s->batch);
  if (json_str(s->sidecar, "targets", s->targets, sizeof s->targets)) {
    const char *v2 = s->targets[0] == 'h' ? "hard" : "soft";
    if (a->targets == NULL) a->targets = v2;
    else if (strcmp(v2, a->targets) != 0)
      die("%s: its .prep.json says targets %s, this run has --targets %s", s->path, v2, a->targets);
  }
  if (json_str(s->sidecar, "weights", s->weights, sizeof s->weights)) {
    const char *v2 = s->weights[0] == 'b' ? "balanced" : "none";
    if (a->weights == NULL) a->weights = v2;
    else if (strcmp(v2, a->weights) != 0)
      die("%s: its .prep.json says weights %s, this run has --weights %s", s->path, v2, a->weights);
  }
  double d = 0;
  if (json_double(s->sidecar, "weight_cap", &d) && fabs(d - a->weight_cap) > 1e-9)
    die("%s: its .prep.json says weight_cap %g, this run has --weight-cap %g", s->path, d, a->weight_cap);
}

static void split_open(Split *s, const char *name, const char *path, int batch, int epochs) {
  if (batch < 1 || batch > JT_MAX_ROWS) die("%s: %d rows per batch, want 1..%d", path, batch, JT_MAX_ROWS);
  memset(s, 0, sizeof *s);
  s->name = name;
  s->path = path;
  s->batch = batch;
  s->epochs = epochs;
  s->j = jtb_open(path);
  s->sidecar = sidecar_of(path);
  split_check_structure(s);
  s->b.targets = (float *)malloc((size_t)JT_MAX_ROWS * JTB_MAX_OPTS * sizeof *s->b.targets);
  if (s->b.targets == NULL) die("out of host memory (a split's target buffer)");
  s->b.cap_ids = 0;
  s->b.ids = NULL;
}

static void split_close(Split *s) {
  free(s->b.targets);
  free(s->b.ids);
  free(s->sidecar);
  jtb_close(s->j);
  memset(s, 0, sizeof *s);
}

// Fill s->b from batch k of the split's file.
static void batch_load(Split *s, uint32_t k) {
  Batch *b = &s->b;
  uint32_t nb = 0;
  const uint32_t *rows = jtb_batch(s->j, k, &nb);
  if (nb < 1 || nb > (uint32_t)JT_MAX_ROWS) die("%s: batch %u has %u row(s)", s->path, k, nb);
  b->rows = (int)nb;
  b->n = 0;
  b->n_opts = 0;
  b->wsum = 0.0f;
  for (int r = 0; r < b->rows; r++) {
    const JtbRow *row = jtb_row(s->j, rows[r]);
    if (row == NULL) die("%s: batch %u names row %u, which the file does not have", s->path, k, rows[r]);
    if (row->len < 1) die("%s: row %u is empty", s->path, rows[r]);
    b->row[r] = (int)rows[r];
    b->off[r] = b->n;
    b->n += (int)row->len;
    b->live[r] = (int)row->n_opts;
    b->wrow[r] = row->weight;
    b->wsum += row->weight;
    if ((int)row->n_opts > b->n_opts) b->n_opts = (int)row->n_opts;
  }
  b->off[b->rows] = b->n;
  if (b->n_opts < 1 || b->n_opts > (int)JTB_MAX_OPTS)
    die("%s: batch %u's widest row has %d option(s), want 1..%u", s->path, k, b->n_opts, JTB_MAX_OPTS);
  if (b->n > b->cap_ids) {
    int *bigger = (int *)realloc(b->ids, (size_t)b->n * sizeof *bigger);
    if (bigger == NULL) die("out of host memory (%d token ids)", b->n);
    b->ids = bigger;
    b->cap_ids = b->n;
  }
  memset(b->targets, 0, (size_t)b->rows * (size_t)b->n_opts * sizeof *b->targets);
  for (int r = 0; r < b->rows; r++) {
    const JtbRow *row = jtb_row(s->j, rows[r]);
    memcpy(b->targets + (size_t)r * (size_t)b->n_opts, jtb_targets(s->j, rows[r]),
           (size_t)row->n_opts * sizeof *b->targets);
    const uint32_t *ids = jtb_ids(s->j, rows[r]);
    for (int t = 0; t < (int)row->len; t++) b->ids[b->off[r] + t] = (int)ids[t];
  }
}

// The weight sum of one batch without loading it (a step needs the group's total before its first batch runs).
static float batch_wsum(const Jtb *j, uint32_t k) {
  uint32_t nb = 0;
  const uint32_t *rows = jtb_batch(j, k, &nb);
  float sum = 0.0f;
  for (uint32_t i = 0; i < nb; i++) {
    const JtbRow *row = jtb_row(j, rows[i]);
    if (row == NULL) die("a batch names a row the file does not have");
    sum += row->weight;
  }
  return sum;
}

// ---- the device buffers one step (or one eval pass) needs --------------------------------------------------------------
typedef struct {
  int cap_tokens, cap_rows;
  int *ids;        // [cap_tokens]
  float *targets;  // [cap_rows * JTB_MAX_OPTS]
  float *wrow;     // [cap_rows]
  float *logits;   // [cap_rows * JTB_MAX_OPTS]
  float *saved;    // [MT_LAYERS + 1][cap_tokens][MT_HIDDEN]: the forward's saved layer inputs (design §6.2)
  float *dres;     // [cap_tokens][MT_HIDDEN]: the sweep's one gradient buffer
  float *partials; // [K_ADAM_BLOCKS]
  float *sumsq;    // [1]
  float *logits_h; // host [cap_rows * JTB_MAX_OPTS]
  double *row_ce;  // host [cap_rows]
} Dev;

static void dev_init(Dev *d, int cap_tokens, int cap_rows) {
  memset(d, 0, sizeof *d);
  d->cap_tokens = cap_tokens;
  d->cap_rows = cap_rows;
  CK(cudaMalloc((void **)&d->ids, (size_t)cap_tokens * sizeof *d->ids));
  d->targets = dev_alloc((size_t)cap_rows * JTB_MAX_OPTS);
  d->wrow = dev_alloc((size_t)cap_rows);
  d->logits = dev_alloc((size_t)cap_rows * JTB_MAX_OPTS);
  d->saved = dev_alloc((size_t)(MT_LAYERS + 1) * (size_t)cap_tokens * MT_HIDDEN);
  d->dres = dev_alloc((size_t)cap_tokens * MT_HIDDEN);
  d->partials = dev_alloc(K_ADAM_BLOCKS);
  d->sumsq = dev_alloc(1);
  d->logits_h = host_alloc((size_t)cap_rows * JTB_MAX_OPTS);
  d->row_ce = (double *)malloc((size_t)cap_rows * sizeof *d->row_ce);
  if (d->row_ce == NULL) die("out of host memory (the per-row losses)");
}

static void dev_free_all(Dev *d) {
  free(d->row_ce);
  free(d->logits_h);
  dev_free(&d->sumsq);
  dev_free(&d->partials);
  dev_free(&d->dres);
  dev_free(&d->saved);
  dev_free(&d->logits);
  dev_free(&d->wrow);
  dev_free(&d->targets);
  if (d->ids != NULL) CK(cudaFree(d->ids));
  d->ids = NULL;
}

static void dev_upload(JtCtx *c, Dev *d, const Batch *b) {
  CK(cudaMemcpyAsync(d->ids, b->ids, (size_t)b->n * sizeof *b->ids, cudaMemcpyHostToDevice, c->stream));
  CK(cudaMemcpyAsync(d->targets, b->targets, (size_t)b->rows * (size_t)b->n_opts * sizeof *b->targets,
                     cudaMemcpyHostToDevice, c->stream));
  CK(cudaMemcpyAsync(d->wrow, b->wrow, (size_t)b->rows * sizeof *b->wrow, cudaMemcpyHostToDevice, c->stream));
}

// A batch's forward sweep and its head's logits. `with_loss` gives the head the targets and the weights too, so it
// computes train.py's `(per * wt).sum() / gw` and saves what its backward reads (JT_FWD_SAVE).
static void batch_forward(JtCtx *c, Dev *d, const Batch *b, const JtBatch *jb, float wgroup, int with_loss) {
  const size_t stride = (size_t)b->n * MT_HIDDEN;
  dev_upload(c, d, b);
  int rc = model_forward(c, d->ids, b->n, jb, d->saved, NULL);
  if (rc == JT_LAYER_NOT_IMPL) die("the forward is incomplete: a layer is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("model_forward returned %d", rc);

  HeadBatch hb;
  hb.n_opts = b->n_opts;
  hb.live = b->live;
  hb.targets = with_loss ? d->targets : NULL;
  hb.wrow = with_loss ? d->wrow : NULL;
  hb.wgroup = with_loss ? wgroup : 1.0f;
  jt_saves_reset(c); // the head's saves start at the arena's base (src/head.c)
  rc = head_fwd(c, d->saved + (size_t)MT_LAYERS * stride, b->n, jb, &hb, NULL, d->logits, NULL,
                with_loss ? JT_FWD_SAVE : JT_FWD_ONLY);
  if (rc == JT_LAYER_NOT_IMPL) die("the head is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("head_fwd returned %d", rc);
}

// ---- per-row losses and probabilities on the host (train.py's soft_ce / evaluate()) ------------------------------------
// -sum_j t_j log softmax(logits)_j over the row's live options: train.py's `soft_ce`, from the same fp32 logits, in
// double. head_fwd only hands back the weighted batch sum, and both the log line (the running mean of per-row CE) and
// --trace-loss want the per-row values.
static double row_soft_ce(const float *logits, int live, const float *targets) {
  if (live < 1) return 0.0;
  double m = logits[0];
  for (int j = 1; j < live; j++)
    if (logits[j] > m) m = logits[j];
  double s = 0.0;
  for (int j = 0; j < live; j++) s += exp((double)logits[j] - m);
  const double lse = m + log(s);
  double ce = 0.0;
  for (int j = 0; j < live; j++) ce += (double)targets[j] * ((double)logits[j] - lse);
  return -ce;
}

static int row_argmax(const float *logits, int live) {
  int best = 0;
  for (int j = 1; j < live; j++)
    if (logits[j] > logits[best]) best = j;
  return best;
}

// train.py's `torch.softmax(logits / temperature, -1)` over the row's live options; the dead ones are exactly 0.
static void row_softmax(const float *logits, int live, double temperature, double *p) {
  double m = logits[0];
  for (int j = 1; j < live; j++)
    if (logits[j] > m) m = logits[j];
  double s = 0.0;
  for (int j = 0; j < live; j++) {
    p[j] = exp(((double)logits[j] - m) / temperature);
    s += p[j];
  }
  for (int j = 0; j < live; j++) p[j] /= s;
}

// ---- --trace-loss (plan M6.2) ------------------------------------------------------------------------------------------
typedef struct {
  FILE *f;
  const char *path;
} Trace;

static void trace_open(Trace *t, const char *path) {
  memset(t, 0, sizeof *t);
  if (path == NULL) return;
  t->f = fopen(path, "w");
  if (t->f == NULL) die("cannot write --trace-loss %s: %s", path, strerror(errno));
  t->path = path;
}

static void trace_close(Trace *t) {
  if (t->f == NULL) return;
  if (fclose(t->f) != 0) die("cannot finish %s: %s", t->path, strerror(errno));
  t->f = NULL;
}

// One step's record: the whole accumulation group's rows in batch order, each with its own unweighted soft CE.
static void trace_step(Trace *t, int step, const int *rows, const double *ce, int n) {
  if (t->f == NULL) return;
  Json *j = json_open_stream(t->f);
  json_compact(j, 1); // one record per line: a JSON-lines reader (py/compare_steps.py) reads it line by line
  json_begin_object(j, NULL);
  json_key_int(j, "step", step);
  json_begin_array(j, "row");
  for (int r = 0; r < n; r++) json_key_int(j, NULL, rows[r]);
  json_end_array(j);
  json_begin_array(j, "ce");
  for (int r = 0; r < n; r++) json_key_double(j, NULL, ce[r], -1);
  json_end_array(j);
  json_end_object(j);
  json_newline(j);
  json_close(j);
}

// ---- one training batch (plan M5.2) -----------------------------------------------------------------------------------
// Forward sweep keeping each layer's input, the head (logits, loss, the saves its backward reads), the per-row CE for the
// log and the trace, then the backward sweep: head_bwd starts it (dres = dL/d(layer 23's out)) and model_backward walks
// the layers 23..0, recomputing each from its saved input. The parameter gradients accumulate into the flat buffer across
// the whole accumulation group (src/linear.h's rule). Returns the batch's summed per-row CE (train.py's `per.sum()`).
static double step_batch(JtCtx *c, Dev *d, Split *s, uint32_t k, float wgroup) {
  Batch *b = &s->b;
  batch_load(s, k);
  JtBatch jb;
  jt_batch_set(&jb, b->rows, b->n, b->off);
  CK(cudaDeviceSynchronize()); // jt_batch_set's uploads are on the null stream (src/model.h)
  batch_forward(c, d, b, &jb, wgroup, 1);
  const size_t stride = (size_t)b->n * MT_HIDDEN;

  CK(cudaMemcpyAsync(d->logits_h, d->logits, (size_t)b->rows * (size_t)b->n_opts * sizeof *d->logits_h,
                     cudaMemcpyDeviceToHost, c->stream));
  CK(cudaStreamSynchronize(c->stream));
  double sum = 0.0;
  for (int r = 0; r < b->rows; r++) {
    const float *lg = d->logits_h + (size_t)r * (size_t)b->n_opts;
    const float *tg = b->targets + (size_t)r * (size_t)b->n_opts;
    d->row_ce[r] = row_soft_ce(lg, b->live[r], tg);
    sum += d->row_ce[r];
  }

  const float *x = d->saved + (size_t)MT_LAYERS * stride;
  int rc = head_bwd(c, x, b->n, &jb, d->dres);
  if (rc != JT_LAYER_OK) die("head_bwd returned %d", rc);
  rc = model_backward(c, d->saved, b->n, &jb, NULL, NULL, d->dres);
  if (rc == JT_LAYER_NOT_IMPL) die("the backward is incomplete: a layer is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("model_backward returned %d", rc);

  jt_batch_free(&jb);
  return sum;
}

// design §2.2's clip, schedule and AdamW for step `step` (0-based: the optimizer steps already taken).
static void optimizer_step(JtCtx *c, Dev *d, const Args *a, int step, int total) {
  Lora *l = c->lora;
  const int n = (int)l->n_param;
  k_sumsq_partial(l->grad, n, d->partials, K_ADAM_BLOCKS, c->stream);
  k_sumsq_final(d->partials, K_ADAM_BLOCKS, d->sumsq, c->stream);
  float sq = 0.0f;
  CK(cudaMemcpyAsync(&sq, d->sumsq, sizeof sq, cudaMemcpyDeviceToHost, c->stream));
  CK(cudaStreamSynchronize(c->stream));
  // the two kernels give the sum of squares; the norm is its square root (kernels/adam.h, ref/adam.h's ref_clip_coef).
  const float norm = sqrtf(sq);
  const float coef = (float)JT_CLIP_NORM / (norm + JT_EPS_CLIP); // train.py: max_norm / (total_norm + 1e-6)
  if (coef < 1.0f) k_clip_scale(l->grad, n, coef, c->stream);

  const double f = lr_at(step, total, 1.0); // the schedule's multiplier at s = step (design §2.2)
  const float bias1 = (float)(1.0 - pow(JT_BETA1, (double)(step + 1)));
  const float bias2 = (float)(1.0 - pow(JT_BETA2, (double)(step + 1)));
  k_adamw_step(l->param, l->grad, l->m, l->v, n, (int)l->readout_off, (float)(a->lr * f), (float)(a->readout_lr * f),
               (float)JT_BETA1, (float)JT_BETA2, (float)JT_ADAM_EPS, bias1, bias2, c->stream);
}

// ---- evaluation (train.py's evaluate()) -------------------------------------------------------------------------------
typedef struct {
  const char *name;
  long n;             // rows scored
  double ce_sum;      // sum of the per-row soft CE at T=1 (the selection metric)
  long agree;         // rows whose argmax is the teacher's top option
  double *probs;      // [n_rows * JTB_MAX_OPTS], filled when the split's predictions are wanted
  unsigned char *seen; // [n_rows]
  double seconds;
} Eval;

// Score one split: forward only, the head's logits, then train.py's ce/agreement per row. `keep` also keeps the
// temperature-softmax probabilities for pred-<name>.jsonl.
static void eval_split(JtCtx *c, Dev *d, Split *s, double temperature, int keep, Eval *e) {
  const double t0 = jt_now();
  e->n = 0;
  e->ce_sum = 0.0;
  e->agree = 0;
  for (uint32_t k = 0; k < s->n_batches; k++) {
    Batch *b = &s->b;
    batch_load(s, k);
    JtBatch jb;
    jt_batch_set(&jb, b->rows, b->n, b->off);
    CK(cudaDeviceSynchronize());
    batch_forward(c, d, b, &jb, 1.0f, 0);
    CK(cudaMemcpyAsync(d->logits_h, d->logits, (size_t)b->rows * (size_t)b->n_opts * sizeof *d->logits_h,
                       cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    for (int r = 0; r < b->rows; r++) {
      const float *lg = d->logits_h + (size_t)r * (size_t)b->n_opts;
      const float *tg = b->targets + (size_t)r * (size_t)b->n_opts;
      const int live = b->live[r];
      const JtbRow *row = jtb_row(s->j, (uint32_t)b->row[r]);
      e->n++;
      e->ce_sum += row_soft_ce(lg, live, tg);
      if (row_argmax(lg, live) == (int)row->teacher_top) e->agree++;
      if (keep && e->probs != NULL) {
        double *p = e->probs + (size_t)b->row[r] * JTB_MAX_OPTS;
        for (int j = 0; j < (int)JTB_MAX_OPTS; j++) p[j] = 0.0;
        row_softmax(lg, live, temperature, p);
        e->seen[b->row[r]] = 1;
      }
    }
    jt_batch_free(&jb);
  }
  e->seconds = jt_now() - t0;
}

static double eval_ce(const Eval *e) { return e->n > 0 ? e->ce_sum / (double)e->n : 0.0; }
static double eval_agree(const Eval *e) { return e->n > 0 ? (double)e->agree / (double)e->n : 0.0; }

// pred-<name>.jsonl (design §9.2): one record per row, in the split's input order (the rows are stored in input order,
// so the row index is that order), keyed by the question's criteria keys, probabilities rounded to 6 decimals.
static void write_preds(const Split *s, const Eval *e, const char *path) {
  FILE *f = fopen(path, "w");
  if (f == NULL) die("cannot write %s: %s", path, strerror(errno));
  for (uint32_t r = 0; r < s->n_rows; r++) {
    if (e->seen == NULL || !e->seen[r]) continue;
    const char *keys[JTB_MAX_OPTS];
    const int nk = jtb_keys(s->j, r, keys, (int)JTB_MAX_OPTS);
    if (nk <= 0) die("%s: row %u has no criteria keys to key its probabilities by", s->path, r);
    Json *j = json_open_stream(f);
    json_compact(j, 1); // train.py writes one `json.dumps(row)` per line
    json_begin_object(j, NULL);
    json_key_str(j, "id", jtb_id(s->j, r));
    json_begin_object(j, "probs");
    for (int i = 0; i < nk; i++) json_key_double(j, keys[i], e->probs[(size_t)r * JTB_MAX_OPTS + i], 6);
    json_end_object(j);
    json_end_object(j);
    json_newline(j);
    json_close(j);
  }
  if (fclose(f) != 0) die("cannot finish %s: %s", path, strerror(errno));
  printf("jefftrain: wrote %s (%ld row(s))\n", path, e->n);
  fflush(stdout);
}

// ---- the dev checks and the report (train.py's check()) ----------------------------------------------------------------
typedef struct {
  int step;
  long rows_seen;
  char tag[JT_TAG_MAX];
  double dev_soft_ce, dev_agreement, train_loss, seconds, peak_gb;
  int saved;
} HistEntry;

typedef struct {
  HistEntry *e;
  int n, cap;
} History;

static HistEntry *hist_add(History *h) {
  if (h->n == h->cap) {
    const int cap = h->cap ? 2 * h->cap : 16;
    HistEntry *bigger = (HistEntry *)realloc(h->e, (size_t)cap * sizeof *bigger);
    if (bigger == NULL) die("out of host memory (the dev-check history)");
    h->e = bigger;
    h->cap = cap;
  }
  HistEntry *e = &h->e[h->n++];
  memset(e, 0, sizeof *e);
  return e;
}

// Python's round(v, ndigits) for report.json's rounded fields (train.py rounds what it writes). The values here are never
// exactly a half, so rounding away from zero is the same as Python's half-to-even.
static double round_to(double v, int ndigits) {
  double scale = 1.0;
  for (int i = 0; i < ndigits; i++) scale *= 10.0;
  const double scaled = v * scale;
  return (scaled >= 0 ? floor(scaled + 0.5) : ceil(scaled - 0.5)) / scale;
}

// Everything a run accumulates for report.json and its log lines.
typedef struct {
  const Args *a;
  JtCtx *c;
  Dev *d;
  Split *train, *dev;
  Split *ev[JT_MAX_EVALS];
  Eval evres[JT_MAX_EVALS];
  int n_ev;
  History hist;
  Trace trace;
  int *tr_row; // the step's per-row CE record being collected: row indices and values (--trace-loss)
  double *tr_ce;
  int tr_n, tr_cap;
  char adapter_dir[JT_PATH_MAX];
  LoraMeta meta;
  double temperature;
  double peak, best, t_train, run_loss;
  long run_n, seen;
  int step, total, per_epoch, per_epoch_batches, trained;
  double train_seconds, rows_per_second;
} Trainer;

// train.py's log line for step `step` (1-based) after its optimizer step: the running mean of per-row CE since the last
// dev check, the rows/s rate since training began, the peak device memory, and the LoRA group's lr for the NEXT step
// (`sched.get_last_lr()` is read after `sched.step()` in train.py, so it is lambda(step), not lambda(step-1)).
static void trainer_log(const Trainer *t, double elapsed) {
  const double loss = t->run_loss / (t->run_n > 0 ? (double)t->run_n : 1.0);
  const double el = jt_now() - t->t_train;
  const double rps = el > 0.0 ? (double)t->seen / el : 0.0;
  const double lr = t->a->lr * lr_at(t->step, t->total, 1.0);
  printf("  step %d/%d  %.2fs  loss %.4f  %.1f rows/s  peak %.2f GB  lr %.2e\n", t->step, t->total, elapsed, loss, rps,
         t->peak, lr);
  fflush(stdout);
}

static void trainer_check(Trainer *t, const char *tag) {
  const double t0 = jt_now();
  Eval e;
  memset(&e, 0, sizeof e);
  e.name = t->dev->name;
  eval_split(t->c, t->d, t->dev, 1.0, 0, &e); // the selection metric is always the teacher's soft CE at T=1
  const double ce = eval_ce(&e), agree = eval_agree(&e);
  HistEntry *h = hist_add(&t->hist);
  h->step = t->step;
  h->rows_seen = t->seen;
  snprintf(h->tag, sizeof h->tag, "%s", tag);
  h->dev_soft_ce = round_to(ce, 5);
  h->dev_agreement = round_to(agree, 4);
  h->train_loss = round_to(t->run_loss / (t->run_n > 0 ? (double)t->run_n : 1.0), 5);
  h->seconds = round_to(jt_now() - t->t_train, 0);
  h->peak_gb = round_to(t->peak, 2);
  if (ce < t->best) { // train.py saves the best dev check's adapter as it goes
    t->best = ce;
    lora_save_adapter(t->c->lora, t->adapter_dir, t->step, ce, &t->meta);
    h->saved = 1;
  }
  printf("  dev @ step %d: soft CE %.4f  agree %.3f%s  (%.0fs)\n", t->step, ce, agree, h->saved ? "  *saved" : "",
         jt_now() - t0);
  fflush(stdout);
}

// One optimizer step over the `nbatch`-batch accumulation group that starts at batch `first` of the train split
// (plan M5.2).
static void trainer_step(Trainer *t, uint32_t first, int nbatch) {
  const double t0 = jt_now();
  if (nbatch < 1 || first + (uint32_t)nbatch > t->train->n_batches)
    die("%s: the accumulation group at batch %u (%d batch(es)) runs past the file's %u batches", t->train->path, first,
        nbatch, t->train->n_batches);
  lora_zero_grad(t->c->lora, t->c->stream);
  float wgroup = 0.0f;
  for (int i = 0; i < nbatch; i++) wgroup += batch_wsum(t->train->j, first + (uint32_t)i);
  if (!(wgroup > 0.0f)) die("%s: the accumulation group's row weights sum to %g", t->train->path, (double)wgroup);
  t->tr_n = 0;
  for (int i = 0; i < nbatch; i++) {
    t->run_loss += step_batch(t->c, t->d, t->train, first + (uint32_t)i, wgroup);
    const Batch *b = &t->train->b;
    t->run_n += b->rows;
    t->seen += b->rows;
    if (t->trace.f != NULL) { // the trace's rows in batch order: this group's batches, each batch's rows in file order
      if (t->tr_n + b->rows > t->tr_cap) {
        const int cap = t->tr_cap ? 2 * t->tr_cap : 8 * JT_MAX_ROWS;
        int *rows = (int *)realloc(t->tr_row, (size_t)cap * sizeof *rows);
        double *ce = (double *)realloc(t->tr_ce, (size_t)cap * sizeof *ce);
        if (rows == NULL || ce == NULL) die("out of host memory (the --trace-loss record)");
        t->tr_row = rows;
        t->tr_ce = ce;
        t->tr_cap = cap;
      }
      for (int r = 0; r < b->rows; r++) {
        t->tr_row[t->tr_n] = b->row[r];
        t->tr_ce[t->tr_n] = t->d->row_ce[r];
        t->tr_n++;
      }
    }
  }
  trace_step(&t->trace, t->step + 1, t->tr_row, t->tr_ce, t->tr_n);
  optimizer_step(t->c, t->d, t->a, t->step, t->total);
  CK(cudaStreamSynchronize(t->c->stream)); // the log's timing and the peak include the whole step
  t->step++;
  const double m = dev_mem_gb();
  if (m > t->peak) t->peak = m;
  const int every = t->a->log_every;
  if (every > 0 ? (t->step % every == 0) : (t->step <= 3 || t->step % 25 == 0)) trainer_log(t, jt_now() - t0);
}

static void trainer_reset_run(Trainer *t) {
  t->run_loss = 0.0;
  t->run_n = 0;
}

// train.py's main loop: a dev check at step 0, then epoch by epoch and accumulation group by accumulation group, a dev
// check every --eval-every steps (or at each epoch's end), stopping at --max-steps.
static void trainer_run(Trainer *t) {
  trainer_check(t, "start");
  int done = 0;
  for (int epoch = 0; epoch < t->a->epochs && !done; epoch++) {
    const uint32_t base = (uint32_t)epoch * (uint32_t)t->per_epoch_batches;
    for (int gi = 0; gi < t->per_epoch_batches && !done; gi += t->a->accum) {
      const int left = t->per_epoch_batches - gi;
      trainer_step(t, base + (uint32_t)gi, left < t->a->accum ? left : t->a->accum);
      if (t->a->eval_every > 0 && t->step % t->a->eval_every == 0) {
        trainer_check(t, "step");
        trainer_reset_run(t);
      }
      if (t->step >= t->total) done = 1;
    }
    if (t->a->eval_every == 0 || done) {
      const int all = t->per_epoch * t->a->epochs;
      char tag[JT_TAG_MAX];
      snprintf(tag, sizeof tag, "epoch %d", epoch + 1);
      trainer_check(t, (!done || t->step == all) ? tag : "max-steps");
      trainer_reset_run(t);
    }
  }
}

// ---- report.json (train.py's payload, design §9.2) --------------------------------------------------------------------
static void report_args(Json *j, const Args *a, double temperature) {
  json_begin_object(j, "args");
  json_key_str(j, "model", a->model);
  json_key_str(j, "train", a->train);
  json_key_str(j, "dev", a->dev);
  json_begin_array(j, "eval");
  for (int i = 0; i < a->n_eval; i++) {
    char spec[JT_PATH_MAX];
    snprintf(spec, sizeof spec, "%s=%s", a->eval_name[i], a->eval_path[i]);
    json_key_str(j, NULL, spec);
  }
  json_end_array(j);
  json_key_str(j, "output", a->output);
  json_key_str(j, "targets", a->targets != NULL ? a->targets : "soft");
  json_key_str(j, "weights", a->weights != NULL ? a->weights : "none");
  json_key_double(j, "weight-cap", a->weight_cap, -1);
  json_key_int(j, "rank", LORA_RANK);
  json_key_double(j, "alpha", (double)MT_LORA_SCALE * LORA_RANK, -1);
  json_key_double(j, "lr", a->lr, -1);
  json_key_double(j, "readout-lr", a->readout_lr, -1);
  json_key_int(j, "epochs", a->epochs);
  json_key_int(j, "batch", a->batch);
  json_key_int(j, "accum", a->accum);
  json_key_int(j, "eval-every", a->eval_every);
  json_key_int(j, "max-steps", a->max_steps);
  json_key_double(j, "eval-temperature", temperature, -1);
  json_key_bool(j, "eval-only", a->eval_only);
  json_key_str(j, "init-adapter", a->adapter);
  json_key_str(j, "device", "cuda");
  json_key_str(j, "dtype", "fp32");
  json_key_int(j, "log-every", a->log_every);
  json_key_str(j, "trace-loss", a->trace);
  json_end_object(j);
}

static void report_write(const Trainer *t) {
  char path[JT_PATH_MAX];
  path_join(path, sizeof path, t->a->output, "report.json");
  Json *j = json_open(path);
  json_begin_object(j, NULL);
  report_args(j, t->a, t->temperature);
  json_key_str(j, "model", t->a->model);
  json_key_str(j, "device", "cuda");
  json_key_double(j, "eval_temperature", t->temperature, -1);
  json_key_null(j, "row_weights"); // prep.py owns the weights (design §9.1); jefftrain only consumes them
  json_begin_array(j, "history");
  for (int i = 0; i < t->hist.n; i++) {
    const HistEntry *h = &t->hist.e[i];
    json_begin_object(j, NULL);
    json_key_int(j, "step", h->step);
    json_key_int(j, "rows_seen", h->rows_seen);
    json_key_str(j, "tag", h->tag);
    json_key_double(j, "dev_soft_ce", h->dev_soft_ce, -1);
    json_key_double(j, "dev_agreement", h->dev_agreement, -1);
    json_key_double(j, "train_loss", h->train_loss, -1);
    json_key_double(j, "seconds", h->seconds, -1);
    json_key_double(j, "peak_gb", h->peak_gb, 2);
    if (h->saved) json_key_bool(j, "saved", 1);
    json_end_object(j);
  }
  json_end_array(j);
  if (t->trained) {
    json_key_double(j, "train_seconds", t->train_seconds, -1);
    json_key_double(j, "rows_per_second", t->rows_per_second, 2);
    json_key_int(j, "steps", t->step);
    json_key_double(j, "best_dev_soft_ce", t->best, -1);
  }
  for (int i = 0; i < t->n_ev; i++) {
    char key[JT_TAG_MAX + 8];
    snprintf(key, sizeof key, "eval_%s", t->evres[i].name);
    json_begin_object(j, key);
    json_key_int(j, "n", t->evres[i].n);
    json_key_double(j, "soft_ce", round_to(eval_ce(&t->evres[i]), 5), -1);
    json_key_double(j, "teacher_argmax_agreement", round_to(eval_agree(&t->evres[i]), 4), -1);
    json_key_double(j, "seconds", round_to(t->evres[i].seconds, 1), -1);
    json_end_object(j);
  }
  json_key_double(j, "peak_gb", round_to(t->peak, 2), 2);
  json_end_object(j);
  json_close(j);
}

// ---- the forward dump (plan M4.3's gate) -------------------------------------------------------------------------------
// dump_ref.py's batch 0: the rows' real tokens packed back to back, each row's live option count and weight, and the
// targets padded with zeros past the live count (train.py's collate does the same, so ce_head sees the same array).
typedef struct {
  int rows;                 // B
  int n;                    // sum of the rows' lengths
  int row[JT_MAX_ROWS];     // the file's row indices of batch 0 (for the log line)
  int off[JT_MAX_ROWS + 1]; // prefix sums of the lengths
  int live[JT_MAX_ROWS];    // each row's live option count
  float wrow[JT_MAX_ROWS];  // each row's weight
  float wgroup;             // train.py's gw: the batch's summed weight
  int n_opts;               // the widest row's live count = the logits width
  int *ids;                 // host [n]
  float *targets;           // host [rows][n_opts], zero past each row's live count
} DumpBatch;

static void dump_batch_free(DumpBatch *d) {
  free(d->ids);
  free(d->targets);
  d->ids = NULL;
  d->targets = NULL;
}

static void read_batch0(const char *path, DumpBatch *d) {
  Jtb *j = jtb_open(path);
  if (jtb_n_batches(j) < 1) die("%s: the file has no batches", path);
  uint32_t nb = 0;
  const uint32_t *rows = jtb_batch(j, 0, &nb);
  if (nb < 1) die("%s: batch 0 is empty", path);
  if ((int)nb > JT_MAX_ROWS) die("%s: batch 0 has %u rows, at most %d are supported", path, nb, JT_MAX_ROWS);

  memset(d, 0, sizeof *d);
  d->rows = (int)nb;
  int n = 0, max_opts = 0;
  double wsum = 0.0;
  for (int r = 0; r < d->rows; r++) {
    const JtbRow *row = jtb_row(j, rows[r]);
    if (row == NULL) die("%s: batch 0 names row %u, which the file does not have", path, rows[r]);
    if (row->len < 1) die("%s: row %u is empty", path, rows[r]);
    if (row->n_opts < 1 || row->n_opts > JTB_MAX_OPTS)
      die("%s: row %u has %u options, want 1..%u", path, rows[r], row->n_opts, JTB_MAX_OPTS);
    d->row[r] = (int)rows[r];
    d->off[r] = n;
    n += (int)row->len;
    d->live[r] = (int)row->n_opts;
    d->wrow[r] = row->weight;
    wsum += (double)row->weight;
    if ((int)row->n_opts > max_opts) max_opts = (int)row->n_opts;
  }
  d->off[d->rows] = n;
  d->n = n;
  d->n_opts = max_opts;
  d->wgroup = (float)wsum; // train.py's `gw` (its loss is `(per * wt).sum() / gw`)
  if (!(d->wgroup > 0.0f)) die("%s: batch 0's row weights sum to %g; the loss divides by that", path, (double)d->wgroup);

  d->ids = (int *)malloc((size_t)n * sizeof *d->ids);
  if (d->ids == NULL) die("out of host memory (%d token ids)", n);
  d->targets = (float *)calloc((size_t)d->rows * (size_t)max_opts, sizeof *d->targets);
  if (d->targets == NULL) die("out of host memory (%d x %d targets)", d->rows, max_opts);
  for (int r = 0; r < d->rows; r++) {
    const JtbRow *row = jtb_row(j, rows[r]);
    const uint32_t *ids = jtb_ids(j, rows[r]);
    const float *t = jtb_targets(j, rows[r]);
    for (int k = 0; k < (int)row->len; k++) d->ids[d->off[r] + k] = (int)ids[k];
    memcpy(d->targets + (size_t)r * (size_t)max_opts, t, (size_t)row->n_opts * sizeof(float));
  }
  jtb_close(j);
}

// Every key dump_ref.py writes for the forward, in its own order, from the device buffers to one savetensors file. The
// in.ids/in.off cast to float is the note in the file's header comment.
static void write_dump(JtCtx *c, const char *path, const DumpBatch *d, const float *saved, const float *post,
                       const float *final, const float *logits, float loss) {
  const size_t stride = (size_t)d->n * MT_HIDDEN;
  const int nk = 2 + 3 * MT_LAYERS + 3;

  const float **data = (const float **)calloc((size_t)nk, sizeof *data);
  const char **names = (const char **)calloc((size_t)nk, sizeof *names);
  int (*shapes)[ST_MAX_DIMS] = (int(*)[ST_MAX_DIMS])calloc((size_t)nk, sizeof *shapes);
  int *ndims = (int *)calloc((size_t)nk, sizeof *ndims);
  char (*namebuf)[32] = (char(*)[32])calloc((size_t)nk, 32);
  const float *off_f = NULL, *ids_f = NULL, *loss_f = NULL;
  float *saved_h = NULL, *post_h = NULL, *final_h = NULL, *logits_h = NULL;
  if (data == NULL || names == NULL || shapes == NULL || ndims == NULL || namebuf == NULL)
    die("out of host memory (the dump's key table, %d keys)", nk);

  saved_h = dev_pull(c, saved, (size_t)(MT_LAYERS + 1) * stride);
  post_h = dev_pull(c, post, (size_t)MT_LAYERS * stride);
  final_h = dev_pull(c, final, stride);
  logits_h = dev_pull(c, logits, (size_t)d->rows * (size_t)d->n_opts);

  // in.ids / in.off as F32: the writer is F32-only, and these values (ids < 248320, offsets < N) are exact.
  float *ids = host_alloc((size_t)d->n);
  float *off = host_alloc((size_t)d->rows + 1);
  for (int k = 0; k < d->n; k++) ids[k] = (float)d->ids[k];
  for (int r = 0; r <= d->rows; r++) off[r] = (float)d->off[r];
  float *loss1 = host_alloc(1);
  loss1[0] = loss;
  ids_f = ids;
  off_f = off;
  loss_f = loss1;

  int k = 0;
  snprintf(namebuf[k], 32, "in.ids");
  shapes[k][0] = d->n, ndims[k] = 1, data[k] = ids_f, k++;
  snprintf(namebuf[k], 32, "in.off");
  shapes[k][0] = d->rows + 1, ndims[k] = 1, data[k] = off_f, k++;
  for (int i = 0; i < MT_LAYERS; i++) {
    snprintf(namebuf[k], 32, "layer.%d.in", i);
    shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = saved_h + (size_t)i * stride, k++;
    snprintf(namebuf[k], 32, "layer.%d.post_mixer", i);
    shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = post_h + (size_t)i * stride, k++;
    snprintf(namebuf[k], 32, "layer.%d.out", i);
    shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = saved_h + (size_t)(i + 1) * stride, k++;
  }
  snprintf(namebuf[k], 32, "final");
  shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = final_h, k++;
  snprintf(namebuf[k], 32, "logits");
  shapes[k][0] = d->rows, shapes[k][1] = d->n_opts, ndims[k] = 2, data[k] = logits_h, k++;
  snprintf(namebuf[k], 32, "loss");
  ndims[k] = 0, data[k] = loss_f, k++;

  for (int i = 0; i < k; i++) names[i] = namebuf[i];
  st_write(path, k, names, (const int(*)[ST_MAX_DIMS])shapes, ndims, data);
  // the data section: n ids + (rows+1) offsets + 49 activation slabs + the logits + the scalar loss
  const size_t nfloat = (size_t)d->n + (size_t)d->rows + 1 + (size_t)(2 * MT_LAYERS + 1) * stride +
                        (size_t)d->rows * (size_t)d->n_opts + 1;
  printf("jefftrain: wrote %s: %d keys (2 in + %d layers x 3 + final, logits, loss), %.1f MiB of data, loss %.8f\n",
         path, k, MT_LAYERS, (double)nfloat * sizeof(float) / 1e6, (double)loss);
  fflush(stdout);
  free(loss1);
  free(off);
  free(ids);
  free(logits_h);
  free(final_h);
  free(post_h);
  free(saved_h);
  free(namebuf);
  free(ndims);
  free(shapes);
  free(names);
  free(data);
}

static void run_dump_forward(Model *m, Lora *l, const char *jtb_path, const char *out) {
  DumpBatch d;
  read_batch0(jtb_path, &d);
  int lmax = 0;
  for (int r = 0; r < d.rows; r++)
    if (d.off[r + 1] - d.off[r] > lmax) lmax = d.off[r + 1] - d.off[r];
  printf("jefftrain: dump-forward: %s batch 0 = rows [", jtb_path);
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.row[r]);
  printf("], lengths [");
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.off[r + 1] - d.off[r]);
  printf("], n %d tokens, %d option(s), gw %g\n", d.n, d.n_opts, (double)d.wgroup);
  fflush(stdout);

  JtBatch b;
  jt_batch_set(&b, d.rows, d.n, d.off);
  JtCtx *c = jt_ctx_alloc(m, l, d.n, d.rows, lmax);

  const size_t stride = (size_t)d.n * MT_HIDDEN;
  int *ids = NULL;
  CK(cudaMalloc((void **)&ids, (size_t)d.n * sizeof(int)));
  float *saved = dev_alloc((size_t)(MT_LAYERS + 1) * stride);
  float *post = dev_alloc((size_t)MT_LAYERS * stride);
  float *final = dev_alloc(stride);
  float *logits = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *targets = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *wrow = dev_alloc((size_t)d.rows);
  CK(cudaMemcpyAsync(ids, d.ids, (size_t)d.n * sizeof(int), cudaMemcpyHostToDevice, c->stream));
  CK(cudaMemcpyAsync(targets, d.targets, (size_t)d.rows * (size_t)d.n_opts * sizeof(float), cudaMemcpyHostToDevice,
                     c->stream));
  CK(cudaMemcpyAsync(wrow, d.wrow, (size_t)d.rows * sizeof(float), cudaMemcpyHostToDevice, c->stream));
  CK(cudaStreamSynchronize(c->stream));

  int rc = model_forward(c, ids, d.n, &b, saved, post);
  if (rc == JT_LAYER_NOT_IMPL) die("the forward is incomplete: a layer is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("model_forward returned %d", rc);

  HeadBatch hb;
  hb.n_opts = d.n_opts;
  hb.live = d.live;
  hb.targets = targets;
  hb.wrow = wrow;
  hb.wgroup = d.wgroup;
  float loss = NAN;
  jt_saves_reset(c); // the sweep keeps nothing (JT_FWD_ONLY); the head's own work comes from the arena's work half
  rc = head_fwd(c, saved + (size_t)MT_LAYERS * stride, d.n, &b, &hb, final, logits, &loss, JT_FWD_ONLY);
  if (rc == JT_LAYER_NOT_IMPL) die("the head is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("head_fwd returned %d", rc);

  write_dump(c, out, &d, saved, post, final, logits, loss);

  dev_free(&wrow);
  dev_free(&targets);
  dev_free(&logits);
  dev_free(&final);
  dev_free(&post);
  dev_free(&saved);
  if (ids != NULL) CK(cudaFree(ids));
  jt_ctx_free(c);
  jt_batch_free(&b);
  dump_batch_free(&d);
}

// ---- the backward dump (plan M5.1 item 4's gate, and the flag plan M5.4 needs) -------------------------------------------
#define JT_DUMP_NAME_MAX 96

// The keys are collected in one table because st_write writes one file: names, shapes and data pointers only.
typedef struct {
  int n, cap;
  const float **data;
  const char **names;
  int (*shapes)[ST_MAX_DIMS];
  int *ndims;
  char (*namebuf)[JT_DUMP_NAME_MAX];
} KeyTable;

static void kt_init(KeyTable *t, int cap) {
  memset(t, 0, sizeof *t);
  t->cap = cap;
  t->data = (const float **)calloc((size_t)cap, sizeof *t->data);
  t->names = (const char **)calloc((size_t)cap, sizeof *t->names);
  t->shapes = (int(*)[ST_MAX_DIMS])calloc((size_t)cap, sizeof *t->shapes);
  t->ndims = (int *)calloc((size_t)cap, sizeof *t->ndims);
  t->namebuf = (char(*)[JT_DUMP_NAME_MAX])calloc((size_t)cap, JT_DUMP_NAME_MAX);
  if (t->data == NULL || t->names == NULL || t->shapes == NULL || t->ndims == NULL || t->namebuf == NULL)
    die("out of host memory (the dump's key table, %d keys)", cap);
}

static void kt_free(KeyTable *t) {
  free(t->namebuf);
  free(t->ndims);
  free(t->shapes);
  free(t->names);
  free(t->data);
  memset(t, 0, sizeof *t);
}

// One key: ndims is 0 (a scalar), 1 (d0) or 2 (d0 x d1). `data` must stay alive until the write.
static void kt_add(KeyTable *t, const char *name, int ndims, int d0, int d1, const float *data) {
  if (t->n >= t->cap) die("the dump's key table is full (%d keys)", t->cap);
  char *dst = t->namebuf[t->n];
  if (snprintf(dst, JT_DUMP_NAME_MAX, "%s", name) >= JT_DUMP_NAME_MAX)
    die("the dump key '%s' does not fit in %d bytes", name, JT_DUMP_NAME_MAX);
  t->names[t->n] = dst;
  t->shapes[t->n][0] = d0;
  t->shapes[t->n][1] = d1;
  t->ndims[t->n] = ndims;
  t->data[t->n] = data;
  t->n++;
}

// The host copies a --dump-backward holds: the forward keys (the same buffers --dump-forward writes), the layer-output
// gradients, grad.final and one pull of the flat parameter-gradient buffer.
typedef struct {
  float *saved;      // (MT_LAYERS + 1) slabs: layer.{i}.in and layer.{i}.out
  float *post;       // MT_LAYERS slabs: layer.{i}.post_mixer, from the recompute
  float *final;      // one slab
  float *logits;     // [rows][n_opts]
  float *loss;       // one scalar
  float *grad_layer; // MT_LAYERS slabs: grad.layer.{i}.out (the value on entry to layer i's backward)
  float *grad_final; // one slab
  float *flat;       // ctx->lora->grad, n_param floats (the LoRA factors and the readout are slices of it)
} GradDump;

static void grad_dump_free(GradDump *g) {
  free(g->flat);
  free(g->grad_final);
  free(g->grad_layer);
  free(g->loss);
  free(g->logits);
  free(g->final);
  free(g->post);
  free(g->saved);
  memset(g, 0, sizeof *g);
}

// dump_ref.py's keys, its order (in.* first, then the layers, then final/logits/loss, then the gradients), from the host
// copies above. st_write sorts the keys itself, so the order is for a human reading the table.
static void write_dump_backward(JtCtx *c, const char *path, const DumpBatch *d, const GradDump *g) {
  const size_t stride = (size_t)d->n * MT_HIDDEN;
  KeyTable t;
  kt_init(&t, 2 + 3 * MT_LAYERS + 3 + MT_LAYERS + 1 + 2 * MT_LORA_PAIRS + 1);

  float *ids = host_alloc((size_t)d->n);
  float *off = host_alloc((size_t)d->rows + 1);
  for (int k = 0; k < d->n; k++) ids[k] = (float)d->ids[k];
  for (int r = 0; r <= d->rows; r++) off[r] = (float)d->off[r];
  kt_add(&t, "in.ids", 1, d->n, 0, ids);
  kt_add(&t, "in.off", 1, d->rows + 1, 0, off);

  char name[JT_DUMP_NAME_MAX];
  for (int i = 0; i < MT_LAYERS; i++) {
    snprintf(name, sizeof name, "layer.%d.in", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->saved + (size_t)i * stride);
    snprintf(name, sizeof name, "layer.%d.post_mixer", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->post + (size_t)i * stride);
    snprintf(name, sizeof name, "layer.%d.out", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->saved + (size_t)(i + 1) * stride);
  }
  kt_add(&t, "final", 2, d->n, MT_HIDDEN, g->final);
  kt_add(&t, "logits", 2, d->rows, d->n_opts, g->logits);
  kt_add(&t, "loss", 0, 0, 0, g->loss);

  for (int i = 0; i < MT_LAYERS; i++) { // dump_ref's keys for the gradient at each layer's output, 0..23
    snprintf(name, sizeof name, "grad.layer.%d.out", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->grad_layer + (size_t)i * stride);
  }
  kt_add(&t, "grad.final", 2, d->n, MT_HIDDEN, g->grad_final);
  for (int k = 0; k < MT_LORA_PAIRS; k++) { // 150 adapted projections: A [rank,in] then B [out,rank]
    const LoraEntry *e = &c->lora->ent[k];
    snprintf(name, sizeof name, "grad.%s.lora_A", e->stem);
    kt_add(&t, name, 2, e->rank, e->in, g->flat + e->off);
    snprintf(name, sizeof name, "grad.%s.lora_B", e->stem);
    kt_add(&t, name, 2, e->out, e->rank, g->flat + e->b_off);
  }
  const LoraEntry *ro = &c->lora->ent[LORA_READOUT_ENTRY];
  kt_add(&t, "grad.readout", 2, ro->out, ro->in, g->flat + ro->off);

  st_write(path, t.n, t.names, (const int(*)[ST_MAX_DIMS])t.shapes, t.ndims, t.data);
  const size_t nfloat = (size_t)d->n + (size_t)d->rows + 1 + (size_t)(3 * MT_LAYERS + MT_LAYERS + 2) * stride +
                        (size_t)d->rows * (size_t)d->n_opts + 1 + c->lora->n_param;
  printf("jefftrain: wrote %s: %d keys (the %d forward keys + %d layer-gradient keys + grad.final + %d LoRA grads + "
         "grad.readout), %.1f MiB of data, loss %.8f\n",
         path, t.n, 2 + 3 * MT_LAYERS + 3, MT_LAYERS, 2 * MT_LORA_PAIRS, (double)nfloat * sizeof(float) / 1e6,
         (double)g->loss[0]);
  fflush(stdout);
  free(off);
  free(ids);
  kt_free(&t);
}

static void run_dump_backward(Model *m, Lora *l, const char *jtb_path, const char *out) {
  DumpBatch d;
  read_batch0(jtb_path, &d);
  int lmax = 0;
  for (int r = 0; r < d.rows; r++)
    if (d.off[r + 1] - d.off[r] > lmax) lmax = d.off[r + 1] - d.off[r];
  printf("jefftrain: dump-backward: %s batch 0 = rows [", jtb_path);
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.row[r]);
  printf("], lengths [");
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.off[r + 1] - d.off[r]);
  printf("], n %d tokens, %d option(s), gw %g\n", d.n, d.n_opts, (double)d.wgroup);
  fflush(stdout);

  JtBatch b;
  jt_batch_set(&b, d.rows, d.n, d.off);
  JtCtx *c = jt_ctx_alloc(m, l, d.n, d.rows, lmax);

  const size_t stride = (size_t)d.n * MT_HIDDEN;
  int *ids = NULL;
  CK(cudaMalloc((void **)&ids, (size_t)d.n * sizeof(int)));
  float *saved = dev_alloc((size_t)(MT_LAYERS + 1) * stride);
  float *post = dev_alloc((size_t)MT_LAYERS * stride);
  float *final = dev_alloc(stride);
  float *logits = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *targets = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *wrow = dev_alloc((size_t)d.rows);
  float *dres = dev_alloc(stride);                           // the sweep's one gradient buffer (design §6.2)
  float *grad_layer = dev_alloc((size_t)MT_LAYERS * stride); // grad.layer.{i}.out, copied out on entry to layer i's bwd
  CK(cudaMemcpyAsync(ids, d.ids, (size_t)d.n * sizeof(int), cudaMemcpyHostToDevice, c->stream));
  CK(cudaMemcpyAsync(targets, d.targets, (size_t)d.rows * (size_t)d.n_opts * sizeof(float), cudaMemcpyHostToDevice,
                     c->stream));
  CK(cudaMemcpyAsync(wrow, d.wrow, (size_t)d.rows * sizeof(float), cudaMemcpyHostToDevice, c->stream));
  CK(cudaStreamSynchronize(c->stream));

  int rc = model_forward(c, ids, d.n, &b, saved, post); // design §6.2: only each layer's input is kept
  if (rc == JT_LAYER_NOT_IMPL) die("the forward is incomplete: a layer is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("model_forward returned %d", rc);

  HeadBatch hb;
  hb.n_opts = d.n_opts;
  hb.live = d.live;
  hb.targets = targets;
  hb.wrow = wrow;
  hb.wgroup = d.wgroup;
  float loss = NAN;
  jt_saves_reset(c); // the head's saves start at the arena's base (src/head.c); the sweep kept nothing
  rc = head_fwd(c, saved + (size_t)MT_LAYERS * stride, d.n, &b, &hb, final, logits, &loss, JT_FWD_SAVE);
  if (rc == JT_LAYER_NOT_IMPL) die("the head is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("head_fwd returned %d", rc);

  lora_zero_grad(l, c->stream); // dR/dA/dB accumulate into the flat buffer (src/linear.h, src/head.c)
  const size_t hmark = jt_work_mark(c);
  rc = head_bwd(c, saved + (size_t)MT_LAYERS * stride, d.n, &b, dres);
  if (rc != JT_LAYER_OK) die("head_bwd returned %d", rc);
  // grad.final is head_bwd's dh, its first (and only) work-half allocation, so it sits at arena + mark - n * MT_HIDDEN
  // (src/head.c's header comment). Pulled here: the layer loop takes those very words for its own scratch.
  float *grad_final = dev_pull(c, c->arena + hmark - (size_t)d.n * MT_HIDDEN, stride);

  rc = model_backward(c, saved, d.n, &b, post, grad_layer, dres); // design §6.2's recompute loop, 23 down to 0
  if (rc != JT_LAYER_OK) die("model_backward returned %d", rc);
  printf("jefftrain: backward: %d layers recomputed from their saved inputs and differentiated\n", MT_LAYERS);
  fflush(stdout);

  GradDump g;
  memset(&g, 0, sizeof g);
  g.saved = dev_pull(c, saved, (size_t)(MT_LAYERS + 1) * stride);
  g.post = dev_pull(c, post, (size_t)MT_LAYERS * stride); // the recompute's post_mixer (what the backwards used)
  g.final = dev_pull(c, final, stride);
  g.logits = dev_pull(c, logits, (size_t)d.rows * (size_t)d.n_opts);
  g.loss = host_alloc(1);
  g.loss[0] = loss;
  g.grad_layer = dev_pull(c, grad_layer, (size_t)MT_LAYERS * stride);
  g.grad_final = grad_final;
  g.flat = dev_pull(c, l->grad, l->n_param); // the 300 LoRA gradients and the readout are slices of this
  write_dump_backward(c, out, &d, &g);

  grad_dump_free(&g);
  dev_free(&grad_layer);
  dev_free(&dres);
  dev_free(&wrow);
  dev_free(&targets);
  dev_free(&logits);
  dev_free(&final);
  dev_free(&post);
  dev_free(&saved);
  if (ids != NULL) CK(cudaFree(ids));
  jt_ctx_free(c);
  jt_batch_free(&b);
  dump_batch_free(&d);
}

// ---- the run ---------------------------------------------------------------------------------------------------------
// --eval-temperature: a number, or "checkpoint" for the answer temperature in the checkpoint's decision_config.json
// (train.py's `--eval-temperature checkpoint`, its load_decision_config()).
static double resolve_temperature(const Args *a, const char *model_dir) {
  const char *spec = a->temperature != NULL ? a->temperature : "1";
  if (strcmp(spec, "checkpoint") != 0) return num_arg("--eval-temperature", spec, 1e-6, 1e6);
  char path[JT_PATH_MAX];
  path_join(path, sizeof path, model_dir, "decision_config.json");
  char *text = json_file(path);
  if (text == NULL) die("--eval-temperature checkpoint: %s does not exist", path);
  double v = 0.0;
  if (!json_double(text, "temperature", &v)) die("--eval-temperature checkpoint: %s has no numeric temperature", path);
  free(text);
  if (!(v > 0.0)) die("--eval-temperature checkpoint: %s says temperature %g", path, v);
  return v;
}

// The largest packed batch, row count and row length over the run's splits: the context and the device buffers are sized
// from these (design §2: never hard-code a maximum, take it from the .jtb header).
typedef struct {
  int cap_tokens, cap_rows, lmax;
} Caps;

static void caps_take(Caps *c, const Split *s) {
  const int tok = (int)jtb_max_tokens_per_batch(s->j);
  const int len = (int)jtb_max_len(s->j);
  if (tok > c->cap_tokens) c->cap_tokens = tok;
  if (len > c->lmax) c->lmax = len;
  for (uint32_t k = 0; k < s->n_batches; k++) {
    uint32_t n = 0;
    jtb_batch(s->j, k, &n);
    if ((int)n > c->cap_rows) c->cap_rows = (int)n;
  }
}

int main(int argc, char **argv) {
  if (argc == 1) {
    usage(stdout);
    return 0;
  }
  Args a;
  parse_args(argc, argv, &a);
  if (a.model == NULL) die("--model is required (--help for the usage)");
  if (a.dump_fwd != NULL && a.dump_bwd != NULL) die("--dump-forward and --dump-backward are separate runs; pass one");
  const int dumping = a.dump_fwd != NULL || a.dump_bwd != NULL;
  if (dumping) {
    if (a.train == NULL) die("--dump-forward/--dump-backward run a batch, so they need --train FILE.jtb");
  } else if (a.eval_only) {
    if (a.n_eval == 0) die("--eval-only needs at least one --eval NAME=FILE.jtb");
  } else {
    if (a.train == NULL) die("--train FILE.jtb is required unless --eval-only");
    if (a.dev == NULL) die("--dev FILE.jtb is required unless --eval-only");
  }
  if (!dumping && a.output == NULL) die("--output DIR is required (the adapter and report.json go there)");

  // ---- the splits: opened and checked before the checkpoint load, so a flag/file mismatch costs no time ---------------
  char mdir[JT_PATH_MAX];
  model_dir_of(a.model, mdir, sizeof mdir);
  if (!dumping) check_output_dir(a.output, !a.eval_only);
  Caps caps;
  memset(&caps, 0, sizeof caps);
  caps.cap_tokens = 1;
  caps.cap_rows = 1;
  caps.lmax = 1;

  Split train_s, dev_s;
  memset(&train_s, 0, sizeof train_s);
  memset(&dev_s, 0, sizeof dev_s);
  Split *evs[JT_MAX_EVALS];
  memset(evs, 0, sizeof evs);
  int n_ev = 0;
  // The dev and eval batch sizes are train.py's: batch*2 for dev, max(2, batch*2) for the eval splits (design §9.1).
  const int dev_batch = a.batch * 2;
  const int ev_batch = a.batch * 2 > 2 ? a.batch * 2 : 2;
  if (!a.eval_only && !dumping) {
    split_open(&train_s, "train", a.train, a.batch, a.epochs);
    split_check_sidecar(&train_s, &a);
    split_open(&dev_s, "dev", a.dev, dev_batch, 1);
    split_check_sidecar(&dev_s, &a);
    caps_take(&caps, &train_s);
    caps_take(&caps, &dev_s);
  }
  for (int i = 0; !dumping && i < a.n_eval; i++) { // the dumps run one batch and write one file: --eval is not theirs
    evs[i] = (Split *)calloc(1, sizeof *evs[i]);
    if (evs[i] == NULL) die("out of memory (an --eval split)");
    split_open(evs[i], a.eval_name[i], a.eval_path[i], ev_batch, 1);
    split_check_sidecar(evs[i], &a);
    caps_take(&caps, evs[i]);
    n_ev++;
  }
  if (a.targets == NULL) a.targets = "soft";
  if (a.weights == NULL) a.weights = "none";

  // ---- the model ------------------------------------------------------------------------------------------------------
  gemm_init(); // prints "gemm: exact fp32" or "gemm: tf32" once, before anything allocates (design §5)
  Lora *lora = lora_alloc();
  Model *m = model_load(a.model, lora);
  if (a.adapter != NULL) {
    lora_load_adapter(lora, a.adapter);
    printf("jefftrain: adapter %s loaded over the checkpoint's readout\n", a.adapter);
    fflush(stdout);
  }

  if (dumping) { // plan M4.3/M5.1: one batch, one file, nothing else
    if (a.dump_fwd != NULL) run_dump_forward(m, lora, a.train, a.dump_fwd);
    if (a.dump_bwd != NULL) run_dump_backward(m, lora, a.train, a.dump_bwd);
    model_free(m);
    lora_free(lora);
    return 0;
  }

  // ---- the context, the step's buffers, and the run -------------------------------------------------------------------
  Trainer t;
  memset(&t, 0, sizeof t);
  t.a = &a;
  t.train = a.eval_only ? NULL : &train_s;
  t.dev = a.eval_only ? NULL : &dev_s;
  for (int i = 0; i < n_ev; i++) t.ev[i] = evs[i];
  t.n_ev = n_ev;
  t.best = HUGE_VAL;
  t.temperature = resolve_temperature(&a, mdir);
  t.meta.base = a.model;
  t.meta.targets = a.targets;
  t.meta.weights = a.weights;
  path_join(t.adapter_dir, sizeof t.adapter_dir, a.output, "adapter");
  mkdirs(a.output);

  t.c = jt_ctx_alloc(m, lora, caps.cap_tokens, caps.cap_rows, caps.lmax);
  Dev dev;
  dev_init(&dev, caps.cap_tokens, caps.cap_rows);
  t.d = &dev;
  t.peak = dev_mem_gb(); // what the load and the context already hold
  trace_open(&t.trace, a.trace);

  if (!a.eval_only) {
    t.per_epoch_batches = (int)train_s.per_epoch;
    t.per_epoch = (int)ceil_div(train_s.per_epoch, (uint32_t)a.accum); // optimizer steps per epoch (train.py's)
    t.total = t.per_epoch * a.epochs;
    if (a.max_steps > 0 && a.max_steps < t.total) t.total = a.max_steps;
    if (t.total < 1) die("%s: no optimizer steps to run (%d per epoch, --max-steps %d)", a.train, t.per_epoch,
                         a.max_steps);
    printf("jefftrain: device cuda  model %s  train %u row(s)  dev %u row(s)  eval [", a.model, train_s.n_rows,
           dev_s.n_rows);
  } else {
    printf("jefftrain: device cuda  model %s  eval-only  eval [", a.model);
  }
  for (int i = 0; i < t.n_ev; i++) printf("%s%s:%u", i != 0 ? ", " : "", t.ev[i]->name, t.ev[i]->n_rows);
  printf("]\n");
  if (!a.eval_only)
    printf("jefftrain: LoRA layers %d  %s targets  weights %s  steps %d (batch %d x accum %d, %d per epoch)\n",
           MT_LORA_PAIRS, a.targets, a.weights, t.total, a.batch, a.accum, t.per_epoch);
  fflush(stdout);

  if (!a.eval_only) {
    t.trained = 1;
    t.t_train = jt_now();
    trainer_run(&t);
    const double wall = jt_now() - t.t_train;
    t.train_seconds = round_to(wall, 0);
    t.rows_per_second = round_to((double)t.seen / (wall > 0 ? wall : 1.0), 2);
    // the evals score the best dev check's adapter, not the last step's (train.py reloads it here)
    lora_load_adapter(lora, t.adapter_dir);
    printf("jefftrain: reloaded the adapter from the best dev check (%.5f) at %s\n", t.best, t.adapter_dir);
    fflush(stdout);
  }

  for (int i = 0; i < t.n_ev; i++) {
    Eval *e = &t.evres[i];
    memset(e, 0, sizeof *e);
    e->name = t.ev[i]->name;
    e->probs = (double *)calloc((size_t)t.ev[i]->n_rows * JTB_MAX_OPTS, sizeof *e->probs);
    e->seen = (unsigned char *)calloc((size_t)t.ev[i]->n_rows, sizeof *e->seen);
    if (e->probs == NULL || e->seen == NULL) die("out of host memory (the predictions of %s)", e->name);
    eval_split(t.c, t.d, t.ev[i], t.temperature, 1, e);
    char pred[JT_PATH_MAX], name[JT_TAG_MAX + 16];
    snprintf(name, sizeof name, "pred-%s.jsonl", e->name);
    path_join(pred, sizeof pred, a.output, name);
    write_preds(t.ev[i], e, pred);
    printf("eval %s: soft CE %.4f  agree %.3f  (%.0fs)\n", e->name, eval_ce(e), eval_agree(e), e->seconds);
    fflush(stdout);
  }
  const double last = dev_mem_gb();
  if (last > t.peak) t.peak = last;
  report_write(&t);
  printf("peak %.2f GB  -> %s\n", round_to(t.peak, 2), a.output);
  fflush(stdout);

  trace_close(&t.trace);
  for (int i = 0; i < t.n_ev; i++) {
    free(t.evres[i].seen);
    free(t.evres[i].probs);
    split_close(evs[i]);
    free(evs[i]);
  }
  if (!a.eval_only) {
    split_close(&train_s);
    split_close(&dev_s);
  }
  free(t.hist.e);
  free(t.tr_row);
  free(t.tr_ce);
  dev_free_all(&dev);
  jt_ctx_free(t.c);
  model_free(m);
  lora_free(lora);
  return 0;
}
