// src/json.c — the indenting JSON writer and the two field scanners src/json.h describes.
//
// The writer keeps a stack of open containers, so the comma/newline/indent rule lives in one place (value_prefix) and no
// caller has to remember whether it is first. Everything goes through stdio; a short write or a write error is caught
// by json_close (fflush + ferror), so a full disk is a message and exit(1), not a truncated report that looks valid.
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

typedef struct {
  int is_object; // 1 = { }, 0 = [ ]
  int count;     // members/elements written so far (drives the comma and the empty-container spelling)
} JsonFrame;

struct Json {
  FILE *f;
  int owns;                  // 1 = json_open() opened it, so json_close() closes it
  int compact;               // 1 = one line, "," separators (a JSON-lines record)
  int depth;                 // open containers; 0 = the top level
  JsonFrame stack[JT_JSON_DEPTH];
};

static void die(const char *fmt, ...) {
  va_list ap;
  fflush(stdout);
  fputs("json: ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

// ---- writing -----------------------------------------------------------------------------------------------------------
Json *json_open(const char *path) {
  if (path == NULL) die("json_open: no path");
  FILE *f = fopen(path, "w");
  if (f == NULL) die("cannot write %s: %s", path, strerror(errno));
  Json *j = json_open_stream(f);
  j->owns = 1;
  return j;
}

Json *json_open_stream(FILE *f) {
  if (f == NULL) die("json_open_stream: no stream");
  Json *j = (Json *)calloc(1, sizeof *j);
  if (j == NULL) die("out of memory (the JSON writer)");
  j->f = f;
  return j;
}

void json_close(Json *j) {
  if (j == NULL) return;
  if (j->depth != 0) die("%d container(s) left open when the document ended", j->depth);
  if (j->owns) {
    if (fflush(j->f) != 0 || ferror(j->f)) die("write error on the output: %s", strerror(errno));
    if (fclose(j->f) != 0) die("cannot finish the output: %s", strerror(errno));
  } else if (fflush(j->f) != 0) {
    die("write error on the stream: %s", strerror(errno));
  }
  free(j);
}

static void put_indent(Json *j) {
  if (j->compact) return;
  for (int i = 0; i < j->depth; i++) fputs("  ", j->f);
}

// "name": with the escaping JSON needs. A name that is not valid UTF-8 is written as-is (all of ours are ASCII: the
// criteria keys "A".."AA", the report's field names and the adapter stems).
static void put_string(Json *j, const char *s) {
  fputc('"', j->f);
  for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; p++) {
    switch (*p) {
    case '"': fputs("\\\"", j->f); break;
    case '\\': fputs("\\\\", j->f); break;
    case '\n': fputs("\\n", j->f); break;
    case '\r': fputs("\\r", j->f); break;
    case '\t': fputs("\\t", j->f); break;
    case '\b': fputs("\\b", j->f); break;
    case '\f': fputs("\\f", j->f); break;
    default:
      if (*p < 0x20) fprintf(j->f, "\\u%04x", (unsigned)*p);
      else fputc((int)*p, j->f);
    }
  }
  fputc('"', j->f);
}

// Everything a value needs before it: a comma if the container already has one, a newline, the indent — and, inside an
// object, the member's name. Top level (depth == 0): nothing.
static void value_prefix(Json *j, const char *key) {
  if (j->depth > 0) {
    JsonFrame *fr = &j->stack[j->depth - 1];
    if (fr->count > 0) fputc(',', j->f);
    if (!j->compact) {
      fputc('\n', j->f);
      put_indent(j);
    }
    fr->count++;
  }
  if (key != NULL) {
    put_string(j, key);
    fputs(j->compact ? ":" : ": ", j->f);
  }
}

static void push(Json *j, int is_object) {
  if (j->depth >= JT_JSON_DEPTH) die("more than %d nested containers", JT_JSON_DEPTH);
  j->stack[j->depth].is_object = is_object;
  j->stack[j->depth].count = 0;
  j->depth++;
}

static void pop(Json *j, char close) {
  if (j->depth <= 0) die("no container to close");
  const JsonFrame fr = j->stack[--j->depth];
  if (fr.count > 0 && !j->compact) { // an empty container stays "{}" / "[]", like json.dumps
    fputc('\n', j->f);
    put_indent(j);
  }
  fputc(close, j->f);
}

void json_begin_object(Json *j, const char *key) {
  value_prefix(j, key);
  fputc('{', j->f);
  push(j, 1);
}

void json_end_object(Json *j) {
  if (j->depth <= 0 || !j->stack[j->depth - 1].is_object) die("json_end_object: the open container is not an object");
  pop(j, '}');
}

void json_begin_array(Json *j, const char *key) {
  value_prefix(j, key);
  fputc('[', j->f);
  push(j, 0);
}

void json_end_array(Json *j) {
  if (j->depth <= 0 || j->stack[j->depth - 1].is_object) die("json_end_array: the open container is not an array");
  pop(j, ']');
}

void json_key_str(Json *j, const char *key, const char *v) {
  value_prefix(j, key);
  if (v == NULL) fputs("null", j->f);
  else put_string(j, v);
}

void json_key_int(Json *j, const char *key, long long v) {
  value_prefix(j, key);
  fprintf(j->f, "%lld", v);
}

void json_key_double(Json *j, const char *key, double v, int decimals) {
  value_prefix(j, key);
  if (!isfinite(v)) { // JSON has no NaN/Infinity
    fputs("null", j->f);
    return;
  }
  if (decimals >= 0) {
    fprintf(j->f, "%.*f", decimals, v);
    return;
  }
  char buf[40];
  snprintf(buf, sizeof buf, "%.15g", v); // 15 digits: what round(v, ndigits) needs and what Python prints back
  fputs(buf, j->f);
}

void json_key_bool(Json *j, const char *key, int v) {
  value_prefix(j, key);
  fputs(v ? "true" : "false", j->f);
}

void json_key_null(Json *j, const char *key) {
  value_prefix(j, key);
  fputs("null", j->f);
}

void json_newline(Json *j) { fputc('\n', j->f); }

void json_compact(Json *j, int compact) { j->compact = compact ? 1 : 0; }

// ---- reading -----------------------------------------------------------------------------------------------------------
char *json_file(const char *path) {
  if (path == NULL) return NULL;
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    if (errno == ENOENT) return NULL;
    die("cannot read %s: %s", path, strerror(errno));
  }
  size_t cap = 4096, n = 0;
  char *buf = (char *)malloc(cap);
  if (buf == NULL) die("out of memory reading %s", path);
  for (;;) {
    if (n + 1 >= cap) {
      cap *= 2;
      char *bigger = (char *)realloc(buf, cap);
      if (bigger == NULL) die("out of memory reading %s", path);
      buf = bigger;
    }
    const size_t got = fread(buf + n, 1, cap - n - 1, f);
    n += got;
    if (got == 0) break;
  }
  if (ferror(f)) die("read error on %s: %s", path, strerror(errno));
  fclose(f);
  buf[n] = '\0';
  return buf;
}

// Where the value of `"key"` starts, or NULL. The quoted token search means "rows_per_batch" never matches "batch".
static const char *value_of(const char *text, const char *key) {
  char token[128];
  const int n = snprintf(token, sizeof token, "\"%s\"", key);
  if (n <= 0 || n >= (int)sizeof token) return NULL;
  const char *p = text;
  while ((p = strstr(p, token)) != NULL) {
    p += (size_t)n;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') continue; // a string that happens to contain the name; keep looking
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
  }
  return NULL;
}

int json_int(const char *text, const char *key, long long *out) {
  if (text == NULL) return 0;
  const char *p = value_of(text, key);
  while (p != NULL) {
    char *end = NULL;
    errno = 0;
    const long long v = strtoll(p, &end, 10);
    if (end != p && errno == 0) {
      *out = v;
      return 1;
    }
    p = value_of(p, key); // the value was a string/null; try the next occurrence
  }
  return 0;
}

int json_double(const char *text, const char *key, double *out) {
  if (text == NULL) return 0;
  const char *p = value_of(text, key);
  while (p != NULL) {
    char *end = NULL;
    errno = 0;
    const double v = strtod(p, &end);
    if (end != p && errno == 0) {
      *out = v;
      return 1;
    }
    p = value_of(p, key);
  }
  return 0;
}

int json_str(const char *text, const char *key, char *out, size_t cap) {
  if (text == NULL || out == NULL || cap == 0) return 0;
  const char *p = value_of(text, key);
  while (p != NULL) {
    if (*p == '"') {
      size_t o = 0;
      for (p++; *p != '\0' && *p != '"'; p++) {
        char c = *p;
        if (c == '\\' && p[1] != '\0') { // the escapes the writer above produces
          p++;
          c = (*p == 'n') ? '\n' : (*p == 't') ? '\t' : (*p == 'r') ? '\r' : (*p == 'b') ? '\b' : (*p == 'f') ? '\f' : *p;
        }
        if (o + 1 < cap) out[o++] = c;
      }
      out[o] = '\0';
      return 1;
    }
    p = value_of(p, key);
  }
  return 0;
}
