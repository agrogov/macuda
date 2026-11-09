// src/json.h — just enough JSON for jefftrain's outputs and its two little input sidecars (design §13: "just enough JSON
// writing for report.json, adapter_config.json, pred-*.jsonl"; plan M5.2, design §9.2).
//
// Writing. A Json is an indenting writer: begin a container, add members/elements, end it. The layout is Python's
// json.dumps(..., indent=2) — one member per line, two spaces per level, "{}"/"[]" for an empty container — because
// report.json has to read like the one train.py writes, and because pred-<name>.jsonl is one record per line.
//
//   Json *j = json_open("/tmp/report.json");
//   json_begin_object(j, NULL);
//     json_key_str(j, "model", "jeff-base");
//     json_key_lldouble(j, "peak_gb", 3.21, 2);
//     json_begin_array(j, "history");           // an empty array prints as []
//     json_end_array(j);
//   json_end_object(j);
//   json_close(j);                              // closes the file and checks for write errors
//
// `key` is the member name inside an object, and NULL for an array element or the top-level value. An object member's
// key is always written by the same call that writes the value, so a key can never be emitted without one (or twice).
//
// Reading. Two small scanners, not a parser: `.jtb`'s `<split>.prep.json` (written by py/prep.py, plan M5.3) says which
// --batch/--accum/--epochs/targets/weights made the file, and a checkpoint's decision_config.json holds the answer
// temperature (`--eval-temperature checkpoint`, train.py's `load_decision_config`). Both are flat files whose fields
// this finds by name: json_file() loads the text, json_int()/json_double()/json_str() read one field. They look for the
// quoted key, skip to its ':' and parse what follows, so a field that is a string ("temperature": "<hash>" in a
// provenance block) is simply not the number the caller asked for and the next occurrence is tried.
//
// Every failure here is fatal with a message (exit 1), like CK()/CB(): a report that cannot be written is a run that
// did not record itself.
#ifndef JT_JSON_H
#define JT_JSON_H

#include <stddef.h>
#include <stdio.h>

// Nesting depth the writer will accept. A report is 4 levels deep (root, history, an entry, nothing else); 32 is room.
#define JT_JSON_DEPTH 32

typedef struct Json Json; // opaque: src/json.c

// Open `path` for writing (created/truncated). Dies on failure.
Json *json_open(const char *path);
// Wrap an already open stream (a JSON-lines file, or stdout). json_close() does NOT close it.
Json *json_open_stream(FILE *f);
// Finish the document, close the file (unless the stream was wrapped), die if a write failed, free the writer.
void json_close(Json *j);

// Containers. `key` may be NULL (an array element or the top-level value).
void json_begin_object(Json *j, const char *key);
void json_end_object(Json *j);
void json_begin_array(Json *j, const char *key);
void json_end_array(Json *j);

// Values. A double with decimals >= 0 prints with that many fractional digits ("%.Nf"); decimals < 0 prints the
// shortest form that round-trips ("%.17g", trimmed), which is what a count-like key wants. A non-finite double has no
// JSON spelling and is written as null.
void json_key_str(Json *j, const char *key, const char *v);
void json_key_int(Json *j, const char *key, long long v);
void json_key_double(Json *j, const char *key, double v, int decimals);
void json_key_bool(Json *j, const char *key, int v);
void json_key_null(Json *j, const char *key);

// End a record: a newline (JSON lines, design §9.2's pred-<name>.jsonl). The next record starts at column 0.
void json_newline(Json *j);

// Compact separators: 0 disables the indented layout and puts everything on one line with "," between
// members/elements (json.dumps' default spelling), which is what a JSON-lines record needs — a pretty-printed record
// would break a reader that reads the file line by line.
void json_compact(Json *j, int compact);

// ---- the two readers ---------------------------------------------------------------------------------------------------
// The whole file as a NUL-terminated malloc'd string (free it). NULL when the file does not exist; any other error is
// fatal. Used for the .prep.json sidecar, whose absence is normal (the synthetic fixtures have none).
char *json_file(const char *path);
// Find `"key"` (the exact quoted token) and read its value. Return 1 and store it when the value has the requested
// kind; 0 when the key is absent or its value is a different kind (a string, null, …). The first occurrence that fits
// wins, so a provenance hash named "temperature" does not hide the real number.
int json_int(const char *text, const char *key, long long *out);
int json_double(const char *text, const char *key, double *out);
// Copies the unescaped value (at most cap-1 bytes plus NUL) and returns 1, or 0 when the key is absent/not a string.
int json_str(const char *text, const char *key, char *out, size_t cap);

#endif // JT_JSON_H
