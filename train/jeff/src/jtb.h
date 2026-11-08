// src/jtb.h — the .jtb training-file reader (design §9.1). One file per split, written by py/prep.py (M5) and, until
// then, by py/make_fixture.py.
//
//   magic "JTB1" | u32 version=1 | u32 n_rows | u32 n_batches | u32 max_tokens_per_batch | u32 max_len | u32 max_opts
//   u64 section_offsets[5]                     (rows, ids, targets, batches, strings; from the start of the file)
//   rows:    n_rows x { u32 len, u32 n_opts, f32 weight, u32 teacher_top, u64 id_offset_in_strings }
//   ids:     sum(len) u32 token ids, row by row
//   targets: sum(n_opts) f32, row by row
//   batches: n_batches x { u32 n, u32 row_index[n] }
//   strings: per row: the id, NUL, its criteria keys each NUL-terminated, then an empty string (a final NUL)
//
// Little-endian, no padding between sections. jtb_open() mmaps the file, validates all of it (design §9.1: magic,
// version, every offset against the file size, every token id < 248320, every n_opts <= 255) and prints a clear
// message and exits(1) on a bad file — the same convention as st_get() (M1.3). The accessors hand out pointers into
// the map; nothing is copied except the parsed row table (24 bytes per row).
#ifndef JT_JTB_H
#define JT_JTB_H

#include <stddef.h>
#include <stdint.h>

#define JTB_MAGIC "JTB1"
#define JTB_VERSION 1u
// The readout has 255 classes and the tokenizer 248320 ids (design §2, §9.1): a file that says otherwise is corrupt,
// and a wrong n_opts would silently index past the end of a logits row.
#define JTB_MAX_OPTS 255u
#define JTB_VOCAB_MAX 248320u

// One row's header record. The first five members are the on-disk 24-byte record; ids_off and tgt_off are derived at
// open time (element indices into the ids and targets sections) so that the accessors are O(1).
typedef struct {
  uint32_t len;          // tokens in this row
  uint32_t n_opts;       // live options: the logits are read as [:n_opts]
  float weight;          // row weight (1.0 unless --weights balanced)
  uint32_t teacher_top;  // argmax of the targets, < n_opts
  uint64_t id_off;       // byte offset of the row's id inside the strings section
  uint64_t ids_off;      // index of the row's first token id inside the ids section
  uint64_t tgt_off;      // index of the row's first target inside the targets section
} JtbRow;

// The sections, in file order. JTB_NSEC is also the length of the offset table in the header.
enum { JTB_SEC_ROWS, JTB_SEC_IDS, JTB_SEC_TARGETS, JTB_SEC_BATCHES, JTB_SEC_STRINGS, JTB_NSEC };

typedef struct Jtb Jtb;  // opaque: src/jtb.c

// mmap and validate. Exits 1 with "jtb: <path>: <reason>" on a missing, truncated or inconsistent file.
Jtb *jtb_open(const char *path);
void jtb_close(Jtb *j);

// Header fields. max_len and max_opts size the trainer's length-dependent buffers and checkpoints (design §2, §9.1);
// max_tokens_per_batch sizes the per-step arena (design §6.2).
uint32_t jtb_version(const Jtb *j);
uint32_t jtb_n_rows(const Jtb *j);
uint32_t jtb_n_batches(const Jtb *j);
uint32_t jtb_max_tokens_per_batch(const Jtb *j);
uint32_t jtb_max_len(const Jtb *j);
uint32_t jtb_max_opts(const Jtb *j);
// The mapped file (header, table and every section) is jtb_file_size() bytes.
uint64_t jtb_file_size(const Jtb *j);
// Where a section starts in the file and how many bytes it takes. A test checks that the sums of len/n_opts agree
// with the ids/targets section sizes; the trainer reads max_tokens_per_batch rather than walking the batches itself.
uint64_t jtb_section_offset(const Jtb *j, int sec);
uint64_t jtb_section_size(const Jtb *j, int sec);

// Row accessors. i < jtb_n_rows(); jtb_row() returns NULL for an out-of-range index, the others do not check.
const JtbRow *jtb_row(const Jtb *j, uint32_t i);
const uint32_t *jtb_ids(const Jtb *j, uint32_t i);      // jtb_row(j,i)->len ids, each < 248320
const float *jtb_targets(const Jtb *j, uint32_t i);    // jtb_row(j,i)->n_opts probabilities, summing to 1
const char *jtb_id(const Jtb *j, uint32_t i);          // NUL-terminated; e.g. "fx-0000"
// The row's criteria keys (the keys of pred-<name>.jsonl's "probs" object, design §9.2): a pointer per key, at most
// max of them. Returns the key count, or -1 if the row has more than max keys (out is then untouched).
int jtb_keys(const Jtb *j, uint32_t i, const char **out, int max);

// Batch k of the file: a step's rows. *n gets the count, the return value is the row indices (n of them). The batch
// order is prep.py's — epoch by epoch for the train split, input order for eval (design §2.3, §9.1).
const uint32_t *jtb_batch(const Jtb *j, uint32_t k, uint32_t *n);

#endif // JT_JTB_H
