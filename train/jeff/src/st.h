// src/st.h — safetensors: an mmap'ed reader and an F32 writer (design §3.2 and §9.2, plan M1.3).
//
// The format is an 8-byte little-endian header length, a JSON header of the shape
//   "name": {"dtype": "F32", "shape": [2,3], "data_offsets": [a,b]}
// (plus an optional "__metadata__" object, which the reader skips), then the raw little-endian tensor data starting at
// offset 8 + header_length. data_offsets are relative to that start, so a bf16 tensor is just its bytes in the file: the
// reader returns a pointer into the mapping and copies nothing.
//
// Two callers matter and neither may guess: st_get takes the dtype and the full shape the caller expects, so a layout
// mistake ("q_proj is [4096,1024]" read as [1024,4096]) fails at load time with a message that names both, instead of
// producing quietly wrong numbers. The writer is what py/compare_layers.py reads back (plan M4.1), so its output must be
// what Python's safetensors accepts: keys sorted, a header padded with spaces to a multiple of 8, no metadata.
//
// Errors are fatal (exit 1 with a message on stderr), like CK()/CB(): a checkpoint that does not parse or does not have
// the tensor the caller asked for is not something any caller can recover from.
#ifndef JT_ST_H
#define JT_ST_H

#include <stddef.h>

// A shape is stored in a fixed array so StTensor is plain data. The model's 5-dim visual.patch_embed.proj.weight is the
// widest entry in the checkpoint the plan targets (design §3.2); 8 leaves room without a second allocation.
#define ST_MAX_DIMS 8

// Every dtype name safetensors defines. ST_DTYPE_NONE means "not a tensor dtype" / "unset"; nothing else has a 0 value
// so a zeroed StTensor never looks like a real F32.
typedef enum {
  ST_DTYPE_NONE = 0,
  ST_DTYPE_BOOL,
  ST_DTYPE_U8,
  ST_DTYPE_I8,
  ST_DTYPE_I16,
  ST_DTYPE_U16,
  ST_DTYPE_F16,
  ST_DTYPE_BF16,
  ST_DTYPE_I32,
  ST_DTYPE_U32,
  ST_DTYPE_F32,
  ST_DTYPE_I64,
  ST_DTYPE_U64,
  ST_DTYPE_F64,
} StDtype;

size_t st_dtype_size(StDtype dtype);      // bytes per element; 0 for ST_DTYPE_NONE
const char *st_dtype_name(StDtype dtype); // "F32", "BF16", … ("?" for ST_DTYPE_NONE)
StDtype st_dtype_parse(const char *name); // the name in the header; ST_DTYPE_NONE if the file says something else

// One tensor of a file. name/file point into memory the St owns (freed by st_close); data points into the mapping and is
// valid until st_close. file is the path it came from, for messages.
typedef struct {
  const char *name;
  StDtype dtype;
  int ndim;
  int shape[ST_MAX_DIMS];
  const void *data;
  size_t nbytes;
  const char *file;
} StTensor;

typedef struct St St; // opaque: a set of opened files and their tensors

// Open one file, or a model directory: a directory resolves to model.safetensors, or, when model.safetensors.index.json
// exists, to every shard its weight_map names (design §3.2, plan M1.3). A missing shard or a weight_map entry that its
// shard does not contain is fatal, so a half-copied checkpoint cannot be read as a complete one.
St *st_open(const char *path);
void st_close(St *st);

size_t st_count(const St *st);                   // number of tensors, in header order across the opened files
const StTensor *st_tensor(const St *st, size_t i); // NULL if i >= st_count
const StTensor *st_find(const St *st, const char *name); // NULL when the name is absent

// The tensor's bytes, after checking dtype, ndim and every dim against what the caller expects. Returns that pointer
// plus .data (== the returned value); dies with a message naming the tensor, what it is and what was expected.
const void *st_get(const St *st, const char *name, StDtype dtype, int ndim, ...);
const void *st_getv(const St *st, const char *name, StDtype dtype, int ndim, const int *shape);

// Write n F32 tensors, keys sorted, at most ST_MAX_DIMS dims each: the design §9.2 recipe. shapes[i] holds ndim[i] dims
// (the rest of the row is ignored), data[i] points at ndim[i] dims of float. Duplicate names, a name that would need
// JSON escaping, and a size that disagrees with the shape are fatal.
void st_write(const char *path, int n, const char *const names[], const int shapes[][ST_MAX_DIMS],
              const int ndims[], const float *const data[]);

#endif // JT_ST_H
