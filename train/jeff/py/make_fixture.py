#!/usr/bin/env python3
"""make_fixture.py — write a synthetic .jtb file, the fixture every test that reads a .jtb uses.

The format is design §9.1 (also written down in train/jeff/src/jtb.h):

    magic "JTB1" | u32 version=1 | u32 n_rows | u32 n_batches | u32 max_tokens_per_batch | u32 max_len | u32 max_opts
    u64 section_offsets[5]                     (rows, ids, targets, batches, strings; from the start of the file)
    rows:    n_rows x { u32 len, u32 n_opts, f32 weight, u32 teacher_top, u64 id_offset_in_strings }
    ids:     sum(len) u32 token ids, row by row
    targets: sum(n_opts) f32, row by row
    batches: n_batches x { u32 n, u32 row_index[n] }
    strings: per row: the id, NUL, its criteria keys each NUL-terminated, then an empty string (a final NUL)

Everything is random from --seed: token ids below 248320 (the tokenizer's vocab), lengths in [--min-len, --max-len],
n_opts in [--min-opts, --max-opts], softmax-normalised targets, weights 1.0, ids "fx-0000"…, criteria keys "A", "B", …
Batches are consecutive chunks of --batch rows, in row order, no shuffle: the real epoch order is prep.py's job (M5).

**No real data** (plan rule 5): this is the only .jtb writer until prep.py, and it never reads CLM-8B's data/.
Real data is not even representable here — synthetic ids and random targets only.

`--lengths` and `--opts` pin the per-row values, which the later reference fixtures need:
    --rows 2 --lengths 37,70 --opts 5,12 --batch 2     # design §10.2's two rows (dump_ref.py, M4.1)
    --rows 8 --seed 1                                  # make test-cpu's fixture

usage:
    python3 py/make_fixture.py --seed 1 --rows 8 --out build/cpu/fx.jtb
    python3 py/make_fixture.py --seed 7 --rows 2 --lengths 37,70 --opts 5,12 --batch 2 --out /tmp/fx.jtb
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import sys

MAGIC = b"JTB1"
VERSION = 1
VOCAB = 248320  # token ids must be below this (design §9.1)
MAX_OPTS = 255
SECTION_COUNT = 5
HEADER_BYTES = 4 + 6 * 4
TABLE_BYTES = SECTION_COUNT * 8


def criteria_key(i: int) -> str:
    """0 -> "A", 25 -> "Z", 26 -> "AA": the row's criteria keys, in the order the target lists them (§9.2)."""
    name = ""
    i += 1
    while i > 0:
        i, rem = divmod(i - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def int_list(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"want comma-separated integers, got {text!r}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="write a synthetic JTB1 fixture (design §9.1)")
    ap.add_argument("--seed", type=int, default=1, help="random seed (default 1); the same seed gives the same file")
    ap.add_argument("--rows", type=int, default=None,
                    help="number of rows (default 8, or the number of --lengths/--opts values)")
    ap.add_argument("--out", required=True, help="where to write the .jtb file")
    ap.add_argument("--min-len", type=int, default=8, help="shortest random row, in tokens (default 8)")
    ap.add_argument("--max-len", type=int, default=120, help="longest random row, in tokens (default 120)")
    ap.add_argument("--min-opts", type=int, default=2, help="fewest random options (default 2)")
    ap.add_argument("--max-opts", type=int, default=6, help="most random options (default 6)")
    ap.add_argument("--batch", type=int, default=4, help="rows per batch (default 4)")
    ap.add_argument("--lengths", type=int_list, default=None,
                    help="explicit per-row lengths, e.g. 37,70 (overrides --min-len/--max-len)")
    ap.add_argument("--opts", type=int_list, default=None,
                    help="explicit per-row option counts, e.g. 5,12 (overrides --min-opts/--max-opts)")
    args = ap.parse_args(argv)

    if args.lengths is not None:
        if args.rows is not None and args.rows != len(args.lengths):
            ap.error(f"--rows {args.rows} but --lengths has {len(args.lengths)} values")
        args.rows = len(args.lengths)
    if args.opts is not None:
        if args.rows is not None and args.rows != len(args.opts):
            ap.error(f"--rows {args.rows} but --opts has {len(args.opts)} values")
        args.rows = len(args.opts)
    if args.rows is None:
        args.rows = 8
    if args.rows < 1:
        ap.error("--rows must be at least 1")
    if args.batch < 1:
        ap.error("--batch must be at least 1")
    if args.min_len < 1 or args.min_len > args.max_len:
        ap.error(f"want 1 <= --min-len <= --max-len, got {args.min_len} and {args.max_len}")
    if args.min_opts < 1 or args.min_opts > args.max_opts or args.max_opts > MAX_OPTS:
        ap.error(f"want 1 <= --min-opts <= --max-opts <= {MAX_OPTS}, "
                 f"got {args.min_opts} and {args.max_opts}")
    for i, n in enumerate(args.opts or []):
        if not 1 <= n <= MAX_OPTS:
            ap.error(f"--opts row {i} is {n}, want 1..{MAX_OPTS}")
    for i, n in enumerate(args.lengths or []):
        if n < 1:
            ap.error(f"--lengths row {i} is {n}, want at least 1")
    return args


def make_rows(args: argparse.Namespace, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(args.rows):
        length = args.lengths[i] if args.lengths is not None else rng.randint(args.min_len, args.max_len)
        n_opts = args.opts[i] if args.opts is not None else rng.randint(args.min_opts, args.max_opts)
        # A softmax of random scores, so the target is a distribution with a clear argmax (the teacher's top option).
        raw = [rng.random() for _ in range(n_opts)]
        top = max(raw)
        exps = [math.exp(u - top) for u in raw]
        total = sum(exps)
        target = [e / total for e in exps]
        rows.append({
            "id": f"fx-{i:04d}",
            "keys": [criteria_key(k) for k in range(n_opts)],
            "len": length,
            "n_opts": n_opts,
            "weight": 1.0,
            "target": target,
            "teacher_top": max(range(n_opts), key=lambda k: target[k]),
            "ids": [rng.randrange(VOCAB) for _ in range(length)],
        })
    return rows


def make_batches(rows: list[dict], size: int) -> list[list[int]]:
    return [list(range(start, min(start + size, len(rows)))) for start in range(0, len(rows), size)]


def serialize(rows: list[dict], batches: list[list[int]]) -> bytes:
    rows_blob = bytearray()
    ids_blob = bytearray()
    targets_blob = bytearray()
    batches_blob = bytearray()
    strings_blob = bytearray()
    for r in rows:
        # the strings section: id NUL, each key NUL-terminated, then the empty string that ends the list (§9.2)
        id_off = len(strings_blob)
        strings_blob += r["id"].encode("utf-8") + b"\0"
        for key in r["keys"]:
            strings_blob += key.encode("utf-8") + b"\0"
        strings_blob += b"\0"
        rows_blob += struct.pack(
            "<IIfIQ", r["len"], r["n_opts"], r["weight"], r["teacher_top"], id_off)
        ids_blob += struct.pack(f"<{r['len']}I", *r["ids"])
        targets_blob += struct.pack(f"<{r['n_opts']}f", *r["target"])
    for batch in batches:
        batches_blob += struct.pack(f"<I{len(batch)}I", len(batch), *batch)

    sections = [rows_blob, ids_blob, targets_blob, batches_blob, strings_blob]
    offsets = []
    at = HEADER_BYTES + TABLE_BYTES
    for blob in sections:
        offsets.append(at)
        at += len(blob)

    # the header's maxima: what the trainer sizes its buffers and checkpoint count from (design §2, §6.2)
    max_len = max(r["len"] for r in rows)
    max_opts = max(r["n_opts"] for r in rows)
    max_tokens = max(sum(rows[i]["len"] for i in batch) for batch in batches)
    header = MAGIC + struct.pack("<6I", VERSION, len(rows), len(batches), max_tokens, max_len, max_opts)
    return header + struct.pack(f"<{SECTION_COUNT}Q", *offsets) + b"".join(bytes(b) for b in sections)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    rows = make_rows(args, rng)
    batches = make_batches(rows, args.batch)
    data = serialize(rows, batches)
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"make_fixture: wrote {args.out}: {len(rows)} rows, {len(batches)} batches, "
          f"max_len {max(r['len'] for r in rows)}, max_opts {max(r['n_opts'] for r in rows)}, "
          f"max_tokens_per_batch {max(sum(rows[i]['len'] for i in b) for b in batches)}, "
          f"{len(data)} bytes, seed {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
