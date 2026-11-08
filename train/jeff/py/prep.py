#!/usr/bin/env python3
"""prep.py — tokenise CLM-8B's splits and write the .jtb files the C trainer reads (design §9.1, §2.3; plan M5.3).

The whole point of this script is that it does **not** reimplement anything: it imports CLM-8B's
``tools.jeff_lora.train`` (``read_jsonl``, ``encode``, ``balance``, ``batches``, ``default_model``) and
``tools.jeff_lora.prompt`` (``load_decision_config``) by path, so the token ids, the row weights and the batch order
are the Python trainer's own (design §9.1: "It never copies their code").

``train.py``'s ``batches()`` uses Python's ``random.Random(seed)``, so its order cannot be reproduced cheaply in C:
the order is made here and handed over (design §2.3), and jefftrain consumes the batches in file order.

The RNG sequence of ``train.py``'s ``main()``, reproduced call for call (plan M5.3):

1. ``train_rows = read_jsonl(--train)``, then ``random.Random(seed).sample(train_rows, --train-rows)``;
2. ``dev_rows = read_jsonl(--dev)``, then ``random.Random(seed + 1).sample(dev_rows, --dev-rows)``;
3. ``encode()`` the train split (``hard = --targets hard``) and the dev split (always soft: selection is against the
   teacher's distribution), then ``balance()`` the train rows if ``--weights balanced``;
4. ``rng = random.Random(seed)``, consumed **only** by ``batches(train, --batch, rng, shuffle=True)``, called once
   per epoch, in epoch order — all epochs' batch lists go in the train file, epoch after epoch;
5. dev and eval splits use ``batches(rows, size, random.Random(0), shuffle=False)`` with ``size = --batch * 2`` for
   dev and ``max(2, --batch * 2)`` for ``--eval``.

The file layout is design §9.1 as written down in ``src/jtb.h`` (see py/make_fixture.py for the same format):

    magic "JTB1" | u32 version=1 | u32 n_rows | u32 n_batches | u32 max_tokens_per_batch | u32 max_len | u32 max_opts
    u64 section_offsets[5]                     (rows, ids, targets, batches, strings; from the start of the file)
    rows:    n_rows x { u32 len, u32 n_opts, f32 weight, u32 teacher_top, u64 id_offset_in_strings }
    ids:     sum(len) u32 token ids, row by row
    targets: sum(n_opts) f32, row by row
    batches: n_batches x { u32 n, u32 row_index[n] }
    strings: per row: the id, NUL, its criteria keys each NUL-terminated, then an empty string (a final NUL)

``id_offset_in_strings`` is relative to the start of the strings section and the header's three maxima are the file's
actual maxima (M1.4's decisions, recorded in src/jtb.h and notes/J05.md).

Next to each ``<name>.jtb`` this writes ``<name>.prep.json`` with the arguments that shaped it (batch, accum, epochs,
seed, weights, source). The JTB1 header has no room for those, and plan M5.2 wants jefftrain to reject a flag
combination that disagrees with the file, so the sidecar is where it can look.

**Real data never enters macuda** (plan rule 5): the .jtb files go to CLM-8B's git-ignored ``data/`` or to /tmp.

usage (CLM-8B's venv has torch and transformers; no GPU and no model weights are needed, only the tokenizer):

    ~/Downloads/CLM-8B/.venv/bin/python py/prep.py --clm ~/Downloads/CLM-8B \\
        --train ~/Downloads/CLM-8B/data/ovril_detect/out/train.jsonl \\
        --dev ~/Downloads/CLM-8B/data/ovril_detect/out/development.jsonl \\
        --batch 8 --accum 1 --epochs 2 --train-rows 64 --dev-rows 16 --weights balanced --seed 0 \\
        --out-dir /tmp/jeff-prep

    # eval only (plan M6.1): writes test.jtb
    ... py/prep.py --clm ~/CLM-8B --eval test=~/CLM-8B/data/ovril_detect/out/test.jsonl --batch 4 --out-dir /tmp/p
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

MAGIC = b"JTB1"
VERSION = 1
MAX_OPTS = 255
VOCAB = 248320  # token ids must be below this (design §9.1, src/jtb.h's JTB_VOCAB_MAX)
SECTION_COUNT = 5
HEADER_BYTES = 4 + 6 * 4
TABLE_BYTES = SECTION_COUNT * 8
ROW_BYTES = 24
TARGET_SUM_TOL = 1e-3  # what src/jtb.c accepts; encode() itself raises above 1e-4


def import_train(clm: str):
    """Import CLM-8B's tools.jeff_lora.train (it pulls in prompt.py and lora.py). Never a copy of their code."""
    root = Path(clm).expanduser().resolve()
    if not (root / "tools" / "jeff_lora" / "train.py").is_file():
        raise SystemExit(f"prep: {root} has no tools/jeff_lora/train.py; pass --clm /path/to/CLM-8B")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import tools.jeff_lora.train as T  # noqa: E402  (imported by path, design §9.1)

    return T


def read_and_sample(T, path: Path, want: int, seed: int) -> list[dict]:
    """read_jsonl then the fixed subset: ``random.Random(seed).sample(...)``, as train.py's main() does."""
    rows = T.read_jsonl(Path(path).expanduser())
    if want:
        rows = random.Random(seed).sample(rows, min(want, len(rows)))
    return rows


def batches_of_indices(rows: list[dict], groups: list[list[dict]]) -> list[list[int]]:
    """Map batches()' row dicts back to indices into ``rows``. batches() returns the same objects it was given."""
    index_of = {id(r): i for i, r in enumerate(rows)}
    out = []
    for group in groups:
        for r in group:
            if id(r) not in index_of:
                raise SystemExit("prep: batches() returned a row that is not from the list it was given")
        out.append([index_of[id(r)] for r in group])
    return out


def row_id(r: dict, i: int) -> str:
    """A row's id, the key predictions are written under (design §9.2). A row without one gets a placeholder."""
    rid = r.get("id")
    return f"row-{i}" if rid is None else str(rid)


def check_rows(rows: list[dict], label: str) -> None:
    """The invariants src/jtb.c enforces, checked here so a bad split names itself instead of the reader."""
    if not rows:
        raise SystemExit(f"prep: {label}: no rows (the .jtb reader needs max_len > 0)")
    seen = set()
    for i, r in enumerate(rows):
        n_opts = len(r["target"])
        if n_opts < 1 or n_opts > MAX_OPTS:
            raise SystemExit(f"prep: {label} row {i}: {n_opts} options, want 1..{MAX_OPTS}")
        if len(r["keys"]) != n_opts:
            raise SystemExit(f"prep: {label} row {i}: {len(r['keys'])} criteria keys, {n_opts} targets")
        if not r["ids"]:
            raise SystemExit(f"prep: {label} row {i}: no token ids")
        for t in r["ids"]:
            if not 0 <= t < VOCAB:
                raise SystemExit(f"prep: {label} row {i}: token id {t} is not below {VOCAB}")
        total = sum(r["target"])
        if not abs(total - 1.0) <= TARGET_SUM_TOL:
            raise SystemExit(f"prep: {label} row {i}: targets sum to {total!r}, want 1")
        if not r["weight"] >= 0.0:
            raise SystemExit(f"prep: {label} row {i}: weight {r['weight']!r} is not non-negative")
        if not 0 <= r["teacher_top"] < n_opts:
            raise SystemExit(f"prep: {label} row {i}: teacher_top {r['teacher_top']} is not < {n_opts}")
        rid = row_id(r, i)
        if rid in seen:
            print(f"prep: warning: {label}: duplicate row id {rid!r} (predictions are keyed by it)", file=sys.stderr)
        seen.add(rid)


def serialize(rows: list[dict], batches: list[list[int]]) -> bytes:
    """The JTB1 bytes: header, section table, rows, ids, targets, batches, strings (design §9.1, src/jtb.h)."""
    rows_blob = bytearray()
    ids_blob = bytearray()
    targets_blob = bytearray()
    batches_blob = bytearray()
    strings_blob = bytearray()
    for i, r in enumerate(rows):
        id_off = len(strings_blob)
        strings_blob += row_id(r, i).encode("utf-8") + b"\0"
        for key in r["keys"]:
            strings_blob += str(key).encode("utf-8") + b"\0"
        strings_blob += b"\0"  # the empty string that ends the key list (§9.2)
        rows_blob += struct.pack("<IIfIQ", len(r["ids"]), len(r["target"]), float(r["weight"]),
                                 int(r["teacher_top"]), id_off)
        ids_blob += struct.pack(f"<{len(r['ids'])}I", *r["ids"])
        targets_blob += struct.pack(f"<{len(r['target'])}f", *r["target"])
    for batch in batches:
        batches_blob += struct.pack(f"<I{len(batch)}I", len(batch), *batch)

    sections = [rows_blob, ids_blob, targets_blob, batches_blob, strings_blob]
    offsets = []
    at = HEADER_BYTES + TABLE_BYTES
    for blob in sections:
        offsets.append(at)
        at += len(blob)

    # the actual maxima: the trainer sizes its buffers and the arena from them (design §2, §6.2, src/jtb.h)
    max_len = max(len(r["ids"]) for r in rows)
    max_opts = max(len(r["target"]) for r in rows)
    max_tokens = max(sum(len(rows[i]["ids"]) for i in batch) for batch in batches)
    header = MAGIC + struct.pack("<6I", VERSION, len(rows), len(batches), max_tokens, max_len, max_opts)
    return header + struct.pack(f"<{SECTION_COUNT}Q", *offsets) + b"".join(bytes(b) for b in sections)


def write_split(out_dir: Path, name: str, rows: list[dict], batches: list[list[int]], args: dict) -> dict:
    """Write <name>.jtb and <name>.prep.json; return the summary printed and stored in the sidecar."""
    check_rows(rows, name)
    if not batches:
        raise SystemExit(f"prep: {name}: no batches")
    data = serialize(rows, batches)
    path = out_dir / f"{name}.jtb"
    path.write_bytes(data)
    summary = {
        "split": name,
        "file": str(path),
        "bytes": len(data),
        "rows": len(rows),
        "batches": len(batches),
        "max_len": max(len(r["ids"]) for r in rows),
        "max_opts": max(len(r["target"]) for r in rows),
        "max_tokens_per_batch": max(sum(len(rows[i]["ids"]) for i in b) for b in batches),
        "tokens": sum(len(r["ids"]) for r in rows),
        "rows_per_batch": [len(b) for b in batches],
        **args,
    }
    (out_dir / f"{name}.prep.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"prep: {path.name}: {summary['rows']} rows, {summary['batches']} batches, "
          f"max_len {summary['max_len']}, max_opts {summary['max_opts']}, "
          f"max_tokens_per_batch {summary['max_tokens_per_batch']}, {summary['bytes']} bytes", flush=True)
    return summary


def split_spec(text: str, ap: argparse.ArgumentParser) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        ap.error(f"want NAME=PATH, got {text!r}")
    return name, Path(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="write the .jtb files from CLM-8B's jsonl splits (design §9.1)")
    ap.add_argument("--clm", required=True, help="CLM-8B checkout to import tools.jeff_lora from")
    ap.add_argument("--model", default=None, help="Jeff base checkpoint (default: train.py's default_model())")
    ap.add_argument("--train", default=None, help="train split (.jsonl)")
    ap.add_argument("--dev", default=None, help="dev/selection split (.jsonl)")
    ap.add_argument("--eval", action="append", default=[], metavar="NAME=PATH",
                    help="a split scored once at the end; repeatable")
    ap.add_argument("--out-dir", required=True, help="where the .jtb files go (CLM-8B's data/ or /tmp, never macuda)")
    ap.add_argument("--targets", choices=("soft", "hard"), default="soft")
    ap.add_argument("--weights", choices=("none", "balanced"), default="none")
    ap.add_argument("--weight-cap", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation (recorded only: it does not move rows)")
    ap.add_argument("--epochs", type=int, default=1, help="train epochs; every epoch's batch list is written")
    ap.add_argument("--train-rows", type=int, default=0, help="fixed random subset of train (train.py's flag)")
    ap.add_argument("--dev-rows", type=int, default=0, help="fixed random subset of dev (train.py's flag)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.train and not a.dev and not a.eval:
        ap.error("nothing to do: pass --train, --dev or --eval")
    if a.batch < 1:
        ap.error("--batch must be at least 1")
    if a.accum < 1:
        ap.error("--accum must be at least 1")
    if a.epochs < 1:
        ap.error("--epochs must be at least 1")
    if a.weight_cap <= 0:
        ap.error("--weight-cap must be positive")
    a.evals = [split_spec(s, ap) for s in a.eval]
    names = [name for name, _ in a.evals]
    if "train" in names or "dev" in names or len(set(names)) != len(names):
        ap.error("--eval names must be distinct and must not be 'train' or 'dev'")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    T = import_train(a.clm)
    model = Path(a.model).expanduser().resolve() if a.model else T.default_model()
    decision = T.load_decision_config(model)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model))
    hard = a.targets == "hard"

    # 1-2. the fixed subsets, each with its own Random (train.py's main())
    train_rows = read_and_sample(T, a.train, a.train_rows, a.seed) if a.train else None
    dev_rows = read_and_sample(T, a.dev, a.dev_rows, a.seed + 1) if a.dev else None
    # 3. encode() the same way (dev is always soft), then the class-balanced weights
    train = T.encode(train_rows, decision, tok, hard) if train_rows is not None else None
    dev = T.encode(dev_rows, decision, tok, False) if dev_rows is not None else None
    weight_table = T.balance(train, a.weight_cap) if (train is not None and a.weights == "balanced") else None
    evals = {name: T.encode(T.read_jsonl(path.expanduser()), decision, tok, False) for name, path in a.evals}
    print(f"prep: model {model}  train {0 if train is None else len(train)}  dev {0 if dev is None else len(dev)}  "
          f"eval {[f'{k}:{len(v)}' for k, v in evals.items()]}  targets {a.targets}  weights {a.weights}", flush=True)
    if weight_table:
        print("prep: row weights by teacher argmax:", weight_table, flush=True)

    out_dir = Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"batch": a.batch, "accum": a.accum, "epochs": None, "seed": a.seed, "targets": a.targets,
              "weights": a.weights, "weight_cap": a.weight_cap, "train_rows": a.train_rows,
              "dev_rows": a.dev_rows, "row_weights": weight_table}

    # 4. train: one rng, one batches() call per epoch, all epochs in one file (design §2.3, §9.1)
    if train is not None:
        rng = random.Random(a.seed)  # consumed only by batches(), starting with epoch 1
        groups = []
        for _epoch in range(a.epochs):
            groups += T.batches(train, a.batch, rng, shuffle=True)
        write_split(out_dir, "train", train, batches_of_indices(train, groups),
                    {**common, "source": str(a.train), "epochs": a.epochs})
    # 5. dev and eval: batches(rows, size, Random(0), shuffle=False) — the order evaluate() walks
    if dev is not None:
        groups = T.batches(dev, a.batch * 2, random.Random(0), shuffle=False)
        write_split(out_dir, "dev", dev, batches_of_indices(dev, groups),
                    {**common, "source": str(a.dev), "eval_batch": a.batch * 2})
    for name, path in a.evals:
        size = max(2, a.batch * 2)
        groups = T.batches(evals[name], size, random.Random(0), shuffle=False)
        write_split(out_dir, name, evals[name], batches_of_indices(evals[name], groups),
                    {**common, "source": str(path), "eval_batch": size})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
