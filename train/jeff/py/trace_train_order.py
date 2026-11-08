#!/usr/bin/env python3
"""trace_train_order.py — prove prep.py's batch order is CLM-8B's train.py's own (design §2.3, plan M5.3).

The design hands the batch order to C because ``batches()`` uses Python's ``random.Random`` and cannot be reproduced
cheaply (design §2.3). This script is the evidence for that handover, and it works the way the plan describes:

1. import CLM-8B's ``tools.jeff_lora.train`` as ``T`` and wrap ``T.batches``, so every call appends the rows it
   returned — the train calls (from ``main()``, ``shuffle=True``) and the dev calls (from ``evaluate()``,
   ``shuffle=False``);
2. run ``T.main([...])`` with the run the plan asks for (``--train-rows 64 --dev-rows 16 --epochs 2 --batch 8
   --max-steps 16``), so two epochs' worth of real ``batches()`` calls are recorded;
3. run ``py/prep.py`` with the same arguments, read its ``train.jtb``/``dev.jtb`` back with this file's own JTB1
   parser (not prep.py's writer), and compare the two lists **call by call**: the row ids, the row order, the per-call
   grouping and every row weight (so ``--weights balanced`` is covered too).

The order is pure Python (``random.Random`` plus the tokenizer's lengths), so the result does not depend on the device
or on where the script runs; on a Mac it can run with CLM-8B's .venv (MPS) instead of gpubox.

usage (CLM-8B's venv: it has torch and transformers):

    ~/Downloads/CLM-8B/.venv/bin/python py/trace_train_order.py \\
        --clm ~/Downloads/CLM-8B \\
        --train ~/Downloads/CLM-8B/data/ovril_detect/out/train.jsonl \\
        --dev ~/Downloads/CLM-8B/data/ovril_detect/out/development.jsonl \\
        --seed 0 --weights none

Exit code 0 only if the trace and prep.py agree; 1 on a mismatch, with the first difference printed.

A note on ``--eval``: the plan's trace covers the train order and the dev order (no ``--eval`` splits). Eval splits
walk the very same ``batches(rows, size, Random(0), shuffle=False)`` path in train.py's ``evaluate()`` that the dev
check does, with ``size = max(2, batch*2)``, so a green dev comparison covers the eval rule as well — but this script
does not re-check it against a real eval split, and does not pretend to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAGIC = b"JTB1"
HEADER_BYTES = 4 + 6 * 4
TABLE_BYTES = 5 * 8
WEIGHT_TOL = 1e-6


# ---- JTB1 reader (design §9.1, src/jtb.h) -------------------------------------------------------------------------
def read_jtb(path: Path) -> dict:
    """Parse a .jtb file the way src/jtb.c does, checking the same invariants. Independent of prep.py's writer."""
    data = Path(path).read_bytes()

    def bad(why: str):
        raise SystemExit(f"trace: {path}: {why}")

    if len(data) < HEADER_BYTES + TABLE_BYTES:
        bad(f"only {len(data)} bytes")
    if data[:4] != MAGIC:
        bad(f"bad magic {data[:4]!r}")
    version, n_rows, n_batches, max_tokens, max_len, max_opts = struct.unpack_from("<6I", data, 4)
    if version != 1:
        bad(f"version {version}, want 1")
    off = struct.unpack_from("<5Q", data, 28)
    if list(off) != sorted(off) or off[0] < HEADER_BYTES + TABLE_BYTES or off[-1] > len(data):
        bad(f"section offsets {off} are not inside the {len(data)}-byte file")
    rows_off, ids_off, tgt_off, bat_off, str_off = off

    rows = []
    ids_at = tgt_at = 0
    for i in range(n_rows):
        ln, n_opts, weight, teacher_top, id_off = struct.unpack_from("<IIfIQ", data, rows_off + 24 * i)
        rows.append({"len": ln, "n_opts": n_opts, "weight": weight, "teacher_top": teacher_top,
                     "id_off": id_off, "ids_at": ids_at, "tgt_at": tgt_at})
        ids_at += ln
        tgt_at += n_opts
    if n_rows and (rows_off + 24 * n_rows != ids_off or 4 * ids_at != tgt_off - ids_off
                   or 4 * tgt_at != bat_off - tgt_off):
        bad("the rows' len/n_opts sums do not fill the ids/targets sections")
    ids = struct.unpack_from(f"<{ids_at}I", data, ids_off) if ids_at else ()
    targets = struct.unpack_from(f"<{tgt_at}f", data, tgt_off) if tgt_at else ()
    for i, r in enumerate(rows):
        r["ids"] = list(ids[r["ids_at"]:r["ids_at"] + r["len"]])
        r["target"] = list(targets[r["tgt_at"]:r["tgt_at"] + r["n_opts"]])
        if r["len"] > max_len or r["n_opts"] > max_opts or r["teacher_top"] >= r["n_opts"]:
            bad(f"row {i}: len/n_opts/teacher_top above the header's {max_len}/{max_opts}")
        if not r["weight"] >= 0.0 or abs(sum(r["target"]) - 1.0) > 1e-3:
            bad(f"row {i}: weight {r['weight']} or targets summing to {sum(r['target'])}")

    strings = data[str_off:]
    for i, r in enumerate(rows):
        at = str_off + r["id_off"]
        if at >= len(data):
            bad(f"row {i}: id_off runs past the strings section")
        end = data.index(b"\0", at)
        rid = data[at:end].decode("utf-8")
        if not rid:
            bad(f"row {i}: empty id")
        at = end + 1
        keys = []
        while data[at:at + 1] != b"\0":
            end = data.index(b"\0", at)
            keys.append(data[at:end].decode("utf-8"))
            at = end + 1
        if len(keys) != r["n_opts"]:
            bad(f"row {i}: {len(keys)} criteria keys, {r['n_opts']} targets")
        r["id"], r["keys"] = rid, keys

    batches, at, seen = [], bat_off, set()
    for k in range(n_batches):
        n = struct.unpack_from("<I", data, at)[0]
        at += 4
        batch = list(struct.unpack_from(f"<{n}I", data, at))
        at += 4 * n
        if not n or n > n_rows or any(j >= n_rows for j in batch) or len(set(batch)) != len(batch):
            bad(f"batch {k}: {batch}")
        if sum(rows[j]["len"] for j in batch) > max_tokens:
            bad(f"batch {k} is above max_tokens_per_batch {max_tokens}")
        seen |= set(batch)
        batches.append(batch)
    if at != str_off:
        bad("the batches section is not walked exactly")
    if n_rows and len(seen) != n_rows:
        bad(f"the batches hold {len(seen)} of {n_rows} rows")
    return {"rows": rows, "batches": batches, "header": {"version": version, "n_rows": n_rows,
            "n_batches": n_batches, "max_tokens_per_batch": max_tokens, "max_len": max_len,
            "max_opts": max_opts}}


# ---- the trace ----------------------------------------------------------------------------------------------------
def trace(T, argv: list[str]) -> list[dict]:
    """Run T.main(argv) with T.batches wrapped; return one record per batches() call, in call order."""
    calls: list[dict] = []
    original = T.batches

    def spy(rows, size, rng, shuffle):
        out = original(rows, size, rng, shuffle)
        # the immediate caller names the split: main() asks for the train order, evaluate() for dev/eval (shuffle=False)
        caller = sys._getframe(1).f_code.co_name
        calls.append({"caller": caller, "size": size, "shuffle": bool(shuffle), "n_rows": len(rows),
                      "batches": [[{"id": str(r.get("id")), "len": len(r["ids"]), "weight": r["weight"],
                                    "teacher_top": r["teacher_top"]} for r in b] for b in out]})
        return out

    T.batches = spy
    try:
        rc = T.main(argv)
    finally:
        T.batches = original
    if rc != 0:
        raise SystemExit(f"trace: train.py main() exited {rc}")
    return calls


def run_prep(a: argparse.Namespace, model: str, out_dir: Path) -> None:
    prep = Path(a.prep).expanduser()
    cmd = [sys.executable, str(prep), "--clm", a.clm, "--model", model, "--out-dir", str(out_dir),
           "--train", a.train, "--dev", a.dev, "--batch", str(a.batch), "--accum", str(a.accum),
           "--epochs", str(a.epochs), "--train-rows", str(a.train_rows), "--dev-rows", str(a.dev_rows),
           "--seed", str(a.seed), "--weights", a.weights, "--weight-cap", str(a.weight_cap),
           "--targets", a.targets]
    print("trace: $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---- comparison ---------------------------------------------------------------------------------------------------
def as_ids(call: dict) -> list[list[str]]:
    return [[row["id"] for row in batch] for batch in call["batches"]]


def compare_split(label: str, traced: list[dict], wanted: list[list[str]], fails: list[str]) -> dict:
    """Compare the traced calls of one split with prep's batch list; the rows must match call by call."""
    n_fails = len(fails)
    flat = [batch for call in traced for batch in as_ids(call)]
    if len(flat) != len(wanted):
        fails.append(f"{label}: {len(flat)} traced batches, prep wrote {len(wanted)}")
        return {"label": label, "batches": len(wanted), "rows": sum(len(b) for b in wanted), "groups": None}
    for k, (got, want) in enumerate(zip(flat, wanted)):
        if got != want:
            where = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            fails.append(f"{label}: batch {k} differs at position {where}: traced {got[where:where + 3]!r}, "
                         f"prep {want[where:where + 3]!r}")
            break
    # the per-call grouping: prep's flat list cut at the traced call boundaries must reproduce every call
    if len(fails) == n_fails:
        at = 0
        for c, call in enumerate(traced):
            n = len(call["batches"])
            if flat[at:at + n] != as_ids(call):
                fails.append(f"{label}: call {c} ({call['caller']}) is not prep's batches {at}..{at + n - 1}")
                break
            at += n
        if at != len(wanted):
            fails.append(f"{label}: the traced calls cover {at} batches, prep wrote {len(wanted)}")
    return {"label": label, "batches": len(wanted), "rows": sum(len(b) for b in wanted), "groups": len(traced)}


def compare_repeated(label: str, traced: list[dict], wanted: list[list[str]], wanted_rows: list[dict],
                     fails: list[str]) -> dict:
    """Dev (and eval) walk the same unshuffled order every time, so *every* call must equal prep's one list."""
    worst = 0.0
    for c, call in enumerate(traced):
        got = as_ids(call)
        if got != wanted:
            at = next((k for k, (x, y) in enumerate(zip(got, wanted)) if x != y), min(len(got), len(wanted)))
            fails.append(f"{label}: call {c} ({call['caller']}) is not prep's list: batch {at} traced "
                         f"{(got[at] if at < len(got) else None)!r}, prep {(wanted[at] if at < len(wanted) else None)!r}")
            break
        worst = max(worst, compare_weights([call], wanted_rows, fails))
    return {"label": label, "batches": len(wanted), "rows": sum(len(b) for b in wanted), "groups": len(traced),
            "worst_weight": worst}


def compare_weights(traced: list[dict], wanted_rows: list[dict], fails: list[str]) -> float:
    """Every traced row's weight against the weight prep wrote for the same id, as f32 (the file rounds it)."""
    prep = {r["id"]: struct.unpack("<f", struct.pack("<f", r["weight"]))[0] for r in wanted_rows}
    worst = 0.0
    for call in traced:
        for batch in call["batches"]:
            for row in batch:
                if row["id"] not in prep:
                    fails.append(f"weights: row {row['id']!r} is in the trace but not in prep's file")
                    return worst
                worst = max(worst, abs(prep[row["id"]] - row["weight"]))
    if worst > WEIGHT_TOL:
        fails.append(f"weights: the largest difference is {worst:.3e}, above {WEIGHT_TOL:.0e}")
    return worst


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compare train.py's batch order with prep.py's (plan M5.3)")
    ap.add_argument("--clm", required=True, help="CLM-8B checkout to import tools.jeff_lora.train from")
    ap.add_argument("--model", default=None, help="Jeff base checkpoint (default: train.py's default_model())")
    ap.add_argument("--train", required=True, help="train split (.jsonl)")
    ap.add_argument("--dev", required=True, help="dev split (.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", choices=("none", "balanced"), default="none")
    ap.add_argument("--weight-cap", type=float, default=4.0)
    ap.add_argument("--targets", choices=("soft", "hard"), default="soft")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-rows", type=int, default=64)
    ap.add_argument("--dev-rows", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=16, help="train.py's smoke-run limit (plan M5.3)")
    ap.add_argument("--device", default="auto", help="train.py's --device (the order does not depend on it)")
    ap.add_argument("--grad-checkpoint", action="store_true", help="train.py's --grad-checkpoint (memory only)")
    ap.add_argument("--mem-cap-gb", default=None, help="train.py's --mem-cap-gb (pass 0 for MPS's default limit)")
    ap.add_argument("--dtype", default="auto", help="train.py's --dtype (the order does not depend on it)")
    ap.add_argument("--work", default="/tmp/j15-trace", help="scratch directory (train.py's --output and the .jtb files)")
    ap.add_argument("--trace-out", default=None, help="where to write the recorded call list as JSON")
    ap.add_argument("--prep", default=str(HERE / "prep.py"), help="the prep.py to compare against (default: this py/)")
    ap.add_argument("--replay", default=None, metavar="TRACE.json",
                    help="compare a saved --trace-out file instead of running train.py again (a few seconds)")
    a = ap.parse_args() if argv is None else ap.parse_args(argv)

    root = Path(a.clm).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import tools.jeff_lora.train as T  # noqa: E402

    model = str(Path(a.model).expanduser().resolve()) if a.model else str(T.default_model())
    work = Path(a.work).expanduser() / f"seed{a.seed}-{a.weights}"
    run_dir, prep_dir, again_dir = work / "run", work / "prep", work / "prep-again"
    shutil.rmtree(work, ignore_errors=True)  # train.py wants a new or empty --output
    run_dir.mkdir(parents=True)

    if a.replay:
        saved = json.loads(Path(a.replay).expanduser().read_text())
        calls = saved["calls"]
        if (saved.get("seed"), saved.get("weights")) != (a.seed, a.weights):
            print(f"trace: warning: {a.replay} was recorded for seed {saved.get('seed')}, weights "
                  f"{saved.get('weights')}; comparing it with --seed {a.seed} --weights {a.weights} anyway", flush=True)
        print(f"trace: replaying {len(calls)} batches() call(s) from {a.replay} "
              f"(train.py is not run again)", flush=True)
    else:
        print(f"trace: train.py main() with seed {a.seed}, weights {a.weights}, {a.train_rows} train rows, "
              f"{a.dev_rows} dev rows, batch {a.batch} x accum {a.accum}, {a.epochs} epoch(s), "
              f"max-steps {a.max_steps}, device/dtype {a.device}/{a.dtype}", flush=True)
        calls = trace(T, ["--model", model, "--train", a.train, "--dev", a.dev, "--output", str(run_dir),
                          "--batch", str(a.batch), "--accum", str(a.accum), "--epochs", str(a.epochs),
                          "--train-rows", str(a.train_rows), "--dev-rows", str(a.dev_rows),
                          "--max-steps", str(a.max_steps), "--seed", str(a.seed), "--weights", a.weights,
                          "--weight-cap", str(a.weight_cap), "--targets", a.targets,
                          "--device", a.device, "--dtype", a.dtype]
                         + (["--grad-checkpoint"] if a.grad_checkpoint else [])
                         + ([] if a.mem_cap_gb is None else ["--mem-cap-gb", str(a.mem_cap_gb)]))
        trace_out = Path(a.trace_out) if a.trace_out else work / "trace.json"
        trace_out.write_text(json.dumps({"seed": a.seed, "weights": a.weights, "calls": calls}, indent=2) + "\n")
        print(f"trace: {len(calls)} batches() call(s) -> {trace_out}", flush=True)

    run_prep(a, model, prep_dir)
    train_jtb, dev_jtb = read_jtb(prep_dir / "train.jtb"), read_jtb(prep_dir / "dev.jtb")

    fails: list[str] = []
    train_calls = [c for c in calls if c["shuffle"]]
    dev_calls = [c for c in calls if not c["shuffle"]]
    if len(train_calls) != a.epochs:
        fails.append(f"train: {len(train_calls)} shuffled batches() call(s), want one per epoch ({a.epochs})")
    if {c["caller"] for c in train_calls} - {"main"}:
        fails.append(f"train: shuffled calls came from {sorted({c['caller'] for c in train_calls})}, want main()")
    if {c["caller"] for c in dev_calls} - {"evaluate"}:
        fails.append(f"dev: unshuffled calls came from {sorted({c['caller'] for c in dev_calls})}, want evaluate()")

    train_ids = [[train_jtb["rows"][i]["id"] for i in b] for b in train_jtb["batches"]]
    dev_ids = [[dev_jtb["rows"][i]["id"] for i in b] for b in dev_jtb["batches"]]
    tstat = compare_split("train", train_calls, train_ids, fails)
    tstat["worst_weight"] = compare_weights(train_calls, train_jtb["rows"], fails)
    dstat = compare_repeated("dev", dev_calls, dev_ids, dev_jtb["rows"], fails)

    # prep.py must be deterministic: the same arguments write the same bytes
    run_prep(a, model, again_dir)
    for name in ("train", "dev"):
        first, again = sha256(prep_dir / f"{name}.jtb"), sha256(again_dir / f"{name}.jtb")
        if first != again:
            fails.append(f"{name}: prep.py is not deterministic ({first[:12]} != {again[:12]})")

    print(f"trace: train.jtb header {train_jtb['header']}", flush=True)
    print(f"trace: dev.jtb   header {dev_jtb['header']}", flush=True)
    print(f"trace: train: {tstat['batches']} batches in {tstat['groups']} call(s), {tstat['rows']} rows, "
          f"max |dweight| {tstat['worst_weight']:.1e}", flush=True)
    print(f"trace: dev:   {dstat['batches']} batch in prep, {dstat['groups']} evaluate() call(s), "
          f"{dstat['rows']} rows, max |dweight| {dstat['worst_weight']:.1e}", flush=True)
    if fails:
        for line in fails:
            print(f"trace: FAIL {line}", flush=True)
        print(f"trace: MISMATCH seed {a.seed} weights {a.weights}", flush=True)
        return 1
    print(f"trace: the batch order matches call by call: seed {a.seed}, weights {a.weights}  -> PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
