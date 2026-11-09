#!/usr/bin/env python3
"""compare_steps.py — plan M6.2's gate (design §10.3.2): per-step loss parity, C against Python.

    py/compare_steps.py py.json c.jsonl

``py.json`` is ``py/trace_train_loss.py``'s trace (one record per optimizer step, each with its per-row soft CE), and
``c.jsonl`` is jefftrain's ``--trace-loss`` file (``{"step", "row", "ce"}`` per line, plan M5.2). Both carry the same
rows of the same prep.py batches, so per step the quantity compared is the **mean of the per-row CE** — the number the
log line's ``loss`` is built from — and the gate is design §10.3.2 as the plan restates it:

* ``|loss_py - loss_c| / loss_py < 1e-3`` for each of steps 1-5;
* the same ratio ``< 1e-2`` through step 20;
* the dev soft CE at step 20 within ``2e-3`` (absolute, in CE units: the gate's number is a difference of losses, so a
  relative reading would let a 1.13 loss be off by more than the gate is worth).

The dev soft CE is not in either trace: it is the run's own ``check()`` (train.py's ``evaluate()``, soft CE at T=1). The
Python side is ``py.json``'s ``dev`` list (read from train.py's ``report.json``) and the C side is jefftrain's
``report.json`` ``history``, found next to ``c.jsonl`` or in the run's ``c/`` directory, or named with ``--c-report``.
Both report.json files round ``dev_soft_ce`` to 5 decimals, which is 400x inside the 2e-3 gate.

The rows-per-step count is checked too: a step whose record holds a different number of rows means the two runs did not
walk the same batches (the same rows in the same order is prep.py's job, plan M5.3), and a different set of rows would
make the CE comparison meaningless — so that is a failure, not a note.

usage:
    py/compare_steps.py /path/py.json /path/steps.jsonl
    py/compare_steps.py /path/py.json /path/steps.jsonl --c-report /path/c-out/report.json

Exit code 0 only if every step and the dev check pass. Tolerances are the plan's; ``--rel-early/--rel-late/--dev-abs``
exist to make them visible (and to let a later, tighter step re-gate without editing the numbers here), not to loosen
them: the defaults are the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EARLY_STEPS = 5          # design §10.3.2: steps 1-5 ...
REL_EARLY = 1e-3         # ... within 1e-3 relative
LATE_STEP = 20           # ... through step 20 ...
REL_LATE = 1e-2          # ... within 1e-2
DEV_ABS = 2e-3           # ... and the dev soft CE at step 20 within 2e-3


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="design §10.3.2's step-parity gate: Python's trace vs jefftrain's")
    ap.add_argument("py", type=Path, help="py/trace_train_loss.py's trace (py.json)")
    ap.add_argument("c", type=Path, help="jefftrain --trace-loss's JSON lines (steps.jsonl)")
    ap.add_argument("--c-report", type=Path, default=None,
                    help="jefftrain's report.json for the dev soft CE (default: next to the trace)")
    ap.add_argument("--rel-early", type=float, default=REL_EARLY, help=f"gate for steps 1-{EARLY_STEPS} (default {REL_EARLY:g})")
    ap.add_argument("--early", type=int, default=EARLY_STEPS, help=f"last step of the tight gate (default {EARLY_STEPS})")
    ap.add_argument("--rel-late", type=float, default=REL_LATE, help=f"gate through the last step (default {REL_LATE:g})")
    ap.add_argument("--late", type=int, default=LATE_STEP, help=f"the step the gate must reach (default {LATE_STEP})")
    ap.add_argument("--dev-abs", type=float, default=DEV_ABS, help=f"dev soft CE gate at step {LATE_STEP} (default {DEV_ABS:g})")
    return ap.parse_args(argv)


def load_py(path: Path) -> tuple[list[dict], list[dict]]:
    """The Python trace: its steps (oldest first) and its dev-check history entries."""
    doc = json.loads(path.read_text())
    steps = doc["steps"] if isinstance(doc, dict) else doc
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"compare_steps: {path} has no steps")
    out = []
    for i, s in enumerate(steps, 1):
        ce = [float(v) for v in s["ce"]]
        if not ce:
            raise SystemExit(f"compare_steps: {path}: step {s.get('step', i)} has no per-row CE")
        out.append({"step": int(s.get("step", i)), "ce": ce})
    dev = doc.get("dev", []) if isinstance(doc, dict) else []
    return out, dev


def load_c(path: Path) -> list[dict]:
    """jefftrain's --trace-loss: one JSON object per line (src/main.c's trace_step())."""
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"compare_steps: {path}:{n}: not JSON ({e})")
        ce = [float(v) for v in rec["ce"]]
        out.append({"step": int(rec["step"]), "ce": ce, "rows": len(rec.get("row", ce))})
    if not out:
        raise SystemExit(f"compare_steps: {path} has no trace records")
    return out


def find_c_report(trace: Path, given: Path | None) -> Path | None:
    """jefftrain's report.json for the dev soft CE: --c-report, else beside the trace, else the run's c/ directory."""
    if given is not None:
        return given
    for cand in (trace.parent / "report.json", trace.parent / "c" / "report.json"):
        if cand.is_file():
            return cand
    found = sorted(trace.parent.glob("*/report.json"))  # a run directory with one trainer output subdirectory
    return found[0] if len(found) == 1 else None


def dev_at(dev: list[dict], step: int) -> float | None:
    """The dev soft CE at `step`: the last check at that step (train.py's check("step"), then the end-of-run one)."""
    vals = [float(d["dev_soft_ce"]) for d in dev if int(d["step"]) == step and d.get("dev_soft_ce") is not None]
    return vals[-1] if vals else None


def main(argv: list[str]) -> int:
    a = parse_args(argv)
    py, py_dev = load_py(a.py)
    c = load_c(a.c)

    failed = 0
    print(f"compare_steps: {a.py} ({len(py)} steps) vs {a.c} ({len(c)} steps)")
    print(f"compare_steps: gate: |dmean|/mean_py < {a.rel_early:g} for steps 1-{a.early}, "
          f"< {a.rel_late:g} through step {a.late}, dev soft CE within {a.dev_abs:g} at step {a.late}")
    print(f"{'step':>4}  {'mean_py':>10}  {'mean_c':>10}  {'rel':>9}  {'gate':>8}  rows  verdict")

    worst = (0.0, 0)
    for i, (p, q) in enumerate(zip(py, c)):
        step = p["step"]
        mp, mc = sum(p["ce"]) / len(p["ce"]), sum(q["ce"]) / len(q["ce"])
        tol = a.rel_early if step <= a.early else a.rel_late
        rel = abs(mp - mc) / abs(mp) if mp != 0.0 else abs(mp - mc)
        ok = rel < tol and len(p["ce"]) == len(q["ce"]) and q["rows"] == len(q["ce"])
        if rel > worst[0]:
            worst = (rel, step)
        if not ok:
            failed += 1
        rows = f"{len(p['ce'])}/{q['rows']}"
        why = "" if ok else ("  rows differ" if not (len(p["ce"]) == len(q["ce"]) == q["rows"]) else "  FAIL")
        print(f"{step:>4}  {mp:>10.6f}  {mc:>10.6f}  {rel:>9.2e}  {tol:>8.0e}  {rows:>4}  "
              f"{'ok' if ok else 'FAIL'}{why}")

    if len(py) != len(c):
        failed += 1
        print(f"compare_steps: FAIL: {len(py)} Python step(s) vs {len(c)} jefftrain step(s): the two runs did not take "
              "the same steps")
    if len(py) < a.late or len(c) < a.late:
        failed += 1
        print(f"compare_steps: FAIL: the gate needs step {a.late}; the trace stops at "
              f"{min(len(py), len(c))} step(s)")
    if len(py) != len(c):
        high = min(len(py), len(c))
    else:
        high = len(py)

    # ---- the dev soft CE at step `late` (design §10.3.2's second half) --------------------------------------------
    report = find_c_report(a.c, a.c_report)
    c_dev: list[dict] = []
    if report is not None and report.is_file():
        c_dev = json.loads(report.read_text()).get("history", [])
        print(f"compare_steps: jefftrain's report: {report}")
    else:
        print(f"compare_steps: note: no jefftrain report.json for {a.c}: the dev gate has no jefftrain number "
              "(pass --c-report)")
    pv, cv = dev_at(py_dev, a.late), dev_at(c_dev, a.late)
    print(f"compare_steps: dev soft CE @ step {a.late}: python {pv if pv is not None else 'missing'}  "
          f"jefftrain {cv if cv is not None else 'missing'}  gate |d| < {a.dev_abs:g}")
    if pv is None or cv is None:
        failed += 1
        print("compare_steps: FAIL: the dev soft CE at the gate's step is missing on one side")
    else:
        d = abs(pv - cv)
        if d >= a.dev_abs:
            failed += 1
        print(f"compare_steps: dev |d| {d:.2e}  {'ok' if d < a.dev_abs else 'FAIL'}")

    verdict = "PASS" if failed == 0 else "FAIL"
    print(f"compare_steps: {verdict}: {high} step(s) compared, {failed} failed, "
          f"worst |dmean|/mean_py {worst[0]:.2e} at step {worst[1]}"
          + (f", dev |d| {abs(pv - cv):.2e}" if pv is not None and cv is not None else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
