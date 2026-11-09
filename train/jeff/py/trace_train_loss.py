#!/usr/bin/env python3
"""trace_train_loss.py — the Python side of plan M6.2's step parity (design §10.3, item 2).

Plan M6.2 asks for "the M5.3 trick" again, this time on the loss instead of the batch order: import CLM-8B's
``tools.jeff_lora.train`` as ``T``, wrap ``T.soft_ce`` and, **when gradients are enabled**, append the call's per-row
values to the current optimizer step's list. The ``torch.enable_grad()`` filter is what keeps the dev/eval calls out of
the trace: ``evaluate()`` is decorated ``@torch.no_grad()``, so its ``soft_ce`` calls (the ``check()`` at step 0, at
step 20 and at the end) never touch the record. What is left is exactly one trace record per optimizer step, holding the
whole accumulation group's rows in batch order — the same shape ``jefftrain --trace-loss`` writes (plan M5.2,
``src/main.c``'s ``trace_step()``).

The step boundary is the optimizer itself: ``torch.optim.AdamW.step`` is wrapped too, and its return closes the current
record and starts the next one. With ``--accum 1`` (plan M6.2's run) that is one ``soft_ce`` call per step anyway, but
grouping on ``opt.step`` keeps ``--accum N`` honest instead of silently merging N batches into one record.

The dev soft CE (design §10.3.2's second half of the gate) comes from the run's own ``report.json``:
``main()``'s ``check()`` appends ``{"step", "tag", "dev_soft_ce", ...}`` for every dev check, so the step-20 entry is
read from there instead of being printed and scraped. ``compare_steps.py`` reads both sides from these two files.

Nothing here reimplements train.py (design §9.1): the trace only *observes* it. The run is the plan's own —

    T.main([--dtype fp32 --init-adapter A --train-rows 256 --dev-rows 64 --batch 8 --max-steps 20 --eval-every 20 ...])

with the real splits, so it runs on gpubox inside CLM-8B's venv, next to the real data (plan rule 5). The adapter ``A``
is ``py/make_adapter.py``'s synthetic one (design §10.1): 256 rows of real data train on a random LoRA, and nothing here
copies weights or rows into macuda.

usage (CLM-8B's gpubox venv):

    ~/CLM-8B/.venv-cuda/bin/python py/trace_train_loss.py --clm ~/CLM-8B --trace /tmp/j33/py.json \\
        --out /tmp/j33/py-out --init-adapter /tmp/j33/adapter --model ~/models/jeff-base \\
        --train ~/CLM-8B/data/ovril_detect/out/train.jsonl --dev ~/CLM-8B/data/ovril_detect/out/development.jsonl

The written JSON:

    {"python": {"argv": [...], "torch": "2.8.0+cu128", "device": "cuda", "dtype": "fp32"},
     "steps":  [{"step": 1, "n": 8, "mean": 1.8288, "ce": [ ... 8 per-row values ... ]}, ...],
     "dev":    [{"step": 0, "tag": "start", "dev_soft_ce": 1.12606}, {"step": 20, "tag": "step", ...}, ...]}

Exit code 0 when the trace has at least one step; 1 on a missing/empty trace (a wrong run should not look green).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="trace train.py's per-row soft CE, one record per optimizer step (M6.2)")
    ap.add_argument("--clm", required=True, type=Path, help="CLM-8B checkout to import tools.jeff_lora.train from")
    ap.add_argument("--model", type=Path, default=None, help="Jeff base checkpoint (default: train.py's default_model())")
    ap.add_argument("--train", required=True, type=Path, help="the train split (.jsonl)")
    ap.add_argument("--dev", required=True, type=Path, help="the dev/selection split (.jsonl)")
    ap.add_argument("--init-adapter", required=True, type=Path, help="the adapter train.py loads (make_adapter.py's)")
    ap.add_argument("--out", required=True, type=Path, help="train.py's --output (must be new or empty)")
    ap.add_argument("--trace", required=True, type=Path, help="where to write the JSON trace")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-rows", type=int, default=256)
    ap.add_argument("--dev-rows", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-temperature", default="1")
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA lr (train.py's default and env.sh's COMMON_ARGS)")
    ap.add_argument("--readout-lr", type=float, default=5e-6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--targets", choices=("soft", "hard"), default="soft")
    ap.add_argument("--weights", choices=("none", "balanced"), default="none")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def import_train(clm: Path):
    """Import CLM-8B's tools.jeff_lora.train by path (design §9.1: its code, never a copy)."""
    root = clm.expanduser().resolve()
    if not (root / "tools" / "jeff_lora" / "train.py").is_file():
        raise SystemExit(f"trace_train_loss: {root} has no tools/jeff_lora/train.py; --clm wants CLM-8B's checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import tools.jeff_lora.train as T  # noqa: E402  (imported by path, design §9.1)

    return T


def main(argv: list[str]) -> int:
    a = parse_args(argv)
    import torch

    T = import_train(a.clm)

    # ---- the trace: T.soft_ce under grad, closed by the optimizer step (this file's whole job) ---------------------
    records: list[dict] = []
    cur: list[float] = []
    calls = {"soft_ce": 0, "train_soft_ce": 0, "steps": 0}

    soft_ce, adamw_step = T.soft_ce, torch.optim.AdamW.step

    def traced_soft_ce(logits, target):
        out = soft_ce(logits, target)
        calls["soft_ce"] += 1
        if torch.is_grad_enabled():  # train.py's loop; evaluate() is @torch.no_grad()
            calls["train_soft_ce"] += 1
            cur.extend(float(v) for v in out.detach().float().cpu().tolist())
        return out

    def traced_step(self, closure=None):
        out = adamw_step(self, closure)
        if cur:  # one optimizer step = one trace record; accum > 1 keeps the group's rows in batch order
            ce = list(cur)
            records.append({"step": len(records) + 1, "n": len(ce), "mean": sum(ce) / len(ce), "ce": ce})
            cur.clear()
            calls["steps"] += 1
        return out

    T.soft_ce = traced_soft_ce
    torch.optim.AdamW.step = traced_step

    out = a.out.expanduser().resolve()
    train_argv = ["--train", str(a.train.expanduser()), "--dev", str(a.dev.expanduser()),
                  "--output", str(out), "--init-adapter", str(a.init_adapter.expanduser()),
                  "--dtype", "fp32", "--device", a.device, "--seed", str(a.seed),
                  "--batch", str(a.batch), "--accum", str(a.accum), "--epochs", str(a.epochs),
                  "--train-rows", str(a.train_rows), "--dev-rows", str(a.dev_rows),
                  "--max-steps", str(a.max_steps), "--eval-every", str(a.eval_every),
                  "--eval-temperature", str(a.eval_temperature),
                  "--lr", str(a.lr), "--readout-lr", str(a.readout_lr),
                  "--rank", str(a.rank), "--alpha", str(a.alpha),
                  "--targets", a.targets, "--weights", a.weights]
    if a.model is not None:
        train_argv = ["--model", str(a.model.expanduser())] + train_argv
    print("trace_train_loss: T.main(%s)" % " ".join(train_argv), flush=True)

    try:
        rc = T.main(train_argv)
    finally:
        T.soft_ce, torch.optim.AdamW.step = soft_ce, adamw_step
    if rc not in (0, None):
        raise SystemExit(f"trace_train_loss: train.py's main() returned {rc}")

    # ---- the dev checks: train.py's report.json history (design §10.3.2's dev soft CE) -----------------------------
    report_path = out / "report.json"
    history = json.loads(report_path.read_text())["history"] if report_path.is_file() else []
    dev = [{"step": h["step"], "tag": h.get("tag"), "dev_soft_ce": h["dev_soft_ce"]}
           for h in history if "dev_soft_ce" in h]

    if not records:
        raise SystemExit("trace_train_loss: no optimizer step ran (no grad-enabled soft_ce under opt.step): "
                         f"soft_ce calls {calls['soft_ce']}, of them under grad {calls['train_soft_ce']}")

    payload = {
        "python": {
            "argv": train_argv,
            "torch": torch.__version__,
            "device": a.device,
            "dtype": "fp32",
            "calls": calls,
        },
        "steps": records,
        "dev": dev,
    }
    path = a.trace.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")

    means = " ".join(f"{r['mean']:.6f}" for r in records)
    print(f"trace_train_loss: {len(records)} step(s), {sum(r['n'] for r in records)} row losses; "
          f"mean CE {means}", flush=True)
    last = dev[-1] if dev else None
    print(f"trace_train_loss: dev checks {[(d['step'], d['dev_soft_ce']) for d in dev]}"
          + (f"  (last: step {last['step']} soft CE {last['dev_soft_ce']})" if last else ""), flush=True)
    print(f"trace_train_loss: wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main(sys.argv[1:]))
