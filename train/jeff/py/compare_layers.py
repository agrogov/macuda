#!/usr/bin/env python3
"""compare_layers.py — key-by-key parity of two dump_ref-style safetensors files (design §10.2's gate).

    python py/compare_layers.py ref.safetensors out.safetensors [--atol-rel 1e-3] [--grad-rel 5e-3]

For every key of both files, in the reference's file order, print `key  rel  ok/FAIL`, where

    rel = max|a - b| / max|b|            (design §10.2: b is the reference, so the denominator is the reference's scale)

gated by `--atol-rel` for activations and `--grad-rel` for the keys that start with `grad.` (gradients are the sum of many
more roundings, so they get the looser tolerance the design sets). Integer keys (`in.ids`, `in.off`) must match exactly.
A key missing from either file, a shape mismatch and a NaN all count as FAIL. Exit 1 if any key failed or the files share
no keys at all — the first failing key is the one to chase, so it is also printed on its own line at the end.

No tolerance is ever loosened here: the numbers come from the command line, whose defaults are the plan's (M4.3: 1e-3
with jefftrain built -DJT_EXACT=1, M5.1: 5e-3 for gradients).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="compare two dump_ref-style safetensors files key by key (design §10.2)")
    ap.add_argument("ref", type=Path, help="the reference (dump_ref.py's ref.safetensors; the gate's denominator)")
    ap.add_argument("other", type=Path, help="the file under test (jefftrain --dump-forward / --dump-backward)")
    ap.add_argument("--atol-rel", type=float, default=1e-3, help="relative tolerance for activations (default 1e-3)")
    ap.add_argument("--grad-rel", type=float, default=5e-3, help="relative tolerance for grad.* keys (default 5e-3)")
    return ap.parse_args(argv)


def relative(ref: torch.Tensor, other: torch.Tensor) -> float:
    """max|a-b| / max|b|, the design's per-tensor measure; inf when the reference is all zeros and the values differ."""
    if ref.numel() == 0:
        return 0.0
    diff = float((other.double() - ref.double()).abs().max())
    scale = float(ref.double().abs().max())
    if scale == 0.0:
        return 0.0 if diff == 0.0 else math.inf
    return diff / scale


def describe(shape) -> str:
    return "x".join(str(n) for n in shape) or "scalar"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ref = load_file(str(args.ref))
    other = load_file(str(args.other))
    grad_rel = float(args.grad_rel)
    order = list(ref) + [k for k in other if k not in ref]
    shared = sum(1 for k in ref if k in other)
    first_fail = None
    n_ok = n_fail = 0
    print(f"compare_layers: {args.ref} (reference) vs {args.other}: {len(ref)} vs {len(other)} keys, "
          f"atol-rel {args.atol_rel:g}, grad-rel {grad_rel:g}", flush=True)
    for key in order:
        if key not in other:
            n_fail += 1
            first_fail = first_fail or key
            print(f"{key}  -  FAIL (not in {args.other.name})", flush=True)
            continue
        if key not in ref:
            n_fail += 1
            first_fail = first_fail or key
            print(f"{key}  -  FAIL (not in {args.ref.name})", flush=True)
            continue
        want, got = ref[key], other[key]
        if tuple(want.shape) != tuple(got.shape):
            n_fail += 1
            first_fail = first_fail or key
            print(f"{key}  -  FAIL (shape {describe(want.shape)} in the reference, {describe(got.shape)} here)", flush=True)
            continue
        if want.dtype in INT_DTYPES or got.dtype in INT_DTYPES:
            ok = torch.equal(want.to(torch.int64), got.to(torch.int64))
            n_ok, n_fail = (n_ok + 1, n_fail) if ok else (n_ok, n_fail + 1)
            if not ok:
                first_fail = first_fail or key
            print(f"{key}  exact  {'ok' if ok else 'FAIL'}", flush=True)
            continue
        rel = relative(want, got)
        tol = grad_rel if key.startswith("grad.") else float(args.atol_rel)
        ok = math.isfinite(rel) and rel < tol
        n_ok, n_fail = (n_ok + 1, n_fail) if ok else (n_ok, n_fail + 1)
        if not ok:
            first_fail = first_fail or key
        print(f"{key}  {rel:.3e}  {'ok' if ok else 'FAIL'}" + ("" if ok else f" (want < {tol:g})"), flush=True)
    print(f"compare_layers: {n_ok} ok, {n_fail} failed, {len(order)} key(s) compared", flush=True)
    if first_fail is not None:
        print(f"first failing key: {first_fail}", flush=True)
    if shared == 0:
        print(f"compare_layers: {args.ref} and {args.other} share no keys", flush=True)
        return 1
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
