#!/usr/bin/env python3
"""make_adapter.py — write a synthetic LoRA adapter for the parity tests, in CLM-8B's own adapter format.

Plan M4.1 / design §10.2: the whole-model parity tests need an adapter whose LoRA path is *not* zero (a freshly attached
adapter has B = 0, which would make every `(x A^T) B^T` term vanish and hide a broken LoRA GEMM). So:

    A ~ U(-1/sqrt(in), 1/sqrt(in))   and   B ~ N(0, 1) * 1e-2, per layer, from one torch.Generator seeded with --seed

the shapes, dtypes and file names being CLM-8B's (`tools/jeff_lora/lora.py`): `attach_lora(model, rank, alpha)` wraps the
150 projection layers this trainer adapts, and `save_lora` writes `adapter_model.safetensors` / `readout.safetensors` /
`adapter_config.json`. The readout is the model's own (the checkpoint's `readout.safetensors`), not a random one.

This is a *synthetic* adapter for tests only (plan rule 5): nothing here reads CLM-8B's data/ and nothing here is a real
trained adapter. It runs where CLM-8B's venv is (the Mac) and imports CLM-8B's code by path, never a copy of it.

usage (CLM-8B's venv, from anywhere):
    python py/make_adapter.py --clm ~/CLM-8B --model ~/models/jeff-base --seed 1 --out /tmp/jeff-ref/adapter
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="write a synthetic LoRA adapter in CLM-8B's format (plan M4.1)")
    ap.add_argument("--model", required=True, type=Path, help="Jeff base checkpoint directory (config.json, model.safetensors, readout.safetensors)")
    ap.add_argument("--out", required=True, type=Path, help="adapter directory to write (created if missing)")
    ap.add_argument("--seed", type=int, default=1, help="torch.Generator seed for A and B (default 1)")
    ap.add_argument("--clm", type=Path, default=None,
                    help="CLM-8B checkout to import tools.jeff_lora.lora from (default: $CLM_DIR)")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (default 16, train.py's)")
    ap.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha (default 32, train.py's)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                    help="where the A/B tensors are drawn and stored (default cpu; save_lora writes CPU fp32 either way)")
    return ap.parse_args(argv)


def find_clm(given: Path | None) -> Path:
    """--clm, else $CLM_DIR: the checkout whose tools/jeff_lora/lora.py is the authority on the adapter format."""
    import os

    clm = given or (Path(os.environ["CLM_DIR"]).expanduser() if os.environ.get("CLM_DIR") else None)
    if clm is None:
        raise SystemExit("make_adapter: pass --clm /path/to/CLM-8B (or set CLM_DIR)")
    clm = clm.expanduser().resolve()
    if not (clm / "tools" / "jeff_lora" / "lora.py").is_file():
        raise SystemExit(f"make_adapter: {clm} has no tools/jeff_lora/lora.py; --clm wants CLM-8B's checkout")
    return clm


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    clm = find_clm(args.clm)
    sys.path.insert(0, str(clm))
    from tools.jeff_lora.lora import attach_lora, save_lora  # noqa: E402  (CLM-8B wins on the format)

    from transformers import AutoModel

    model_dir = args.model.expanduser().resolve()
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32, attn_implementation="eager")
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    layers = attach_lora(model, args.rank, args.alpha)
    weight = load_file(str(model_dir / "readout.safetensors"))["weight"]
    readout = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        readout.weight.copy_(weight.float())
    readout.to(device)

    gen = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for layer in layers:
            bound = 1.0 / math.sqrt(layer.A.shape[1])
            layer.A.uniform_(-bound, bound, generator=gen)
            layer.B.normal_(0.0, 1.0, generator=gen).mul_(1e-2)

    args.out.expanduser().mkdir(parents=True, exist_ok=True)
    save_lora(args.out.expanduser(), layers, readout,
              {"step": 0, "base": model_dir.name, "targets": "soft", "weights": "none",
               "seed": args.seed, "synthetic": True})

    a_abs = max(float(layer.A.detach().abs().max()) for layer in layers)
    b_abs = max(float(layer.B.detach().abs().max()) for layer in layers)
    b_rms = math.sqrt(sum(float((layer.B.detach().float() ** 2).mean()) for layer in layers) / len(layers))
    print(f"make_adapter: {len(layers)} LoRA layers, rank {args.rank}, scale {layers[0].scale:g}, "
          f"A ~ U(+-1/sqrt(in)) max|A| {a_abs:.6g}, B ~ N(0,1)*1e-2 rms {b_rms:.6g} max|B| {b_abs:.6g}, seed {args.seed}",
          flush=True)
    print(f"make_adapter: readout from {model_dir / 'readout.safetensors'} [{weight.shape[0]},{weight.shape[1]}], "
          f"fp32, copied exactly (max|diff| {float((readout.weight.detach().cpu() - weight.float()).abs().max()):g})", flush=True)
    for name in ("adapter_model.safetensors", "readout.safetensors", "adapter_config.json"):
        path = args.out.expanduser() / name
        print(f"make_adapter: {path}  {path.stat().st_size} bytes", flush=True)
    print(f"make_adapter: {len(layers)} layers x 2 factors, first stem "
          f"{layers[0].key.removesuffix('.weight')}, last stem {layers[-1].key.removesuffix('.weight')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
