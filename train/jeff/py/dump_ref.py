#!/usr/bin/env python3
"""dump_ref.py — the layer-by-layer reference for jefftrain: one forward + backward, every activation and gradient.

Design §10.2, plan M4.1. Loads jeff-base in fp32 with eager attention, attaches CLM-8B's LoRA, loads a synthetic adapter
(`py/make_adapter.py`), runs one forward+backward on one packed batch from a `.jtb` fixture, and writes `ref.safetensors`
with, for every decoder layer i:

    layer.{i}.in         [N,1024]   the layer's input (packed: the rows' real tokens, no padding)
    layer.{i}.post_mixer [N,1024]   the residual right after the mixer, i.e. the input of post_attention_layernorm
                                    -> test a mixer as layer.{i}.in -> post_mixer and the MLP as post_mixer -> layer.{i}.out
    layer.{i}.out        [N,1024]   the layer's output
    final                [N,1024]   the final norm's output (GatedRMSNorm-free: plain RMSNorm, 1+w)
    logits               [B,n]      the readout logits over the rows' live options (dead ones -1e9), train.py's width
    loss                 []         train.py's weighted soft-CE of the batch
    grad.layer.{i}.out   [N,1024]   d(loss)/d(layer.{i}.out), packed
    grad.final           [N,1024]   d(loss)/d(final)
    grad.{stem}.lora_A/B            d(loss)/d(A), d(loss)/d(B) for all 150 adapted projections (fp32, CLM-8B's stems)
    grad.readout         [255,1024] d(loss)/d(readout.weight)
    in.ids               [N]        i32 packed token ids (row b is in.ids[off[b]:off[b+1]])
    in.off               [B+1]      i32 prefix sums of the rows' real lengths

The GDN recurrence is replaced by `literal_gdn`, a plain torch loop of design §3.6 with
`torch_chunk_gated_delta_rule`'s signature and return value, and its call count is asserted to be 18 per forward: on
gpubox, where fla is installed, transformers' hub-kernel decorators could otherwise route even a CPU run to a chunked
kernel whose summation order is not the one jefftrain implements. Everything else is the model's own code, so the only
difference to jefftrain is the order of sums inside GEMMs and reductions (the 1e-3/5e-3 gates of design §10.2).

Reference numbers must not depend on where they were made, so run this on CPU (`--device cpu`, the default: fp32, eager,
literal GDN) and keep the file outside macuda — jefftrain's copy on the card compares against the same file (plan M4.4).
The fixture is synthetic (plan rule 5) and the model is public.

usage (CLM-8B's venv):
    python py/make_fixture.py --rows 2 --lengths 37,70 --out /tmp/jeff-ref/fx.jtb
    python py/make_adapter.py --clm ~/CLM-8B --model ~/models/jeff-base --seed 1 --out /tmp/jeff-ref/adapter
    python py/dump_ref.py --clm ~/CLM-8B --model ~/models/jeff-base --adapter /tmp/jeff-ref/adapter \\
        --fixture /tmp/jeff-ref/fx.jtb --out /tmp/jeff-ref/ref.safetensors --device cpu
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

# ---------------------------------------------------------------------------------------------------------------
# the .jtb fixture (design §9.1, written by py/make_fixture.py)
# ---------------------------------------------------------------------------------------------------------------
JTB_MAGIC = b"JTB1"
JTB_VERSION = 1
JTB_SECTIONS = 5


def read_jtb(path: Path) -> tuple[list[dict], list[list[int]]]:
    """The rows and the batch list of a JTB1 file: what collate() needs (len/n_opts/weight/ids/target)."""
    blob = Path(path).read_bytes()
    if blob[:4] != JTB_MAGIC:
        raise SystemExit(f"dump_ref: {path} does not start with {JTB_MAGIC!r}")
    version, n_rows, n_batches, _max_tokens, _max_len, _max_opts = struct.unpack_from("<6I", blob, 4)
    if version != JTB_VERSION:
        raise SystemExit(f"dump_ref: {path} is version {version}, want {JTB_VERSION}")
    at = 28
    off = struct.unpack_from(f"<{JTB_SECTIONS}Q", blob, at)
    rows_off, ids_off, targets_off, batches_off, _strings_off = off
    rows, rows_at = [], rows_off
    for _ in range(n_rows):
        length, n_opts, weight, _top, _id_off = struct.unpack_from("<IIfIQ", blob, rows_at)
        rows_at += struct.calcsize("<IIfIQ")
        rows.append({"len": length, "n_opts": n_opts, "weight": float(weight)})
    ids, ids_at = [], ids_off
    targets, targets_at = [], targets_off
    for row in rows:
        row["ids"] = list(struct.unpack_from(f"<{row['len']}I", blob, ids_at))
        ids_at += 4 * row["len"]
        row["target"] = list(struct.unpack_from(f"<{row['n_opts']}f", blob, targets_at))
        targets_at += 4 * row["n_opts"]
    batches, batches_at = [], batches_off
    for _ in range(n_batches):
        (n,) = struct.unpack_from("<I", blob, batches_at)
        batches_at += 4
        batches.append(list(struct.unpack_from(f"<{n}I", blob, batches_at)))
        batches_at += 4 * n
    return rows, batches


# ---------------------------------------------------------------------------------------------------------------
# the GDN recurrence, literally (design §3.6), with torch_chunk_gated_delta_rule's signature and return value
# ---------------------------------------------------------------------------------------------------------------
GDN_CALLS = 0


def literal_gdn(query, key, value, g, beta, chunk_size: int = 64, initial_state=None, output_final_state: bool = False,
                use_qk_l2norm_in_kernel: bool = False, **kwargs):
    """Design §3.6's loop, per head: S = S*exp(g); delta = (v - S^T k)*beta; S += k delta^T; o = S^T q.

    Same arguments, defaults and return value as transformers' `torch_chunk_gated_delta_rule` (which it stands in for):
    query/key [B,L,Hk,Dk], value [B,L,Hv,Dv], g and beta [B,L,Hv], returns (out [B,L,Hv,Dv], final state or None).
    `chunk_size` is accepted and ignored (the loop is not chunked); extra hub-kernel kwargs are ignored too.
    """
    global GDN_CALLS
    GDN_CALLS += 1
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm  # the model's own l2 norm (sum, not mean)

    dtype = query.dtype
    batch, length, _, k_dim = key.shape
    v_heads, v_dim = value.shape[-2:]
    # fp32 and [B,H,L,D], exactly as the original does; the model's own code then rounds back to `dtype`
    q, k, v, be, de = (x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                       for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    q = q / (k_dim ** 0.5)
    if initial_state is None:
        state = torch.zeros((batch, v_heads, k_dim, v_dim), dtype=torch.float32, device=value.device)
    else:
        state = initial_state.to(torch.float32)
    out = torch.zeros_like(v)
    for t in range(length):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
        state = state * torch.exp(de[:, :, t])[..., None, None]
        kv = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv) * be[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    out = out.transpose(1, 2).contiguous().to(dtype)
    return out, (state if output_final_state else None)


def patch_gdn(model) -> int:
    """Point the module-level rule and conv, and every GDN instance, at the literal versions; return instances patched."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    modeling.torch_chunk_gated_delta_rule = literal_gdn
    modeling.causal_conv1d_fn = literal_conv  # gpubox's causal_conv1d is a CUDA-only custom op; see literal_conv
    patched = 0
    for module in model.modules():
        if isinstance(module, modeling.Qwen3_5GatedDeltaNet):
            module.torch_chunk_gated_delta_rule = literal_gdn
            module.causal_conv1d_fn = literal_conv
            patched += 1
    return patched


# ---------------------------------------------------------------------------------------------------------------
# the GDN conv, literally (design §3.6), with causal_conv1d_fn's signature
# ---------------------------------------------------------------------------------------------------------------
CONV_CALLS = 0


def literal_conv(hidden_states, weight, bias=None, activation: str | None = None, **kwargs):
    """Design §3.6's causal depthwise conv, one tap at a time: c[t,ch] = sum_k w[ch,0,k] * x[t-(K-1)+k,ch].

    The same arguments and return value as transformers' `causal_conv1d_fn`: hidden_states [B,C,L], weight [C,K] (the
    model passes `conv1d.weight.squeeze(1)`; a [C,1,K] weight would work too), bias [C] or None, activation or None. On gpubox `causal_conv1d` is installed, and the hub-kernel decorator on that
    function routes it to a CUDA-only custom op which fails on a CPU tensor; the loop below is the algorithm the
    Python fallback and jefftrain both implement (`F.conv1d(padding=K-1)` truncated to L is this, tap order included).
    """
    global CONV_CALLS
    CONV_CALLS += 1
    from torch.nn import functional as F

    taps = weight.shape[-1]
    w = weight.reshape(weight.shape[0], taps)  # the model passes conv1d.weight.squeeze(1): [C,K]
    out = torch.zeros_like(hidden_states)
    for tap in range(taps):
        shift = (taps - 1) - tap  # tap 0 multiplies x[t-(K-1)], tap K-1 multiplies x[t]
        column = w[:, tap].view(1, -1, 1)
        if shift:
            out[..., shift:] += hidden_states[..., :hidden_states.shape[-1] - shift] * column
        else:
            out += hidden_states * column
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    if activation is not None:
        out = F.silu(out) if activation == "silu" else getattr(F, activation)(out)
    return out.to(hidden_states.dtype)


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def pack(tensor: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """[B,W,...] -> [N,...]: keep only each row's real tokens, concatenated (the packed layout jefftrain uses)."""
    parts = [tensor[b, :n] for b, n in enumerate(lengths)]
    return torch.cat(parts, dim=0) if parts else tensor[:0]


def cpu32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu", torch.float32).contiguous()


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="dump jefftrain's layer-by-layer reference (design §10.2, plan M4.1)")
    ap.add_argument("--model", required=True, type=Path, help="Jeff base checkpoint directory")
    ap.add_argument("--adapter", required=True, type=Path, help="adapter directory from py/make_adapter.py")
    ap.add_argument("--fixture", required=True, type=Path, help=".jtb file from py/make_fixture.py (synthetic rows only)")
    ap.add_argument("--out", required=True, type=Path, help="where to write ref.safetensors")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                    help="cpu (default; fp32 + eager + literal GDN: the reference) or cuda (a smoke check only)")
    ap.add_argument("--clm", type=Path, default=None,
                    help="CLM-8B checkout to import tools.jeff_lora.{lora,train} from (default: $CLM_DIR)")
    return ap.parse_args(argv)


def find_clm(given: Path | None) -> Path:
    import os

    clm = given or (Path(os.environ["CLM_DIR"]).expanduser() if os.environ.get("CLM_DIR") else None)
    if clm is None:
        raise SystemExit("dump_ref: pass --clm /path/to/CLM-8B (or set CLM_DIR)")
    clm = clm.expanduser().resolve()
    if not (clm / "tools" / "jeff_lora" / "train.py").is_file():
        raise SystemExit(f"dump_ref: {clm} has no tools/jeff_lora/train.py; --clm wants CLM-8B's checkout")
    return clm


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    clm = find_clm(args.clm)
    sys.path.insert(0, str(clm))
    from tools.jeff_lora.lora import attach_lora  # noqa: E402
    from tools.jeff_lora.train import collate, load_adapter, option_logits, soft_ce  # noqa: E402  (train.py wins)

    from transformers import AutoModel, AutoTokenizer

    model_dir = args.model.expanduser().resolve()
    device = args.device
    rows_all, batches = read_jtb(args.fixture.expanduser())
    if not batches:
        raise SystemExit(f"dump_ref: {args.fixture} has no batches")
    rows = [rows_all[i] for i in batches[0]]
    lengths = [len(r["ids"]) for r in rows]
    print(f"dump_ref: fixture {args.fixture}: {len(rows_all)} rows, {len(batches)} batches; using batch 0 = rows "
          f"{batches[0]}, lengths {lengths}, n_opts {[r['n_opts'] for r in rows]}, weights {[r['weight'] for r in rows]}",
          flush=True)

    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32, attn_implementation="eager").to(device)
    model.requires_grad_(False)
    model.eval()
    layers = attach_lora(model, 16, 32)
    weight = load_file(str(model_dir / "readout.safetensors"))["weight"]
    readout = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        readout.weight.copy_(weight.float())
    readout.to(device)
    load_adapter(args.adapter.expanduser().resolve(), layers, readout)
    print(f"dump_ref: model {model_dir} fp32, eager, {len(layers)} LoRA layers (rank {layers[0].rank}, scale "
          f"{layers[0].scale:g}), adapter {args.adapter.expanduser()}", flush=True)

    text = model.language_model  # Qwen3_5TextModel: .layers (24 x Qwen3_5DecoderLayer) and .norm (the final RMSNorm)
    n_layers = len(text.layers)
    gdn_patched = patch_gdn(model)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids, mask, index, target, live, wt = collate(rows, pad_id, device)
    gw = sum(r["weight"] for r in rows)
    offline = [int(o) for o in mask.sum(1).tolist()]
    if offline != lengths:
        raise SystemExit(f"dump_ref: collate's lengths {offline} != fixture lengths {lengths}")
    print(f"dump_ref: batch {tuple(ids.shape)} padded (pad id {pad_id}), packed N={sum(lengths)}, live {live.shape[1]}, "
          f"gdn layers patched {gdn_patched}", flush=True)

    saved: dict[str, torch.Tensor] = {}      # packed fp32 tensors, keyed the way ref.safetensors wants them
    padded: dict[str, torch.Tensor] = {}     # the padded tensors whose .grad we need after backward
    seen: dict[int, int] = {}

    def layer_hook(i: int):
        def hook(module, args_in, output):
            seen[i] = seen.get(i, 0) + 1
            saved[f"layer.{i}.in"] = cpu32(pack(args_in[0], lengths))
            saved[f"layer.{i}.out"] = cpu32(pack(output, lengths))
            output.retain_grad()
            padded[f"layer.{i}.out"] = output
        return hook

    def post_mixer_hook(i: int):
        # post_attention_layernorm is the first thing the mixer's residual feeds (design §3.4), so its input is it
        def hook(module, args_in):
            saved[f"layer.{i}.post_mixer"] = cpu32(pack(args_in[0], lengths))
        return hook

    def final_hook(module, args_in, output):
        saved["final"] = cpu32(pack(output, lengths))
        output.retain_grad()
        padded["final"] = output

    handles = []
    for i, layer in enumerate(text.layers):
        handles.append(layer.register_forward_hook(layer_hook(i)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(post_mixer_hook(i)))
    handles.append(text.norm.register_forward_hook(final_hook))

    try:
        with torch.enable_grad():
            logits = option_logits(model, readout, ids, mask, index, live, device)
            per = soft_ce(logits, target)
            loss = (per * wt).sum() / gw
            if GDN_CALLS != 18 or CONV_CALLS != 18:  # one per GDN layer, i.e. the patched code really ran everywhere
                raise SystemExit(f"dump_ref: per forward literal_gdn ran {GDN_CALLS} and literal_conv ran {CONV_CALLS} "
                                 f"times, want 18 each (one per GDN layer)")
            print(f"dump_ref: forward ok, literal_gdn calls {GDN_CALLS}, literal_conv calls {CONV_CALLS}, "
                  f"rows' CE {[round(float(x.detach()), 6) for x in per]}", flush=True)
            loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    if sorted(seen) != list(range(n_layers)):
        raise SystemExit(f"dump_ref: layer hooks fired for {sorted(seen)}, want one each for 0..{n_layers - 1}")
    if any(count != 1 for count in seen.values()):
        raise SystemExit(f"dump_ref: a layer hook fired more than once: {seen}")

    for i in range(n_layers):
        grad = padded[f"layer.{i}.out"].grad
        if grad is None:
            raise SystemExit(f"dump_ref: no gradient at layer.{i}.out (retain_grad on a tensor outside the graph?)")
        saved[f"grad.layer.{i}.out"] = cpu32(pack(grad, lengths))
    saved["grad.final"] = cpu32(pack(padded["final"].grad, lengths))
    off = [0] + [sum(lengths[:b + 1]) for b in range(len(lengths))]
    saved["in.ids"] = torch.tensor([t for r in rows for t in r["ids"]], dtype=torch.int32)
    saved["in.off"] = torch.tensor(off, dtype=torch.int32)
    saved["logits"] = cpu32(logits)
    saved["loss"] = cpu32(loss.reshape(()))

    missing = 0
    for layer in layers:
        stem = layer.key.removesuffix(".weight")
        if layer.A.grad is None or layer.B.grad is None:
            missing += 1
            continue
        saved[f"grad.{stem}.lora_A"] = cpu32(layer.A.grad)
        saved[f"grad.{stem}.lora_B"] = cpu32(layer.B.grad)
    if missing:
        raise SystemExit(f"dump_ref: {missing} of {len(layers)} LoRA layers got no gradient")
    if readout.weight.grad is None:
        raise SystemExit("dump_ref: the readout got no gradient")
    saved["grad.readout"] = cpu32(readout.weight.grad)

    # the padding must carry no gradient: jefftrain is packed, so a non-zero one means the batch is not right-padded
    stray = 0.0
    for key, tensor in padded.items():
        for b, n in enumerate(lengths):
            if tensor.shape[1] > n:
                stray = max(stray, float(tensor.grad[b, n:].abs().max()))
    out_path = args.out.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(saved, str(out_path))

    print(f"dump_ref: loss {float(loss.detach()):.8f}, logits {tuple(saved['logits'].shape)} "
          f"(width = the widest row's live options, train.py's option_logits), padded-grad max|.| {stray:g}", flush=True)
    print(f"dump_ref: wrote {out_path}: {len(saved)} keys, N={saved['in.ids'].shape[0]}, "
          f"{out_path.stat().st_size / 2**20:.1f} MiB "
          f"({n_layers} layers x in/post_mixer/out, final, logits, loss, grads, {len(layers)} LoRA pairs)", flush=True)
    print(f"dump_ref: example keys {sorted(saved)[:4]} ... {sorted(saved)[-3:]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
