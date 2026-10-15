# Serving Qwen3.6-35B-A3B (APEX I-Compact) on one card

Notes from the 2026-10-06/07 sessions: the command that runs this model at 260K context on one card, what to expect
from it, and the two problems met on the way (a driver fault, and generation loops). Numbers are from `logs/serve-*.log`
and are one user's coding-agent sessions, not a benchmark.

## The command

```sh
BIN=$PWD/cuda-shim/build/bin/llama-server-null \
TINYNV_UNLOAD=1 \
CTX=260000 MTP= \
SERVE_ARGS="--no-host --no-repack -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 2048 \
  --spec-type draft-mtp --spec-draft-n-max 4" \
sh tools/serve.sh start models/Qwen3.6-35B-A3B-APEX-I-Compact.gguf
```

- `MTP=` turns off the separate MTP head file; this GGUF has the head embedded, so `--spec-type draft-mtp` is enough.
- `TINYNV_UNLOAD=1` makes the card unload cleanly at exit, so the next start needs no enclosure power-cycle (it is the
  warm-restart path; see `logs/gb203-warm-restart.patch` and Davinchy/macuda#1).
- `serve.sh` already passes `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0`. Everything after that comes from `SERVE_ARGS`.
- Memory: 17.0 GB of weights, about 2.9 GiB of q8_0 KV at 260K, plus compute buffers at `-ub 2048`. Fits a 24 GB card.

## Generation loops, and why DRY is not in the command

**What happened (confirmed from the agent's own session log, 2026-10-07).** The client sends only `maxTokens: 32000`
and `reasoningEffort: xhigh`; no sampler settings, so the server's `--temp 0.6 --top-p 0.95 --top-k 20` apply. Two kinds of
loop were found:

- *Write-call loops.* Three `write` tool calls for `stress/config-boundary.test.ts` (steps 4, 9 and 10 of one turn) each
  ran to the 32,000-token cap and ended in `["(?:a|b|c|...){10000}"], ` repeated thousands of times (about 52K characters,
  compression ratio 0.03). Each took about 190-200 s of GPU time. In the server log they are the two requests that stopped
  at exactly 32,000 tokens with 99.8% MTP acceptance (`draft acceptance = 0.998, mean len = 4.99`).
- *Thinking loops.* Turns 8, 9 and 10 (a "create the commits" request) looped in reasoning ("Let me create the commits
  now: 1. **Security: ...**" repeated, 135-595 sentences of which 98-229 distinct); the user aborted all three.

The near-100% acceptance is the tell, but it is not proof by itself: a session writing many similar files also copies its own
text. Here the text was checked and these were loops.

**It is not the engine.** The same 116K-token prompt at temperature 0 gave byte-identical output (and identical MTP draft
counts) with delta delivery on and off, and a coherent summary. The loops are model behaviour (Compact is the smallest APEX
quant, q8_0 KV, long context, high reasoning effort).

**Why DRY (`--dry-multiplier`) does not fix it cleanly.** DRY cuts the score of a token that would extend a sequence that
already occurred, by `multiplier * base ^ (repeat length - allowed length)`. I replayed it offline over the real session
text (rough tokenizer, so lengths are approximate; the script is not kept in the repo): the loop unit above is about 22
tokens but the default sequence breakers (newline, `:`, `"`, `*`) cap the measurable repeat length at about 13, so the
penalty can only grow to what a 13-token match earns. Against the 11 files the model wrote legitimately in the same session:

| `--dry-allowed-length` | legit tokens penalized by more than 10 logits | loop stopped |
|---|---|---|
| 2 (default) | 19% | after about 1.5 repeats |
| 4 | 13% | after about 1.6 |
| 8 | 6% | after about 1.8 |
| 12 | 3.5% | not at all (match never reaches it) |

So any setting that stops this loop also pushes several percent of ordinary tokens (boilerplate lines in test files that
repeat by design) tens of logits down, and the setting that spares code does not stop the loop. DRY was therefore **not**
added. If you want to try it anyway (`--dry-multiplier 0.8 --dry-allowed-length 8 --dry-penalty-last-n 8192`), expect
distorted boilerplate and watch the tool-call failure rate. Presence/repetition penalties are no better: the model card's
coding setting is presence_penalty 0.0 and repetition_penalty 1.0.

**What bounds the damage instead.** Neither of these needs a server change.
- A smaller `maxTokens` for `write` calls: a 52K-character file is already near the 32,000-token cap, so a truncated write is
  a bad write anyway, and a looped one costs 190-200 s of GPU time.
- A client-side abort on repetitive output. Measured on this session with zlib: the four looped `write` bodies compress to
  0.026-0.032 of their size, the 11 legitimate ones to 0.124-0.298, so a ratio under about 0.06 over a growing write body
  separates them cleanly. Thinking loops do not separate that way (0.10-0.22 over the whole reasoning text, overlapping
  ordinary reasoning at 0.23-0.44); for those, detect a suffix repeating verbatim several times in a row instead.

## The fault this run exposed

Before DRY, the server died every ~7 minutes on an RTX 5090 with an mmu fault: compute channel, read, "no page table
entry". Delta delivery was off, so it was not that. The VA ledger (libtinynv, `mmu.c`) showed the fault address was
exactly the end of a live 1,042,944-byte `cudaMalloc` (a ggml pool buffer), with the next buffer starting one page
later: a kernel read just past the end of its buffer and hit unmapped memory. Real CUDA pads allocations to 2 MB pages,
so the same read is harmless there. The fix maps 64 KB of slack after every allocation (`TINYNV_ALLOC_SLACK_KB`, default
64, 0 restores the old layout). After the fix the server ran more than 30 minutes at up to 189,600 tokens of context
without a fault. That is encouraging, not proof; the code is on branch `fix-oob-read-alloc-slack`.

If a fault does happen, do not restart first. The log then holds a `VA LEDGER` block (what the address is) and a
`POST-MORTEM` block (descriptor window from the failing chain). Power-cycle the enclosure afterwards.

## Reading the log

Log timestamps are `minutes.seconds.milliseconds` from server start, not hours. Check uptime with
`ps -o etime= -p <pid>`. Per request, the `slot print_timing` lines give prompt-eval time (the TTFT proxy), prompt tokens
per second, generation tokens per second, and draft acceptance.

Reference numbers, 5090 (x4, 16 GT/s), this model, delta delivery off:

| | value |
|---|---|
| prefill, 78K-token prompt | 4,928 tok/s |
| decode, normal turns | 120-175 tok/s |
| MTP acceptance, normal turns | 58-93% |

The 4090 in the earlier runs was on x4 at 8 GT/s and decoded about 100-115 tok/s with 68% acceptance on average.
