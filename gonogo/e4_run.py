"""E4: where does primacy come from? A 2x2 intervention on the E2 layouts (direct readout, HF forward pass).

Both factors act only on the N candidate blocks ("Solution k:" header + text); the problem prefix and the
instruction / readout tokens are untouched.

  mask  causal    standard causal attention
        isolated  a candidate block attends only to the prefix and to its own earlier tokens; the instruction and
                  readout tokens still attend to everything before them
        bidir     (follow-up, --conditions bidir:seq,bidir:shared) a candidate block attends to every other block in
                  both directions, as in PINE's inter-document attention; attention inside a block stays causal.
                  With pos=shared the readout is again order-invariant, but blocks can still compare each other.
  pos   seq       standard position ids 0..L-1
        shared    every block starts at the same position id (right after the prefix) and the instruction continues
                  after the longest block (--align right: every block ends at the same position id instead)

With mask=isolated and pos=shared every block is encoded exactly as if it were the only one, and the readout sees the
blocks as an unordered set (as in Set-Based Prompting / PINE), so the choice cannot depend on block order; any
remaining preference reflects the printed labels. If isolation alone removes primacy, primacy arises while later
blocks are encoded in the context of earlier ones; if shared position ids alone remove it, it is carried by the
positional encoding.

Readout as in E2 direct: assistant turn pre-filled with "The best solution is Solution ", next-token distribution over
the printed labels 1..8.

  python e4_run.py --manifest out/e2_position_label/manifest_qwen3-8b.jsonl --model qwen3-8b --selftest
  python e4_run.py --manifest ... --model qwen3-8b --out out/e4_mask_position/e4_qwen3-8b.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from common import MODELS, build_chat, build_user_content, load_jsonl

HERE = Path(__file__).resolve().parent
CONDITIONS = [("causal", "seq"), ("isolated", "seq"), ("causal", "shared"), ("isolated", "shared")]
PREFILL = "The best solution is Solution "


# ---------------------------------------------------------------------------
# Prompt layout (pure Python / NumPy, testable without a model)
# ---------------------------------------------------------------------------

def pieces(row, problem, candidates):
    """[(kind, text)] with kind in {"prefix", "block", "inst"}; joins to exactly e2_run.user_prompt(row, ...)."""
    order, labels, n = row["order"], row["labels"], row["N"]
    if row["task"] != "select" or sorted(labels) != list(range(1, n + 1)) or len(set(order)) != n:
        raise ValueError(f"Not an E2 selection layout: {row['request_id']}")
    content, spans = build_user_content(problem, [candidates[row["pid"], k] for k in order], "A")
    out, header = [], None
    for name, start, end in spans:
        if name == "prefix":
            out.append(("prefix", content[start:end]))
        elif name.startswith("h"):
            header = f"Solution {labels[int(name[1:]) - 1]}:\n"
        elif name.startswith("c"):
            out.append(("block", header + content[start:end]))
        elif name == "inst":
            out.append(("inst", "Which solution is correct? Return the printed Solution label of the best solution."
                        f" Labels may appear in any order. Reply with only the printed number in the form "
                        f"\\boxed{{k}}, where k is an integer from 1 to {n}."))
    return out


def block_bounds(offsets, content_start, parts):
    """Token index bounds [b_0, b_1, ..., b_N] of the N blocks, from tokenizer character offsets."""
    starts = np.array([a for a, _ in offsets])
    pos, bounds = content_start, []
    for kind, text in parts:
        if kind == "block" and not bounds:
            bounds.append(int(np.searchsorted(starts, pos, side="left")))
        pos += len(text)
        if kind == "block":
            bounds.append(int(np.searchsorted(starts, pos, side="left")))
    if len(bounds) < 2 or any(b >= a for a, b in zip(bounds[1:], bounds)):
        raise ValueError(f"Degenerate block bounds {bounds}")
    return bounds


def block_ids(bounds, length):
    """Per-token block index: -1 before the first block (prefix), -2 after the last (instruction/readout)."""
    blk = np.full(length, -1, dtype=np.int64)
    blk[bounds[-1]:] = -2
    for b, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        blk[s:e] = b
    return blk


def position_ids(bounds, length, shared, align="left"):
    pos = np.arange(length, dtype=np.int64)
    if not shared:
        return pos
    start, lens = bounds[0], np.diff(bounds)
    longest = int(lens.max())
    for s, e in zip(bounds[:-1], bounds[1:]):
        offset = start if align == "left" else start + longest - (e - s)
        pos[s:e] = offset + np.arange(e - s)
    pos[bounds[-1]:] = start + longest + np.arange(length - bounds[-1])
    return pos


MASKS = ("causal", "isolated", "bidir")


def allowed(blk_q, blk_k, q_idx, k_idx, mask):
    """Boolean attention pattern for query rows q_idx and key columns k_idx (True = may attend).

    causal    standard causal attention
    isolated  no attention between different candidate blocks
    bidir     candidate blocks attend to every other candidate block in both directions (PINE-style inter-document
              attention); attention within a block, and everything outside the blocks, stays causal
    """
    if mask not in MASKS:
        raise ValueError(f"Unknown mask {mask}")
    ok = k_idx[None, :] <= q_idx[:, None]
    bq, bk = blk_q[:, None], blk_k[None, :]
    cross = (bq >= 0) & (bk >= 0) & (bq != bk)
    if mask == "isolated":
        ok = ok & ~cross
    elif mask == "bidir":
        ok = ok | cross
    return ok


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------

STATE = {"mask": None}


def install_attention():
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def masked_sdpa(module, query, key, value, attention_mask, **kw):
        if STATE["mask"] is not None:
            attention_mask = STATE["mask"]
            kw["is_causal"] = False
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register("sdpa", masked_sdpa)


def additive_mask(blk, mask_kind, dtype, device, chunk=4096):
    """[1, 1, L, L] additive mask (0 / dtype min), allocated with an 8-aligned row stride for the SDPA kernels."""
    import torch
    L = len(blk)
    Lp = (L + 7) // 8 * 8
    mask = torch.full((Lp, Lp), torch.finfo(dtype).min, dtype=dtype, device=device)
    b = torch.as_tensor(blk, device=device)
    idx = torch.arange(L, device=device)
    for s in range(0, L, chunk):
        q = idx[s:s + chunk]
        ok = allowed(b[q], b, q, idx, mask_kind)
        mask[s:s + len(q), :L].masked_fill_(ok, 0.0)
    return mask[:L, :L][None, None]


class Selector:
    def __init__(self, model_name, device="cuda", align="left"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        install_attention()
        self.torch, self.device, self.align = torch, device, align
        self.tok = AutoTokenizer.from_pretrained(MODELS[model_name])
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], torch_dtype=dtype,
                                                          attn_implementation="sdpa").to(device).eval()
        self.dtype = dtype
        self.digits = []
        for d in range(1, 10):
            ids = self.tok.encode(str(d), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Digit {d} is not a single token")
            self.digits.append(ids[0])

    def encode(self, row, problem, candidates):
        from e2_run import user_prompt
        parts = pieces(row, problem, candidates)
        content = "".join(t for _, t in parts)
        if content != user_prompt(row, problem, candidates):
            raise ValueError(f"E4 prompt differs from the E2 prompt for {row['request_id']}")
        full = build_chat(self.tok, content, enable_thinking=False, prefill=PREFILL)
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        bounds = block_bounds(enc["offset_mapping"], full.index(content), parts)
        if len(bounds) != row["N"] + 1:
            raise ValueError(f"Expected {row['N']} blocks in {row['request_id']}")
        return enc["input_ids"], bounds

    def label_distribution(self, ids, bounds, mask_kind, shared, stock=False):
        """Distribution over printed labels 1..N at the readout position, plus the raw digit mass."""
        torch = self.torch
        L, n = len(ids), len(bounds) - 1
        pos = position_ids(bounds, L, shared, self.align)
        with torch.no_grad():
            STATE["mask"] = None if stock else additive_mask(block_ids(bounds, L), mask_kind, self.dtype, self.device)
            try:
                out = self.model.model(input_ids=torch.tensor([ids], device=self.device),
                                       position_ids=torch.as_tensor(pos[None], device=self.device), use_cache=False)
            finally:
                STATE["mask"] = None
            logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()[0]
        p = logits.softmax(-1)[self.digits[:n]].cpu().numpy().astype(float)
        mass = float(p.sum())
        return (p / mass).tolist(), mass


def run_rows(sel, rows, probs, cands, sink, done, conditions=CONDITIONS):
    t0 = time.time()
    for i, r in enumerate(rows):
        todo = [c for c in conditions if (r["request_id"], *c) not in done]
        if not todo:
            continue
        ids, bounds = sel.encode(r, probs[r["pid"]], cands)
        for mask, pos in todo:
            p, mass = sel.label_distribution(ids, bounds, mask, pos == "shared")
            label = int(np.argmax(p)) + 1
            slot = r["labels"].index(label)
            cand = r["order"][slot]
            rec = dict(r, mode="direct", mask=mask, pos=pos, align=sel.align, p_label=[round(x, 6) for x in p],
                       p_mass=round(mass, 4), choice_label=label, choice_slot=slot + 1, choice_candidate=cand,
                       correct=bool(r["cor"][str(cand)]), n_tokens=len(ids),
                       block_lens=np.diff(bounds).tolist())
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
        if i % 25 == 0:
            print(f"{i + 1}/{len(rows)} prompts, {time.time() - t0:.0f}s", flush=True)


INVARIANT = {("isolated", "shared"), ("bidir", "shared")}   # order-invariant by construction


def pattern_ok():
    """Mask sanity on a toy layout: prefix 0-1, block 0 = 2-3, block 1 = 4-6, instruction 7-8."""
    blk = block_ids([2, 4, 7], 9)
    q = np.arange(9)
    iso = allowed(blk, blk, q, q, "isolated")
    bid = allowed(blk, blk, q, q, "bidir")
    cau = allowed(blk, blk, q, q, "causal")
    return bool(not iso[5, 3] and iso[5, 1] and iso[8, 3] and iso[6, 4]            # isolated: no block-to-block
                and bid[3, 5] and bid[5, 3] and not bid[2, 3] and not bid[5, 8]    # bidir: both ways, causal inside
                and not bid[1, 2] and bid[8, 3]
                and (cau == (q[None, :] <= q[:, None])).all())


def selftest(sel, rows, probs, cands, conditions=CONDITIONS):
    """Fails loudly if the intervention does not do what it claims on this model and transformers version."""
    ok = True
    r = min(rows, key=lambda x: sum(len(cands[x["pid"], k]) for k in x["order"]))
    ids, bounds = sel.encode(r, probs[r["pid"]], cands)
    print(f"selftest prompt {r['request_id']}: {len(ids)} tokens, block lengths {np.diff(bounds).tolist()}")
    stock, _ = sel.label_distribution(ids, bounds, "causal", False, stock=True)
    ours, mass = sel.label_distribution(ids, bounds, "causal", False)
    d = max(abs(a - b) for a, b in zip(stock, ours))
    print(f"[1] explicit causal mask vs stock forward: max |dp| = {d:.2e} (digit mass {mass:.3f})")
    ok &= d < 2e-2
    # Reverse the block order (labels stay with their candidates): the INVARIANT conditions must not change.
    rev = dict(r, order=r["order"][::-1], labels=r["labels"][::-1], request_id=r["request_id"] + "/reversed")
    ids2, bounds2 = sel.encode(rev, probs[r["pid"]], cands)
    for mask, pos in conditions:
        a, _ = sel.label_distribution(ids, bounds, mask, pos == "shared")
        b, _ = sel.label_distribution(ids2, bounds2, mask, pos == "shared")
        d = max(abs(x - y) for x, y in zip(a, b))
        must = (mask, pos) in INVARIANT
        print(f"[2] {mask:8s} {pos:6s} reversed order: max |dp| over labels = {d:.3f}"
              + ("  (must be ~0)" if must else ""))
        if must:
            ok &= d < 3e-2
    p_ok = pattern_ok()
    ok &= p_ok
    print(f"[3] mask pattern sanity (isolated, bidir, causal): {'ok' if p_ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="E2 manifest (selection rows are used)")
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--align", choices=("left", "right"), default="left")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--conditions", default=",".join(f"{m}:{p}" for m, p in CONDITIONS),
                    help="comma list of mask:pos, mask in causal/isolated/bidir, pos in seq/shared "
                         "(default: the original 2x2; the PINE-style follow-up uses bidir:seq,bidir:shared)")
    args = ap.parse_args()
    conditions = [tuple(c.split(":")) for c in args.conditions.split(",")]
    if any(m not in MASKS or p not in ("seq", "shared") for m, p in conditions):
        ap.error(f"bad --conditions {args.conditions}")
    rows = [r for r in load_jsonl(args.manifest) if r["model"] == args.model and r["task"] == "select"]
    if args.limit:
        rows = rows[:args.limit]
    probs = {p["pid"]: p["problem"] for p in json.loads((HERE / "data/problems.json").read_text())}
    cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(HERE / "out" / f"graded_{args.model}.jsonl")}
    sel = Selector(args.model, align=args.align)
    if args.selftest:
        sys.exit(0 if selftest(sel, rows, probs, cands, conditions) else 1)
    if args.out is None:
        ap.error("--out is required unless --selftest")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for rec in load_jsonl(args.out):
            if rec.get("align") != args.align:
                raise ValueError(f"{args.out} holds results for a different --align")
            done.add((rec["request_id"], rec["mask"], rec["pos"]))
    print(f"{args.model}: {len(rows)} layouts x {len(conditions)} conditions, {len(done)} already saved", flush=True)
    with args.out.open("a", encoding="utf-8") as sink:
        run_rows(sel, rows, probs, cands, sink, done, conditions)


if __name__ == "__main__":
    main()
