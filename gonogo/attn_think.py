"""Step 6: attention from the reasoning tokens of thinking-mode selection to each candidate block.

For each (prompt, reasoning trace) from run_select.py --think we run one HF forward pass over
prompt + trace. The trace is cut right after '\\boxed{' of the final answer when the run finished,
so the last token is the one that predicts the chosen index ("decision row").

Query rows: every --stride-th reasoning token (between '<think>' and '</think>') plus the decision row.
For every row and layer we sum attention mass inside each segment
[sink, tmpl, prefix, h1, c1, ..., hN, cN, inst, gen] (head mean), then average reasoning rows inside
--nbins equal bins of reasoning progress.

Saved per sequence (float32):
  bins[L, nbins, S]   mean over reasoning rows in each progress bin
  dec[L, S]           decision row, or None when the run did not finish with a boxed index
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, build_chat, build_user_content, load_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--sel", help="selT jsonl with full traces ('gen')")
ap.add_argument("--caps", default="4:30,8:30,16:20", help="max problems per N")
ap.add_argument("--stride", type=int, default=16)
ap.add_argument("--nbins", type=int, default=4)
ap.add_argument("--chunk", type=int, default=128, help="query rows per softmax chunk")
ap.add_argument("--out")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402

STATE = {"on": False}


def capture_sdpa(module, query, key, value, attention_mask, **kw):
    out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    if STATE["on"]:
        qi, bounds = STATE["q_idx"], STATE["bounds"]
        k = key[0].float()                                   # Hkv, T, D
        Hkv, T, D = k.shape
        H = query.shape[1]
        pos = torch.arange(T, device=k.device)
        segs, full = [], []
        for c0 in range(0, len(qi), args.chunk):
            qc = qi[c0:c0 + args.chunk]
            q = query[0, :, qc, :].float().view(Hkv, H // Hkv, len(qc), D)
            sc = torch.einsum("kgmd,ktd->kgmt", q, k) * module.scaling
            sc = sc.masked_fill(pos[None, :] > qc[:, None], float("-inf"))
            p = sc.softmax(-1).reshape(H, len(qc), T)
            cs = torch.nn.functional.pad(p.cumsum(-1), (1, 0))
            segs.append((cs[..., bounds[1:]] - cs[..., bounds[:-1]]).mean(0))   # M, S
            if STATE.get("keep_full"):
                full.append(p.mean(0).cpu())
            del sc, p, cs
        STATE["seg"][module.layer_idx] = torch.cat(segs).cpu().numpy()
        if STATE.get("keep_full"):
            STATE["full"][module.layer_idx] = torch.cat(full)
    return out


AttentionInterface.register("sdpa", capture_sdpa)

here = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained(MODELS[args.model])
dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], torch_dtype=dtype,
                                             attn_implementation="sdpa").to(args.device).eval()
NL = model.config.num_hidden_layers


def prep(problem, texts, gen, mode="A"):
    """Build prompt + trace. Returns ids, names, bounds, reasoning row range, decision row (or None)."""
    content, spans = build_user_content(problem, texts, mode)
    prompt = build_chat(tok, content, enable_thinking=True)
    j = gen.find("</think>")
    dec_char = None
    if j >= 0:
        bi = gen.rfind("\\boxed{", j)
        if bi >= 0:
            gen = gen[: bi + len("\\boxed{")]        # last token predicts the chosen index
            dec_char = True
        else:
            gen = gen[: j + len("</think>")]
    full = prompt + gen
    off = full.index(content)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    starts = np.array([a for a, _ in enc["offset_mapping"]])
    T = len(starts)

    def tidx(ch):
        return int(np.searchsorted(starts, ch, side="left"))

    names = ["sink", "tmpl"] + [n for n, _, _ in spans] + ["gen"]
    bounds = [0, 1, tidx(off)] + [tidx(off + e) for _, _, e in spans[:-1]] + [tidx(len(prompt)), T]
    think_open = full.find("<think>", len(prompt))
    r0 = tidx(think_open + len("<think>")) if think_open >= 0 else tidx(len(prompt))
    r1 = tidx(len(prompt) + j) if j >= 0 else T
    dec = T - 1 if dec_char else None
    return enc["input_ids"], names, bounds, (r0, r1), dec


@torch.no_grad()
def run(ids, bounds, rows, keep_full=False):
    STATE.update(on=True, q_idx=torch.tensor(rows, device=args.device), bounds=torch.tensor(bounds, device=args.device),
                 seg={}, full={}, keep_full=keep_full)
    model(torch.tensor([ids], device=args.device), use_cache=False, logits_to_keep=1)
    STATE["on"] = False
    return np.stack([STATE["seg"][l] for l in range(NL)])     # L, M, S


def capture(problem, texts, gen):
    ids, names, bounds, (r0, r1), dec = prep(problem, texts, gen)
    rows = list(range(r0, r1, args.stride))
    if not rows:
        return None
    seg = run(ids, bounds, rows + ([dec] if dec is not None else []))
    rs = seg[:, : len(rows)]
    prog = (np.array(rows) - r0) / max(r1 - r0, 1)
    b = np.minimum(args.nbins - 1, (prog * args.nbins).astype(int))
    bins = np.stack([rs[:, b == i].mean(1) if (b == i).any() else np.full(rs.shape[::2], np.nan) for i in range(args.nbins)], 1)
    return dict(names=names, seglen=np.diff(bounds).astype(np.int32), bins=bins.astype(np.float32),
                dec=None if dec is None else seg[:, -1].astype(np.float32), T=len(ids), n_reason=r1 - r0)


if args.selftest:
    texts = ["We compute 2+2=4. So \\boxed{4}.", "It is 5. \\boxed{5}", "Answer \\boxed{4}."]
    gen = "<think>\nSolution 1 and 3 both get 4, solution 2 says 5, which is wrong.\n</think>\n\nThe best is \\boxed{1}."
    ids, names, bounds, (r0, r1), dec = prep("What is 2+2?", texts, gen)
    rows = list(range(r0, r1, 3)) + [dec]
    seg = run(ids, bounds, rows, keep_full=True)
    print("tokens", len(ids), "reasoning rows", (r0, r1), "decision row", dec, "->", repr(tok.decode(ids[dec:])))
    print("reasoning text:", repr(tok.decode(ids[r0:r1])))
    m2 = AutoModelForCausalLM.from_pretrained(MODELS[args.model], torch_dtype=torch.float32,
                                              attn_implementation="eager").to(args.device).eval()
    with torch.no_grad():
        o = m2(torch.tensor([ids], device=args.device), output_attentions=True)
    for l in [0, 1, NL // 2, NL - 1]:
        ref = o.attentions[l][0][:, rows, :].mean(0).cpu()
        mine = STATE["full"][l]
        b = torch.tensor(bounds)
        ref_seg = torch.stack([ref[:, b[i]:b[i + 1]].sum(1) for i in range(len(b) - 1)], 1).numpy()
        print(f"layer {l}: max|row diff|={float((ref - mine).abs().max()):.2e}  max|segment diff|={np.abs(ref_seg - seg[l]).max():.2e}"
              f"  rowsums {float(mine.sum(1).min()):.4f}-{float(mine.sum(1).max()):.4f}")
    print("segments:", list(zip(names, np.diff(bounds).tolist())))
    sys.exit(0)

PROBS = {p["pid"]: p for p in __import__("json").load(open(os.path.join(here, "data/problems.json")))}
cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(os.path.join(here, f"out/graded_{args.model}.jsonl"))}
sel = [r for r in load_jsonl(args.sel) if r.get("gen")]
caps = {int(a): int(b) for a, b in (x.split(":") for x in args.caps.split(","))}
keep = {}
for r in sel:
    pids = keep.setdefault(r["N"], [])
    if r["pid"] not in pids and len(pids) < caps.get(r["N"], 0):
        pids.append(r["pid"])
sel = [r for r in sel if r["pid"] in keep.get(r["N"], [])]
print(f"{len(sel)} sequences ({ {N: len(v) for N, v in keep.items()} } problems per N)", flush=True)

res = []
t0 = time.time()
for i, r in enumerate(sel):
    texts = [cands[(r["pid"], k)] for k in r["ks"]]
    c = capture(PROBS[r["pid"]]["problem"], texts, r["gen"])
    if c is not None:
        res.append(dict(N=r["N"], draw=r["draw"], pid=r["pid"], pos=r["pos"], mode="T", chosen=r["chosen"],
                        correct=r["correct"], finished="</think>" in r["gen"], **c))
    if i % 50 == 0:
        print(f"{i}/{len(sel)} {time.time()-t0:.0f}s T={c['T'] if c else '-'}", flush=True)
        with open(args.out, "wb") as f:   # checkpoint
            pickle.dump(res, f)
with open(args.out, "wb") as f:
    pickle.dump(res, f)
print(f"saved {len(res)} -> {args.out}", flush=True)
