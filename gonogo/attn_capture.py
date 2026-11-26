"""Step 5: block-level attention from the query/instruction rows, for every layer.

For each prompt we run one HF forward pass (sdpa). A wrapper around the sdpa attention
function additionally recomputes softmax(q k^T) for the *query rows only* (instruction
tokens after the last candidate + chat suffix), then sums the attention mass inside
each structural segment: [sink, template, prefix, h1, c1, ..., hN, cN, inst].

Saved per prompt (head-averaged, float32):
  last[L, S]  attention row of the final prompt token
  inst[L, S]  mean over all instruction rows
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, PREFILL, build_chat, build_user_content, load_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--specs", required=True)
ap.add_argument("--modes", default="A,C")
ap.add_argument("--caps", default="4:60,8:60,16:50,32:40", help="max problems per N (draw 0 only)")
ap.add_argument("--out", required=True)
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402

STATE = {"on": False}


def capture_sdpa(module, query, key, value, attention_mask, **kw):
    out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    if STATE["on"]:
        qi = STATE["q_idx"]                      # [M]
        bounds = STATE["bounds"]                 # [S+1]
        q = query[0, :, qi, :].float()           # H, M, D
        k = key[0].float()                       # Hkv, T, D
        H, M, D = q.shape
        Hkv, T, _ = k.shape
        q = q.view(Hkv, H // Hkv, M, D)
        sc = torch.einsum("kgmd,ktd->kgmt", q, k) * module.scaling
        causal = torch.arange(T, device=k.device)[None, :] > qi[:, None]
        sc = sc.masked_fill(causal, float("-inf"))
        p = sc.softmax(-1).reshape(H, M, T)
        cs = torch.nn.functional.pad(p.cumsum(-1), (1, 0))
        seg = cs[..., bounds[1:]] - cs[..., bounds[:-1]]          # H, M, S
        seg = seg.mean(0)                                         # M, S (head mean)
        STATE["last"][module.layer_idx] = seg[-1].cpu().numpy()
        STATE["inst"][module.layer_idx] = seg.mean(0).cpu().numpy()
        if STATE.get("keep_full"):
            STATE["full"][module.layer_idx] = p.mean(0)[-1].cpu()
    return out


AttentionInterface.register("sdpa", capture_sdpa)

here = os.path.dirname(os.path.abspath(__file__))
PROBS = {p["pid"]: p for p in json.load(open(os.path.join(here, "data/problems.json")))}
tok = AutoTokenizer.from_pretrained(MODELS[args.model])
model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                                             attn_implementation="sdpa").to(args.device).eval()
NL = model.config.num_hidden_layers


def prep(problem, texts, mode):
    content, spans = build_user_content(problem, texts, mode)
    full = build_chat(tok, content, enable_thinking=False, prefill=PREFILL[mode])
    off = full.index(content)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    starts = np.array([a for a, _ in enc["offset_mapping"]])
    T = len(starts)

    def tidx(ch):  # first token whose start >= ch
        return int(np.searchsorted(starts, ch, side="left"))

    names = ["sink", "tmpl"] + [n for n, _, _ in spans]
    b = [0, 1, tidx(off)] + [tidx(off + e) for _, _, e in spans[:-1]] + [T]
    inst_start = b[-2]
    return enc["input_ids"], names, b, inst_start


@torch.no_grad()
def run(ids, bounds, inst_start, keep_full=False):
    T = len(ids)
    STATE.update(on=True, q_idx=torch.arange(inst_start, T, device=args.device),
                 bounds=torch.tensor(bounds, device=args.device), last={}, inst={}, full={}, keep_full=keep_full)
    model(torch.tensor([ids], device=args.device), use_cache=False, logits_to_keep=1)
    STATE["on"] = False
    return (np.stack([STATE["last"][l] for l in range(NL)]), np.stack([STATE["inst"][l] for l in range(NL)]))


if args.selftest:
    # compare captured last-row attention with eager output_attentions on a short prompt
    texts = ["We compute 2+2=4. So \\boxed{4}.", "It is 5. \\boxed{5}", "Answer \\boxed{4}."]
    ids, names, bounds, inst_start = prep("What is 2+2?", texts, "A")
    last, inst = run(ids, bounds, inst_start, keep_full=True)
    m2 = AutoModelForCausalLM.from_pretrained(MODELS[args.model], torch_dtype=torch.float32,
                                              attn_implementation="eager").to(args.device).eval()
    with torch.no_grad():
        o = m2(torch.tensor([ids], device=args.device), output_attentions=True)
    for l in [0, 1, NL // 2, NL - 1]:
        ref = o.attentions[l][0, :, -1, :].mean(0).cpu()
        mine = STATE["full"][l]
        print(f"layer {l}: max|diff|={float((ref - mine).abs().max()):.2e} rowsum={float(mine.sum()):.4f}")
    print("segments:", list(zip(names, np.diff(bounds).tolist())))
    print("last-row seg mass L0:", np.round(last[0], 3).tolist(), "sum", float(last[0].sum()))
    sys.exit(0)

specs = load_jsonl(args.specs)
cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(os.path.join(here, f"out/graded_{args.model}.jsonl"))}
caps = {int(a): int(b) for a, b in (x.split(":") for x in args.caps.split(","))}
keep = {}
for s in specs:
    if s["draw"] != 0:
        continue
    pids = keep.setdefault(s["N"], [])
    if s["pid"] not in pids and len(pids) < caps.get(s["N"], 0):
        pids.append(s["pid"])
specs = [s for s in specs if s["draw"] == 0 and s["pid"] in keep.get(s["N"], [])]
print(f"{len(specs)} specs x modes {args.modes}", flush=True)

res = []
t0 = time.time()
for i, s in enumerate(specs):
    texts = [cands[(s["pid"], k)] for k in s["ks"]]
    for m in args.modes.split(","):
        ids, names, bounds, inst_start = prep(PROBS[s["pid"]]["problem"], texts, m)
        last, inst = run(ids, bounds, inst_start)
        res.append(dict(N=s["N"], draw=s["draw"], pid=s["pid"], pos=s["pos"], mode=m, names=names,
                        seglen=np.diff(bounds).astype(np.int32), last=last.astype(np.float32),
                        inst=inst.astype(np.float32), T=len(ids)))
    if i % 100 == 0:
        print(f"{i}/{len(specs)} {time.time()-t0:.0f}s", flush=True)
with open(args.out, "wb") as f:
    pickle.dump(res, f)
print(f"saved {len(res)} -> {args.out}", flush=True)
