"""Why does hosted Llama-3.1-8B (OpenRouter, DeepInfra) prefer the label "Solution 1" when the local run does not?

The E2 run through OpenRouter (api_select.py, e2_llama-3.1-8b_direct.jsonl) and the local float32 forward pass
(e4_run.py, unmodified condition) send the same user prompt and pre-fill, but the hosted prompts are 26 tokens
shorter: DeepInfra's template omits the system turn that Llama-3.1's reference template adds by default
("Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024") and trims the pre-fill's final space.
This probe restores that turn as an explicit system message on the same provider and compares the three runs on
the same one-correct 2x2 blocks (greedy label within 16 tokens; DeepInfra returns no log-probabilities).

  python llama_system_probe.py run --cands ../out/graded_qwen3-8b.jsonl --key-file ../../openrouter.txt --blocks 50
  python llama_system_probe.py compare      (writes out/e2_position_label/api/llama_system_probe/report.md)
"""
import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from e2_run import user_prompt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out/e2_position_label/api/llama_system_probe"
PREFILL = "The best solution is Solution "
SYSTEM = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
LOCAL = HERE / "out/e4_mask_position/judges/e4_llama-3.1-8b-instruct_on_qwen3-8b.jsonl"
HOSTED = HERE / "out/e2_position_label/api/e2_llama-3.1-8b_direct.jsonl"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def post(key, body):
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers={"Authorization": "Bearer " + key,
                                                         "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code not in (429, 500, 502, 503, 504):
                return {"error": f"{e.code} {msg}"}
        except Exception:
            pass
        time.sleep(10 * 2 ** attempt)
    return {"error": "gave up after retries"}


def blocks_of(rows):
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["set_id"], r["probe_slot"], r["a"], r["b"]].append(r)
    return blocks


def run(args):
    key = Path(args.key_file).read_text(encoding="utf-8").strip()
    probs = {p["pid"]: p["problem"] for p in json.loads((HERE / "data/problems.json").read_text(encoding="utf-8"))}
    cands = {(r["pid"], r["k"]): r["text"] for r in load(args.cands)}
    rows = [r for r in load(HERE / "out/e2_position_label/manifest_qwen3-8b.jsonl")
            if r["model"] == "qwen3-8b" and r["task"] == "select" and r["stratum"] == "one_correct_pair"]
    blocks = blocks_of(rows)
    keys = sorted(blocks)
    random.Random(0).shuffle(keys)
    rows = [r for k in keys[:args.blocks] for r in blocks[k]]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "probe_chatsys.jsonl"
    done = {j["request_id"] for j in load(path) if "text" in j} if path.exists() else set()
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            if r["request_id"] in done:
                continue
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(r, probs[r["pid"]], cands)},
                    {"role": "assistant", "content": PREFILL}]
            js = post(key, {"model": "meta-llama/llama-3.1-8b-instruct", "messages": msgs, "max_tokens": 16,
                            "temperature": 0, "provider": {"order": ["deepinfra"], "allow_fallbacks": False}})
            rec = dict(request_id=r["request_id"], variant="chatsys")
            if "choices" in js:
                rec.update(text=js["choices"][0]["message"].get("content"), usage=js.get("usage"), provider=js.get("provider"))
            else:
                rec["error"] = str(js.get("error"))[:300]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def compare(args):
    local = {r["request_id"]: r for r in load(LOCAL) if r["mask"] == "causal" and r["pos"] == "seq"}
    hosted = {r["request_id"]: r for r in load(HOSTED) if "error" not in r and r.get("task") == "select"}
    probe, cost = {}, 0.0
    for j in load(OUT / "probe_chatsys.jsonl"):
        m = re.match(r"\s*(\d+)", j.get("text") or "")
        if m:
            probe[j["request_id"]] = int(m.group(1))
            cost += (j.get("usage") or {}).get("cost", 0.0)
    runs = {"本地 float32，参考 chat template（E4 不干预条件）": {k: local[k]["choice_label"] for k in probe},
            "OpenRouter/DeepInfra，无 system 轮（论文 E2 运行）": {k: hosted[k]["choice_label"] for k in probe},
            "OpenRouter/DeepInfra，补回默认 system 轮（本探针）": probe}
    by_block = defaultdict(list)
    for k in probe:
        by_block[k.rsplit("/", 1)[0]].append(k)
    full = sorted(b for b, ks in by_block.items() if len(ks) == 4)
    cand = lambda r, lab: r["order"][r["labels"].index(lab)]

    def effects(ch):
        out = {}
        for b in full:
            m = {(local[k]["a_first"], local[k]["a_label1"]): float(cand(local[k], ch[k]) == local[k]["a"]) for k in by_block[b]}
            out[b] = ((m[True, True] + m[True, False] - m[False, True] - m[False, False]) / 2,
                      (m[True, True] + m[False, True] - m[True, False] - m[False, False]) / 2)
        return out

    E = {n: effects(ch) for n, ch in runs.items()}
    pids = sorted({local[by_block[b][0]]["pid"] for b in full})
    pb = {p: [b for b in full if local[by_block[b][0]]["pid"] == p] for p in pids}
    draws = np.random.default_rng(0).integers(0, len(pids), (5000, len(pids)))

    def ci(f):
        bs = [np.mean([f(b) for i in d for b in pb[pids[i]]]) for d in draws]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"{100 * np.mean([f(b) for b in full]):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"

    names = list(runs)
    loc = runs[names[0]]
    lines = ["# Llama-3.1-8B：托管与本地的编号效应差异来自哪里", "",
             "由 `llama_system_probe.py compare` 生成。50 个 one_correct 四格区组（A 正确、B 错误），各取三种运行在同一批布局上的选择。"
             "效应为区组内对另一因素等权平均的 P(选 A) 差，单位 pp；区间按原题整群重采样 5000 次。", "",
             "| 运行 | 位置效应 | 编号 1 效应 | 选编号 1 (%) | 与本地一致 (%) |", "|---|---|---|---:|---:|"]
    for n in names:
        ks = [k for b in full for k in by_block[b]]
        lines.append(f"| {n} | {ci(lambda b, n=n: E[n][b][0])} | {ci(lambda b, n=n: E[n][b][1])} | "
                     f"{100 * np.mean([runs[n][k] == 1 for k in ks]):.1f} | {100 * np.mean([runs[n][k] == loc[k] for k in ks]):.1f} |")
    H, S, L = E[names[1]], E[names[2]], E[names[0]]
    lines += ["", "配对差：",
              f"- 编号效应，补回 system − 无 system：{ci(lambda b: S[b][1] - H[b][1])}",
              f"- 编号效应，补回 system − 本地：{ci(lambda b: S[b][1] - L[b][1])}",
              f"- 位置效应，补回 system − 本地：{ci(lambda b: S[b][0] - L[b][0])}", "",
              f"区组 {len(full)}，原题 {len(pids)}，探针调用 {len(probe)} 次，费用 ${cost:.4f}。",
              "提示长度：同一布局本地 8,641 token，无 system 的托管请求 8,615，补回 system 后 8,639（差 2 个 token：system 末尾换行与预填末尾空格被模板裁掉）。", "",
              "全部 944 个布局上的本地与托管对比见 `compare_llama_local.md`：由 `e2_compare.py --reference llama-3.1-8b` 生成，"
              "本地一行取 E4 文件中 mask=causal、pos=seq 的行。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cands", required=True, help="graded_qwen3-8b.jsonl")
    r.add_argument("--key-file", required=True)
    r.add_argument("--blocks", type=int, default=50)
    sub.add_parser("compare")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
