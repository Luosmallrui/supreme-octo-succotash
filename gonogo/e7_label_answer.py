"""E7: does the selector bind printed labels to candidates? Label -> final-answer lookup on the E2 layouts.

For each E2 selection layout (8 candidates, printed labels permuted against the physical order) we ask for the final
answer of the candidate printed as "Solution L", without reasoning, with the assistant turn pre-filled as
"The final answer of Solution L is $\\boxed{". Two queries per layout, when the layout allows them:
  label   the smallest L >= 2 whose candidate sits neither in slot L nor in slot 1, and whose final answer differs from
          the answers of the candidates in slot L and in slot 1;
  label1  L = 1, when "Solution 1" is not printed on slot 1 and its answer differs from the slot-1 candidate's.
The reply is matched (after the same string normalisation that groups answers) against the candidate printed L
(reads labels), the candidate in ordinal slot L (counts blocks), the slot-1 candidate (primacy), or another candidate.

  python e7_label_answer.py --manifest out/e2_position_label/manifest_qwen3-8b.jsonl --model qwen3-8b \\
      --out out/e7_label_answer/e7_qwen3-8b.jsonl            # GPU (vLLM)
  python e7_label_answer.py --manifest ... --model qwen3-8b --dry-run      # build queries only
  python e7_label_answer.py --analyze out/e7_label_answer/e7_*.jsonl --output out/e7_label_answer/report_e7.md
"""
import argparse
import json
import os
import zlib
from collections import defaultdict
from pathlib import Path

from common import MODELS, build_chat, load_jsonl, norm_key, parse_prefilled_boxed, vllm_prompt

HERE = Path(__file__).resolve().parent
SELECT_INSTRUCTION = "Which solution is correct?"


def queries(rows, keys):
    """(row, query type, L) for every layout that admits the query."""
    out = []
    for r in rows:
        order, labels = r["order"], r["labels"]
        slot_of = {lab: i + 1 for i, lab in enumerate(labels)}
        key_at = lambda slot: keys[r["pid"], order[slot - 1]]
        for L in range(2, 9):
            s = slot_of[L]
            if s not in (L, 1) and key_at(s) != key_at(L) and key_at(s) != key_at(1):
                out.append((r, "label", L))
                break
        s = slot_of[1]
        if s != 1 and key_at(s) != key_at(1):
            out.append((r, "label1", 1))
    return out


def prompt_text(r, L, probs, cands):
    from e2_run import user_prompt
    content = user_prompt(dict(r, task="select"), probs[r["pid"]], cands)
    i = content.rindex(SELECT_INSTRUCTION)
    return content[:i] + (f"What final answer does the candidate printed as Solution {L} give? Labels may appear in "
                          f"any order. Reply with only that final answer in the form \\boxed{{answer}}.")


def prefill(L):
    return f"The final answer of Solution {L} is $\\boxed{{"


def classify(r, L, answer, keys):
    order, labels = r["order"], r["labels"]
    slot = labels.index(L) + 1
    k_label, k_ord, k_first = order[slot - 1], order[L - 1], order[0]
    a = norm_key(answer) if answer is not None else None
    match = lambda k: a is not None and a == keys[r["pid"], k]
    in_set = [k for k in order if match(k)]
    return dict(L=L, label_slot=slot, k_label=k_label, k_ordinal=k_ord, k_first=k_first,
                chance_label=sum(keys[r["pid"], k] == keys[r["pid"], k_label] for k in order) / len(order),
                match_label=match(k_label), match_ordinal=match(k_ord), match_first=match(k_first),
                match_other=bool(in_set) and not (match(k_label) or match(k_ord) or match(k_first)),
                match_none=not in_set)


def run(args):
    rows = [r for r in load_jsonl(args.manifest) if r["model"] == args.model and r["task"] == "select"]
    graded = load_jsonl(HERE / "out" / f"graded_{args.model}.jsonl")
    keys = {(g["pid"], g["k"]): g["key"] for g in graded}
    cands = {(g["pid"], g["k"]): g["text"] for g in graded}
    probs = {p["pid"]: p["problem"] for p in json.loads((HERE / "data/problems.json").read_text(encoding="utf-8"))}
    qs = queries(rows, keys)
    judge = (args.judge_name or Path(args.judge).name.lower()) if args.judge else None
    print(f"{judge or args.model} on {args.model}: {len(qs)} queries from {len(rows)} layouts "
          f"({sum(q[1] == 'label' for q in qs)} label, {sum(q[1] == 'label1' for q in qs)} label1)", flush=True)
    if args.dry_run:
        r, kind, L = qs[0]
        text = prompt_text(r, L, probs, cands)
        print(text[:300] + "\n...\n" + text[-320:] + "\n[assistant] " + prefill(L))
        return
    out = Path(args.out).resolve()   # vLLM may change the working directory
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["request_id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [q for q in qs if f"{q[0]['request_id']}/{q[1]}" not in done]
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODELS.get(args.judge or args.model, args.judge or args.model), max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    with out.open("a", encoding="utf-8") as sink:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            prompts = [vllm_prompt(tok, build_chat(tok, prompt_text(r, L, probs, cands), enable_thinking=False,
                                                   prefill=prefill(L))) for r, _, L in chunk]
            for (r, kind, L), o in zip(chunk, llm.generate(prompts, sp, use_tqdm=False), strict=True):
                text = o.outputs[0].text
                answer = parse_prefilled_boxed(text)
                rec = dict(request_id=f"{r['request_id']}/{kind}", model=args.model, pid=r["pid"], set_id=r["set_id"],
                           stratum=r["stratum"], query=kind, response=text[:200], answer=answer,
                           **classify(r, L, answer, keys))
                if judge:
                    rec["judge"] = judge
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            print(f"saved {len(done) + start + len(chunk)}/{len(qs)}", flush=True)


def analyze(paths, output, B):
    import numpy as np
    from e2_analyze import cluster_mean
    groups = defaultdict(list)
    for path in paths:
        for r in load_jsonl(path):
            groups[r.get("judge", r["model"]), r["model"], r["query"]].append(r)
    lines = ["# E7 打印编号 → 最终答案的查找（direct）", "",
             "由 `e7_label_answer.py --analyze` 生成。要求给出印着 “Solution L” 的候选的最终答案；L 的选取使该候选既不在第 L 个位置、"
             "也不在第 1 个位置，且三者答案不同。按答案字符串（与分组相同的归一化）匹配。“随机候选”为若随机报一个候选的答案时与目标一致的期望比例。"
             f"区间按原题整群重采样 {B} 次。", "",
             "| 评审 | 候选来自 | 查询 | 次数 | 印着 L 的候选 (%) | 随机候选 (%) | 第 L 个位置的候选 (%) | 第 1 个位置的候选 (%) | 其他候选 (%) | 无法匹配 (%) |",
             "|---|---|---|---:|---|---:|---|---|---:|---:|"]
    for (judge, model, query), rs in sorted(groups.items()):
        vals = [[r["match_label"], r["match_ordinal"], r["match_first"], r["match_other"], r["match_none"]] for r in rs]
        seed = zlib.crc32(f"e7|{judge}|{model}|{query}".encode())
        est, lo, hi = cluster_mean([r["pid"] for r in rs], vals, B, np.random.default_rng(seed))
        ci = lambda j: f"{100 * est[j]:.1f} [{100 * lo[j]:.1f}, {100 * hi[j]:.1f}]"
        lines.append(f"| {judge} | {model} | {query} | {len(rs)} | {ci(0)} | {100 * np.mean([r['chance_label'] for r in rs]):.1f} | "
                     f"{ci(1)} | {ci(2)} | {100 * est[3]:.1f} | {100 * est[4]:.1f} |")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--model", choices=MODELS, help="candidate generator (manifest rows, candidates)")
    ap.add_argument("--judge", help="model that answers (default: --model); a MODELS key or a local path")
    ap.add_argument("--judge-name")
    ap.add_argument("--out")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", type=Path, nargs="+", help="E7 outputs to summarise (no GPU)")
    ap.add_argument("--output", help="report path for --analyze")
    ap.add_argument("--B", type=int, default=5000)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.analyze, args.output or HERE / "out/e7_label_answer/report_e7.md", args.B)
    if not (args.manifest and args.model) or not (args.out or args.dry_run):
        ap.error("--manifest, --model and --out (or --dry-run) are required to run")
    run(args)


if __name__ == "__main__":
    main()
