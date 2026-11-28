"""Pre-registered confirmation of the selection rules on another judge (plan: out/e6_duels/prereg_llama.md).

Runs the eight registered tests on one judge's natural-set data, with the analysis code frozen at the commit that
registered the plan. Nothing here is tuned on the confirmation data; every rule, fold, seed and threshold is fixed.

Inputs (one judge, selecting among --model's candidates on --model's natural sets):
  --calls    listwise E1 calls, direct, pre-filled label distribution (e1_run.py)
  --duels    order-balanced answer duels (e6_run.py)
  --allcand  order-balanced duels between all candidates of the N = 8 sets (e6_run.py --all-candidates --n 8)

Registered tests (difference in accuracy, pp; 95% intervals Bonferroni-corrected for 8 tests, i.e. 99.375%,
from 20,000 bootstrap draws over problems; a test is confirmed when the corrected interval lies on the predicted side
of its threshold):
  L1  calibration by the slot prior hurts at N >= 8          pride1 - raw1        < 0
  L2  voting over all cyclic shifts collapses at N >= 16       cycsum - vote3       < 0
  L3  the listwise posterior repairs cyclic voting (all N)     ours-cyc - cycsum    > 0
  D1  duel posterior (Eq. 2) beats the listwise posterior      duel_re - list_ours3 > 0
  D2  duel posterior (Eq. 2) beats majority voting             duel_re - majority   > 0
  D3  label-free EM beats majority voting                      em_shared - majority > 0
  D4  label-free EM is not worse than supervised by 3 pp       em_shared - duel_re  > -3
  C1  count-free duels within the 3-order token budget beat    budget - vote3c      > 0
      the 3-order vote counted per candidate (N = 8)
Methods as implemented in bayes_select.py (raw1, pride1, vote3, cycsum, ours-cyc), e6_em.py (list_ours3, duel_re =
Eq. 2 with the answer-level weight delta / (s2e + K tau2), supervised and 5-fold cross-fitted; em_shared = EM without
labels, duel parameters shared across groups, gamma per group) and e6_allcand.py (budget = as many random
order-balanced pairs as the token budget of the 3-order vote allows, least-squares scores; vote3c).

  python confirm_rules.py --judge llama-3.1-8b --model qwen3-8b --data results \
      --calls out/e1_direct_llama-3.1-8b_on_qwen3-8b.jsonl --duels out/e6_duels/e6_llama-3.1-8b_on_qwen3-8b.jsonl \
      --allcand out/e6_duels/e6_allcand_N8_llama-3.1-8b_on_qwen3-8b.jsonl \
      --output out/e6_duels/confirm_llama-3.1-8b.md
"""
import argparse
from pathlib import Path

import numpy as np

import bayes_select as B
import e6_allcand
import e6_em

HERE = Path(__file__).resolve().parent
K_TESTS = 8
DRAWS = 20000
TESTS = [  # id, claim, source, a, b, keep(set), direction, threshold
    ("L1", "calibration by the slot prior hurts (N >= 8)", "list", "pride1", "raw1", lambda v: v["N"] >= 8, "<", 0.0),
    ("L2", "cyclic-shift voting collapses (N >= 16)", "list", "cycsum", "vote3", lambda v: v["N"] >= 16, "<", 0.0),
    ("L3", "listwise posterior repairs cyclic voting", "list", "ours-cyc", "cycsum", lambda v: True, ">", 0.0),
    ("D1", "duel posterior beats the listwise posterior", "duel", "duel_re", "list_ours3", lambda v: True, ">", 0.0),
    ("D2", "duel posterior beats majority voting", "duel", "duel_re", "majority", lambda v: True, ">", 0.0),
    ("D3", "label-free EM beats majority voting", "duel", "em_shared", "majority", lambda v: True, ">", 0.0),
    ("D4", "label-free EM within 3 pp of supervised", "duel", "em_shared", "duel_re", lambda v: True, ">", -3.0),
    ("C1", "count-free duels beat the per-candidate vote", "allcand", "budget", "vote3c", lambda v: True, ">", 0.0),
]


def boot(per_set, a, b, keep, k=K_TESTS, draws=DRAWS, seed=0):
    ids = [i for i, v in per_set.items() if keep(v) and a in v and b in v]
    pids = sorted({per_set[i]["pid"] for i in ids})
    ix = {p: j for j, p in enumerate(pids)}
    g = np.array([ix[per_set[i]["pid"]] for i in ids])
    v = 100 * np.array([per_set[i][a] - per_set[i][b] for i in ids], float)
    tot, cnt = np.bincount(g, v, len(pids)), np.bincount(g, None, len(pids)).astype(float)
    rng = np.random.default_rng(seed)
    bs = np.concatenate([(lambda d: tot[d].sum(1) / cnt[d].sum(1))(rng.integers(0, len(pids), (1000, len(pids))))
                         for _ in range(draws // 1000)])
    tail = 100 * 0.025 / k
    lo, hi = np.percentile(bs, [tail, 100 - tail])
    lo95, hi95 = np.percentile(bs, [2.5, 97.5])
    return dict(est=float(v.mean()), lo=float(lo), hi=float(hi), lo95=float(lo95), hi95=float(hi95),
                sets=len(ids), problems=len(pids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--model", default="qwen3-8b", help="whose candidates and natural sets")
    ap.add_argument("--data", type=Path, default=HERE / "results", help="folder with e1_sets_<model>.jsonl")
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--duels", type=Path, required=True)
    ap.add_argument("--allcand", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    sets = {s["set_id"]: s for s in B.load(args.data / f"e1_sets_{args.model}.jsonl")}
    calls = B.load_calls(args.calls)
    lst, _, _ = B.listwise_per_set(sets, calls)
    for v in lst.values():
        v["N"] = v["group"][1]
    duel, _, fits = e6_em.evaluate(args.judge, args.model, args.duels, args.calls, args.data, 0)
    allc, cost, diag, n_allc = e6_allcand.evaluate(args.model, args.data, args.allcand, args.calls)
    src = dict(list=lst, duel=duel, allcand=allc)

    mass = [r["p_mass"] for rs in calls.values() for r in rs]
    lines = [f"# 预注册确认：{args.judge} 评 {args.model} 的候选", "",
             "由 `confirm_rules.py` 生成，计划见 `out/e6_duels/prereg_llama.md`。差值单位为百分点；“校正区间”为对 8 项检验做 "
             f"Bonferroni 校正后的 {100 - 5 / K_TESTS:.3f}% 区间（按原题整群重采样 {DRAWS} 次），落在预测一侧的阈值之外即为确认。", "",
             "## 数据", "",
             f"- 列表调用：{sum(len(v) for v in calls.values())} 次，覆盖 {len(calls)} 个集合；编号概率质量均值 {np.mean(mass):.3f}，"
             f"低于 0.5 的 {sum(m < 0.5 for m in mass)} 次。",
             f"- 列表方法可评估的集合 {len(lst)}，答案对决 {len(duel)}，全候选对决 {n_allc}（N = 8）。",
             f"- 等 token 预算下的双序对数：中位数 {np.median(diag['m_budget']):.0f}；预算规则的 token 为 3 顺序投票的 "
             f"{np.mean(cost['budget']) / np.mean(cost['vote3']):.2f} 倍。", "",
             "## 登记的检验", "",
             "| 编号 | 主张 | 比较 | 差值 | 95% 区间 | 校正区间 | 阈值 | 集合 / 原题 | 结论 |",
             "|---|---|---|---:|---|---|---|---|---|"]
    confirmed = 0
    for tid, claim, s, a, b, keep, direction, thr in TESTS:
        r = boot(src[s], a, b, keep)
        ok = r["hi"] < thr if direction == "<" else r["lo"] > thr
        confirmed += ok
        lines.append(f"| {tid} | {claim} | {a} − {b} | {r['est']:+.1f} | [{r['lo95']:+.1f}, {r['hi95']:+.1f}] | "
                     f"[{r['lo']:+.1f}, {r['hi']:+.1f}] | {direction} {thr:+.0f} | {r['sets']} / {r['problems']} | "
                     f"{'确认' if ok else '未确认'} |")
    lines += ["", f"确认 {confirmed} / {len(TESTS)} 项。", "", "## 描述性准确率（%）", "",
              "| 来源 | 方法 | 准确率 | 集合 |", "|---|---|---:|---:|"]
    for s, methods in (("list", ("majority", "raw1", "pride1", "vote3", "cycsum", "ours-cyc", "ours3")),
                       ("duel", ("majority", "list_ours3", "duel_prereg", "duel_re", "em_group", "em_shared")),
                       ("allcand", ("random", "majority", "raw1", "vote3", "vote3c", "budget", "full56", "knockout14"))):
        for m in methods:
            vals = [v[m] for v in src[s].values() if m in v]
            if vals:
                lines.append(f"| {s} | {m} | {100 * np.mean(vals):.1f} | {len(vals)} |")
    gam, th, ll = fits["em_shared"]
    lines += ["", f"EM（共享对决参数）：δ = {th['*'][0]:.2f}，τ² = {th['*'][1]:.2f}，σ²ε = {th['*'][2]:.2f}；γ 按组 = "
              + "，".join(f"{g}-{n} {v:.2f}" for (g, n), v in sorted(gam.items())) + "。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
