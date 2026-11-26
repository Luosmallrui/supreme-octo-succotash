"""E1 figure: slot prior vs N and selection accuracy by method vs N (palette from the dataviz reference instance)."""
import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results")
ap.add_argument("--model", default="qwen3-8b")
ap.add_argument("--group", default="math500")
args = ap.parse_args()
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
d = pickle.load(open(os.path.join(args.dir, f"e1_summary_{args.model}.pkl"), "rb"))
present = {k[0] for k in d if isinstance(k, tuple)}
modes = [m for m in ["direct", "think"] if m in present] + sorted(m for m in present if m.startswith("api-"))
fig, axes = plt.subplots(1, 1 + len(modes), figsize=(4.0 * (1 + len(modes)), 3.3), squeeze=False)
axes = axes[0]
ax = axes[0]
for j, m in enumerate(modes):
    Ns = sorted(k[2] for k in d if k[0] == m and k[1] == args.group and k[3] == "fa")
    fa = [d[(m, args.group, N, "fa")] for N in Ns]
    ax.plot(Ns, [100 * f[0] for f in fa], color=S[j % len(S)], lw=2, marker="o", ms=4, mec="#fcfcfb", label=f"{m}: slot 1")
    ax.plot(Ns, [100 * np.mean(f[1:-1]) for f in fa], color=S[j % len(S)], lw=1.5, ls=":", label=f"{m}: middle slots")
ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels(["4", "8", "16", "32"])
ax.set_xlabel("number of candidates N")
ax.set_ylabel("% picked | slot holds a wrong one")
ax.set_title("slot prior", fontsize=9, color=INK)
ax.legend(fontsize=7.5)
methods = [("raw", "raw, 1 call", S[0], "-"), ("calibrated", "calibrated, 1 call", S[1], "-"),
           ("maj-first", "majority-first order, 1 call", S[2], "-"), ("vote K", "vote over 3 orders, 3 calls", S[3], "-"),
           ("cyclic", "cyclic permutation, N calls", S[4], "-"), ("majority vote", "majority vote (no LLM)", INK2, "--")]
for j, m in enumerate(modes):
    ax = axes[1 + j]
    Ns = sorted(k[2] for k in d if k[0] == m and k[1] == args.group and k[3] == "acc")
    for key, lab, col, ls in methods:
        ys = [d[(m, args.group, N, "acc")].get(key, np.nan) for N in Ns]
        if np.all(np.isnan(ys)):
            continue
        ax.plot(Ns, [100 * y for y in ys], color=col, lw=2 if ls == "-" else 1.5, ls=ls, marker="o" if ls == "-" else None,
                ms=4, mec="#fcfcfb", label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels(["4", "8", "16", "32"])
    ax.set_xlabel("number of candidates N")
    ax.set_title(f"{m} mode: selection accuracy ({args.group}, mixed sets)", fontsize=9, color=INK)
    if j == 0:
        ax.set_ylabel("% sets, chosen candidate correct")
seen = {}
for a in axes[1:]:
    for h_, l_ in zip(*a.get_legend_handles_labels()):
        seen.setdefault(l_, h_)
fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=3, fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(os.path.join(args.dir, f"fig_e1_{args.model}.png"), dpi=160)
print("figure written")
