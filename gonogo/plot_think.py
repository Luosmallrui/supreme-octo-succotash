"""Figure for the thinking-mode sweep (palette from the dataviz reference instance)."""
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
args = ap.parse_args()

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

d = pickle.load(open(os.path.join(args.dir, f"think_summary_{args.model}.pkl"), "rb"))
beh, att = d["behaviour"], d["attn"]
Ns = [N for N in [4, 8, 16] if N in beh]
rows = 2 + (2 if att else 0)
fig, axes = plt.subplots(rows, len(Ns), figsize=(3.7 * len(Ns), 2.6 * rows), squeeze=False)


def line(ax, x, y, col, lab, **kw):
    ax.plot(x, y, color=col, lw=2, label=lab, **kw)


for j, N in enumerate(Ns):
    b = beh[N]
    x = np.arange(1, N + 1)
    ax = axes[0, j]
    line(ax, x, 100 * np.array(b["all"]["curve"]), S1, "accuracy, all runs", marker="o", ms=3.5, mec="#fcfcfb")
    line(ax, x, 100 * np.array(b["fin"]["curve"]), S2, "accuracy, finished runs", marker="o", ms=3.5, mec="#fcfcfb")
    line(ax, x, 100 * np.array(b["trunc"]["curve"]), S3, "truncated (no answer)", ls="--")
    ax.axhline(100 / N, color=INK2, lw=1, ls=":")
    ax.set_title(f"N={N}  ({b['n_prob']} problems)", fontsize=9, color=INK)
    ax.set_xlabel("correct-candidate position")
    ax.set_ylim(0, 100)
    ax = axes[1, j]
    ax.bar(x, 100 * np.array(b["lift"]), color=S1, width=0.7)
    ax.set_xlabel("slot")
    ax.set_title(f"content lift by slot  (edge−mid {100*b['edge_mid']:+.1f})", fontsize=9, color=INK)
    if att and N in att and "early" in att[N]:
        a = att[N]
        slot = np.array(a["early"]["slot"]) * N
        NL = slot.shape[0]
        ax = axes[2, j]
        for lab, sl, col in [("layers 1-2", slice(0, 2), S1), (f"layers {NL//3+1}-{2*NL//3}", slice(NL // 3, 2 * NL // 3), S2),
                             (f"layers {2*NL//3+1}-{NL}", slice(2 * NL // 3, NL), S3)]:
            line(ax, x, slot[sl].mean(0), col, lab)
        ax.axhline(1, color=INK2, lw=1, ls=":")
        ax.set_xlabel("candidate slot")
        ax.set_title("attention from early reasoning (×N)", fontsize=9, color=INK)
        ax = axes[3, j]
        L = np.arange(1, NL + 1)
        for which, col in [("early", S1), ("dec", S2)]:
            c = a.get(f"choice_{which}")
            if c:
                line(ax, L, np.clip(c["expl"], -0.5, 1.5), col, {"early": "early reasoning", "dec": "decision token"}[which])
        if not any(np.isfinite(a.get(f"choice_{w}", {}).get("expl", [np.nan])).any() for w in ("early", "dec")):
            ax.text(0.5, 0.5, "no cross-validated position effect\nto explain at this N", ha="center", va="center",
                    transform=ax.transAxes, color=INK2, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        else:
            ax.axhline(0, color=INK2, lw=1)
            ax.axhline(1, color=INK2, lw=1, ls=":")
        ax.set_xlabel("layer")
        ax.set_title("share of position effect on choice\nrecovered by attention (clipped)", fontsize=9, color=INK)
for ax in axes[:3].ravel():
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
axes[0, 0].set_ylabel("% of prompts")
axes[1, 0].set_ylabel("P(hit) − P(false alarm), %")
for r_ in range(rows):
    h, l = axes[r_, 0].get_legend_handles_labels()
    if l:
        axes[r_, 0].legend(fontsize=7.5, loc="best")
fig.suptitle(f"{args.model}: thinking-mode selection", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(args.dir, f"fig4_think_{args.model}.png"), dpi=160)
print("figure written")
