"""Figures for the go/no-go report (static PNG, palette from the dataviz reference instance)."""
import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results")
ap.add_argument("--models", default="qwen3-8b,qwen3-4b")
args = ap.parse_args()

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4
SEQ = ["#b9d3f2", "#7aaee8", "#3987e5", "#1a5aa8"]  # one hue, light -> dark (N = 4, 8, 16, 32)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

d = pickle.load(open(os.path.join(args.dir, "summary.pkl"), "rb"))
summ, attn = d["summary"], d["attn"]
models = [m for m in args.models.split(",") if any(k[0] == m for k in summ)]
Ns = [4, 8, 16, 32]


def binned(curve, N):
    c = np.asarray(curve, float)
    b = 1 if N <= 8 else N // 8
    x = np.arange(N).reshape(-1, b).mean(1) + 1
    return x, c.reshape(-1, b).mean(1)


# Fig 1: accuracy vs position of the correct candidate
fig, axes = plt.subplots(len(models), 4, figsize=(11, 2.6 * len(models)), squeeze=False)
for i, m in enumerate(models):
    for j, N in enumerate(Ns):
        ax = axes[i, j]
        for mode, col, lab in [("A", S1, "A: select index"), ("B", S2, "B: aggregate, headers"), ("C", S3, "C: aggregate, no headers"), ("T", S4, "T: select index, thinking")]:
            s = summ.get((m, mode, N))
            if not s:
                continue
            x, y = binned(s["curve"], N)
            ax.plot(x, 100 * y, color=col, lw=2, marker="o", ms=4, mec="#fcfcfb", mew=1, label=lab)
        if (m, "A", N) in summ:
            ax.axhline(100 / N, color=INK2, lw=1, ls=":")
            ax.set_title(f"{m}  N={N}  (n={summ[(m, 'A', N)]['n']})", fontsize=9, color=INK)
        ax.set_xlabel("correct-candidate position" + (f" (bin={N // 8})" if N > 8 else ""))
        if j == 0:
            ax.set_ylabel("accuracy (%)")
        ax.set_xlim(0.5, N + 0.5)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8)
fig.tight_layout(rect=(0, 0.07 / len(models) + 0.02, 1, 1))
fig.savefig(os.path.join(args.dir, "fig1_position_curves.png"), dpi=160)

# Fig 2: block attention share per slot (x N, 1 = uniform), shallow / middle / deep layers
for m in models:
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), squeeze=False)
    for i, mode in enumerate(["A", "C"]):
        for j, N in enumerate(Ns):
            ax = axes[i, j]
            a = attn.get((m, mode, N))
            if not a:
                ax.set_visible(False)
                continue
            slot = np.array(a["slot"]) * N
            NL = slot.shape[0]
            groups = [("layers 1-2", slice(0, 2), S1), (f"layers {NL//3+1}-{2*NL//3}", slice(NL // 3, 2 * NL // 3), S2),
                      (f"layers {2*NL//3+1}-{NL}", slice(2 * NL // 3, NL), S3)]
            for lab, sl, col in groups:
                ax.plot(np.arange(1, N + 1), slot[sl].mean(0), color=col, lw=2, label=lab)
            ax.axhline(1, color=INK2, lw=1, ls=":")
            ax.set_title(f"mode {mode}  N={N}", fontsize=9, color=INK)
            ax.set_xlabel("candidate slot")
            if j == 0:
                ax.set_ylabel("attention share × N")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle(f"{m}: attention from instruction rows to each candidate block (head mean)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, f"fig2_attention_slots_{m}.png"), dpi=160)

# Fig 3: layer-wise link between attention and behaviour
for m in models:
    ch = attn.get((m, "choice"), {})
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    for k, N in enumerate(Ns):
        col = SEQ[k]
        a = attn.get((m, "A", N))
        if a:
            axes[0].plot(np.arange(1, len(a["rho_full"]) + 1), a["rho_full"], color=col, lw=2, label=f"N={N}")
        if N in ch:
            axes[1].plot(np.arange(1, len(ch[N]["expl"]) + 1), np.clip(ch[N]["expl"], -0.5, 1.5), color=col, lw=2, label=f"N={N}")
            axes[2].plot(np.arange(1, len(ch[N]["dll"]) + 1), ch[N]["dll"], color=col, lw=2, label=f"N={N}")
    axes[0].set_title("r(attention on correct block, accuracy)\nacross positions", fontsize=9)
    axes[1].set_title("share of position effect on choice\nexplained by attention (cor+attn vs pos+cor)", fontsize=9)
    axes[2].set_title("extra log-lik / prompt from attention\nbeyond position + correctness", fontsize=9)
    for ax in axes:
        ax.set_xlabel("layer")
    axes[0].axhline(0, color=INK2, lw=1)
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].axhline(1, color=INK2, lw=1, ls=":")
    axes[0].legend(fontsize=7.5)
    fig.suptitle(m, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, f"fig3_layerwise_{m}.png"), dpi=160)
print("figures written")
