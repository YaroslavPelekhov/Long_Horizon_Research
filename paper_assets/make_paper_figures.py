from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(ax, xy, wh, text, fc="#f7f8fa", ec="#333333", lw=0.85, fs=8.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def arrow(ax, start, end, color="#222222", lw=0.9, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def method_overview():
    fig, ax = plt.subplots(figsize=(7.05, 3.18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.05)
    ax.axis("off")

    ax.text(0.05, 3.83, "One shared hypothesis state", fontsize=9.8, fontweight="bold")
    ax.text(
        0.05,
        3.59,
        "Adapters expose observations; all reasoning is routed through typed artifacts, executable evidence, and residuals.",
        fontsize=7.7,
        color="#333333",
    )

    colors = {
        "input": "#eaf2fb",
        "contract": "#eef7ee",
        "ctx": "#fff4dc",
        "exec": "#f3eefb",
        "resid": "#fdeeed",
        "out": "#eef5f7",
    }

    box(ax, (0.25, 1.55), (1.15, 0.62), "Task\ninterface", colors["input"])
    box(ax, (1.8, 1.55), (1.38, 0.62), "Evidence\ncontract", colors["contract"])
    box(ax, (3.65, 1.34), (1.72, 1.04), "HypothesisContext\nC, H, theta, E,\nR, O, h*", colors["ctx"], fs=7.8)
    box(ax, (5.95, 1.55), (1.42, 0.62), "Rendered\nartifact", colors["out"])
    box(ax, (8.05, 1.55), (1.32, 0.62), "Native\nscore", "#f7f7f7")

    box(ax, (1.8, 0.34), (1.38, 0.58), "LLM proposes\ntyped sketch", colors["exec"], fs=7.5)
    box(ax, (3.65, 0.34), (1.72, 0.58), "Execution closes\nholes", colors["exec"], fs=7.5)
    box(ax, (5.95, 0.34), (1.42, 0.58), "Validation\nand replay", colors["exec"], fs=7.5)

    box(ax, (3.65, 2.78), (1.72, 0.58), "Typed\nresiduals", colors["resid"], fs=7.6)
    box(ax, (5.95, 2.78), (1.42, 0.58), "Operator\npromotion", colors["resid"], fs=7.6)
    box(ax, (7.88, 2.78), (1.48, 0.58), "Shared operator\nlibrary", "#f5f7fb", fs=7.4)

    arrow(ax, (1.4, 1.86), (1.8, 1.86))
    arrow(ax, (3.18, 1.86), (3.65, 1.86))
    arrow(ax, (5.37, 1.86), (5.95, 1.86))
    arrow(ax, (7.37, 1.86), (8.05, 1.86))
    arrow(ax, (2.49, 1.55), (2.49, 0.92))
    arrow(ax, (3.18, 0.63), (3.65, 0.63))
    arrow(ax, (4.51, 0.92), (4.51, 1.34))
    arrow(ax, (5.37, 0.63), (5.95, 0.63))
    arrow(ax, (6.66, 0.92), (6.66, 1.55))
    arrow(ax, (4.51, 2.38), (4.51, 2.78))
    arrow(ax, (5.37, 3.07), (5.95, 3.07))
    arrow(ax, (7.37, 3.07), (7.88, 3.07))
    arrow(ax, (8.62, 2.78), (2.49, 2.17), rad=0.16)
    arrow(ax, (8.71, 2.17), (6.66, 2.17), color="#666666", lw=0.75, style="->", rad=-0.15)
    ax.text(7.18, 2.30, "held-out gain - beta K", fontsize=6.9, color="#444444")

    ax.text(0.25, 1.14, "tables / simulators / traces", fontsize=6.8, color="#444444")

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / "rghli_method_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rghli_method_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main_results():
    labels = ["Discovery\nHMS", "Newton\nSA-all", "UltraHorizon\nscore"]
    rghli = np.array([28.09, 49.7, 51.04])
    baseline = np.array([15.4, 47.8, 14.33])
    gpt4o = np.array([30.03, 36.1, np.nan])

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    ax.bar(x - width, baseline, width, label="Published ref.", color="#c8d6e5", edgecolor="#333333", linewidth=0.35)
    ax.bar(x, rghli, width, label="RG-HLI 4o-mini", color="#8fc1a9", edgecolor="#333333", linewidth=0.35)
    ax.bar(x + width, np.nan_to_num(gpt4o, nan=0), width, label="RG-HLI 4o", color="#f4c95d", edgecolor="#333333", linewidth=0.35)
    for i, val in enumerate(gpt4o):
        if np.isnan(val):
            ax.text(x[i] + width, 2.5, "n/a", ha="center", va="bottom", fontsize=6.8, color="#555555")
    for xpos, vals in [(x - width, baseline), (x, rghli), (x + width, gpt4o)]:
        for xi, yi in zip(xpos, vals):
            if np.isnan(yi):
                continue
            ax.text(xi, yi + 1.5, f"{yi:.1f}", ha="center", va="bottom", fontsize=6.7)
    ax.set_ylim(0, 66)
    ax.set_ylabel("Native benchmark score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "main_results_chart.pdf", bbox_inches="tight")
    fig.savefig(OUT / "main_results_chart.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def discovery_ablation():
    labels = ["fixed-language\nnone", "RG-HLI\nfull"]
    values = np.array([0.68, 28.09])
    colors = ["#e8e8e8", "#8fc1a9"]
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor="#333333", linewidth=0.35)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("DiscoveryBench HMS, full239")
    ax.set_ylim(0, 34)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(
        "+27.41 HMS",
        xy=(1, 28.09),
        xytext=(0.55, 31.5),
        ha="center",
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "discovery_ablation_chart.pdf", bbox_inches="tight")
    fig.savefig(OUT / "discovery_ablation_chart.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def discovery_external_controls():
    labels = ["Direct", "ReAct", "CodeAct", "RG-HLI"]
    values = np.array([9.54, 11.68, 12.31, 28.09])
    colors = ["#e8edf3", "#e8edf3", "#e8edf3", "#8fc1a9"]
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor="#333333", linewidth=0.35)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("DiscoveryBench HMS, full239")
    ax.set_ylim(0, 33)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(
        "typed residual state",
        xy=(3, 28.09),
        xytext=(2.25, 31.0),
        ha="center",
        fontsize=6.7,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "discovery_external_controls_chart.pdf", bbox_inches="tight")
    fig.savefig(OUT / "discovery_external_controls_chart.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def uh_breakdown():
    labels = ["Grid", "Seq", "Bio"]
    vals = np.array([54.38, 81.25, 90.47])
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    bars = ax.barh(labels, vals, color=["#c8d6e5", "#8fc1a9", "#f4c95d"], edgecolor="#333333", linewidth=0.35)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Paper-style score")
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(v + 1.4, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=7)
    ax.set_title("UltraHorizon full96 breakdown", pad=3)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "uh_breakdown_chart.pdf", bbox_inches="tight")
    fig.savefig(OUT / "uh_breakdown_chart.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def residual_language_induction_ravr_style():
    fig, ax = plt.subplots(figsize=(6.9, 3.10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.75)
    ax.axis("off")

    def rbox(x, y, w, h, text, fc, ec, fs=7.3, lw=0.75):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.055",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
        return patch

    def small_label(x, y, text, color="#555555", fs=6.2, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color)

    def arr(x1, y1, x2, y2, color="#222222", lw=0.7, rad=0.0, style="-|>"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=style,
                mutation_scale=7.5,
                linewidth=lw,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    # RAVR-like framed loops: one per-task lane and one language-growth lane.
    base = FancyBboxPatch(
        (0.42, 0.86),
        3.80,
        4.55,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=0.65,
        edgecolor="#b7b8ff",
        facecolor="#f7f6ff",
        linestyle=(0, (4, 3)),
    )
    ax.add_patch(base)
    ax.text(0.62, 5.50, "per-task executable search", fontsize=6.3, color="#5f5fa8", va="center")

    rbox(0.96, 4.88, 2.70, 0.38, r"$L_t$: current hypothesis language", "#eef4ff", "#2f5597", fs=7.0)
    rbox(0.86, 4.24, 2.92, 0.38, r"1. Contract search: $C(x)\rightarrow h\in L_t$", "#eef8ef", "#3c7f3c", fs=6.75)
    rbox(0.86, 3.58, 2.92, 0.38, "2. Execute probes and validators", "#fff4df", "#c88900", fs=6.75)

    # Decision diamond.
    diamond = plt.Polygon(
        [[2.32, 3.21], [2.78, 2.97], [2.32, 2.73], [1.86, 2.97]],
        closed=True,
        facecolor="#fffafa",
        edgecolor="#d14a4a",
        linewidth=0.7,
    )
    ax.add_patch(diamond)
    ax.text(2.32, 2.97, "valid?", ha="center", va="center", fontsize=6.4, color="#9d2323")

    rbox(0.92, 2.08, 2.82, 0.48, "3. Typed residualization\n" + r"$r=(\tau,\ell,c,e,v,\sigma)$", "#fff1f1", "#c62f2f", fs=6.55)
    rbox(0.92, 1.28, 2.82, 0.48, r"4. Residual class $R_k$", "#f5efff", "#6d3fa0", fs=6.85)
    rbox(3.04, 2.70, 1.00, 0.46, "validated\nartifact", "#eef7f7", "#357c7c", fs=5.8)

    # Growth lane on the right.
    growth = FancyBboxPatch(
        (5.46, 0.86),
        4.10,
        4.55,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=0.65,
        edgecolor="#f0b7a5",
        facecolor="#fff8f4",
        linestyle=(0, (4, 3)),
    )
    ax.add_patch(growth)
    ax.text(5.66, 5.50, "language growth from failures", fontsize=6.3, color="#a45536", va="center")

    rbox(5.96, 4.38, 2.92, 0.52, "5. Pattern discovery\nresiduals = deficits of " + r"$L_t$", "#f5efff", "#6d3fa0", fs=6.45)
    rbox(5.96, 3.48, 2.92, 0.52, r"6. Induce operator $o^\star$" + "\ncloses class " + r"$R_k$", "#f5efff", "#6d3fa0", fs=6.45)
    rbox(5.96, 2.42, 2.92, 0.64, "7. Promotion gate\n" + r"$\Delta_{\mathrm{heldout}}(o^\star)>\lambda C(o^\star)$", "#fff0e8", "#df5b2f", fs=6.35)
    rbox(5.96, 1.28, 2.92, 0.54, r"8. $L_{t+1}=L_t\cup\{o^\star\}$", "#eef4ff", "#2f5597", fs=6.65)

    # Arrows in base lane.
    arr(2.32, 4.90, 2.32, 4.62)
    arr(2.32, 4.24, 2.32, 3.96)
    arr(2.32, 3.58, 2.32, 3.21)
    arr(1.90, 2.88, 1.46, 2.56)
    small_label(1.42, 2.88, "failure", color="#9d2323", fs=5.9, ha="right")
    arr(2.78, 2.97, 3.04, 2.93)
    small_label(3.10, 3.22, "success", fs=5.7, ha="left")
    arr(2.32, 2.08, 2.32, 1.76)

    # Failure-to-language-growth path.
    arr(3.74, 1.52, 5.96, 4.64, color="#6d3fa0", lw=0.85, rad=-0.18)
    ax.text(4.93, 4.27, "typed failure\nclass", fontsize=5.8, color="#6d3fa0", ha="center", va="center")
    arr(7.42, 4.38, 7.42, 4.00, color="#6d3fa0", lw=0.75)
    arr(7.42, 3.48, 7.42, 3.06, color="#6d3fa0", lw=0.75)
    arr(7.42, 2.42, 7.42, 1.82, color="#2f5597", lw=0.75)

    ax.text(
        0.46,
        0.38,
        "Novelty: operators are induced from typed validation residuals, not from successful-program compression or generic text feedback.",
        fontsize=6.4,
        color="#222222",
        ha="left",
        va="center",
    )

    fig.tight_layout(pad=0.12)
    fig.savefig(OUT / "residual_language_induction_ravr_style.pdf", bbox_inches="tight")
    fig.savefig(OUT / "residual_language_induction_ravr_style.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    method_overview()
    main_results()
    discovery_ablation()
    discovery_external_controls()
    uh_breakdown()
    residual_language_induction_ravr_style()
