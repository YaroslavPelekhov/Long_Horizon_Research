from __future__ import annotations

import csv
import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
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


def concrete_operator_growth_example():
    """Render a concrete, auditable NewtonBench language-growth trace."""
    fig, ax = plt.subplots(figsize=(7.05, 2.32))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.06)
    ax.axis("off")

    ax.text(0.06, 2.84, "A concrete language-growth trace", fontsize=10.0, fontweight="bold")
    ax.text(
        0.06,
        2.62,
        "One audited NewtonBench extension: source residuals propose an operator; admission and transfer are held out.",
        fontsize=7.2,
        color="#444444",
    )

    box(
        ax,
        (0.15, 0.94),
        (1.85, 1.15),
        "Development sources\n\nGravity: F(m1, m2, r)\nCoulomb: F(q1, q2, r)",
        fc="#eef4ff",
        ec="#2f5597",
        fs=6.75,
    )
    box(
        ax,
        (2.32, 0.94),
        (1.85, 1.15),
        "Typed residual R\n\nPositive multiplicative\nrelation is unclosed\nin the current language",
        fc="#fff1f1",
        ec="#c62f2f",
        fs=6.25,
    )
    arrow(ax, (2.00, 1.51), (2.32, 1.51), color="#a22d2d", lw=0.95)

    box(
        ax,
        (4.58, 0.78),
        (2.05, 1.46),
        "Induced executable operator $O_{log}$\n\n$\\log y = a + \\sum_i p_i \\log x_i$\n\nfit exponents by regression,\nnot by guessing a complete law",
        fc="#eef8ef",
        ec="#3c7f3c",
        fs=6.2,
    )
    arrow(ax, (4.17, 1.51), (4.58, 1.51), color="#6d3fa0", lw=0.95)

    box(
        ax,
        (7.32, 0.78),
        (2.40, 1.46),
        "Disjoint transfer: Hooke\n\n$L_0$ loss = 1.000\n$O_{log}$ exposes the $F=kx$ family\n$L_1$ loss = 0.206",
        fc="#fff7e8",
        ec="#c88900",
        fs=6.25,
    )
    arrow(ax, (6.63, 1.51), (7.32, 1.51), color="#3c7f3c", lw=1.05)

    ax.text(6.98, 2.36, "promotion held out: net utility = +0.599", ha="center", fontsize=5.75, color="#3c7f3c")
    ax.text(
        5.0,
        0.28,
        "The final law is not copied from the source tasks: the admitted operator changes the hypothesis family available to a new task.",
        ha="center",
        fontsize=6.75,
        color="#333333",
    )
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "concrete_operator_growth_example.pdf", bbox_inches="tight")
    fig.savefig(OUT / "concrete_operator_growth_example.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def language_growth_audit():
    """Visualize task-level held-out evidence for the induced language."""
    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.16), gridspec_kw={"width_ratios": [1, 1, 0.86]})
    ax_promote, ax_transfer, ax_gate = axes
    palette = {"$O_1$": "#2f7d50", "$O_2$": "#2e6db4", "duplicate": "#c65a37"}

    def paired_scatter(ax, before, after, groups, title):
        for x, y, group in zip(before, after, groups):
            ax.plot([x, y], [x, y], alpha=0)  # Keep autoscaling symmetric.
            ax.scatter(x, y, s=32, color=palette[group], edgecolor="white", linewidth=0.45, zorder=3)
        ax.plot([0, 1.02], [0, 1.02], color="#737373", lw=0.7, linestyle="--", zorder=1)
        ax.fill_between([0, 1.02], [0, 1.02], [0, 0], color="#eef7ef", alpha=0.8, zorder=0)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontweight="bold", pad=5)
        ax.set_xlabel("loss before candidate")
        ax.grid(color="#dddddd", linewidth=0.35, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # Promotion modules: O1 is tested on two modules, O2 on one; the last two
    # points are the re-proposed O1 family and receive no new utility.
    paired_scatter(
        ax_promote,
        np.array([1.0, 1.0, 0.4326, 0.5090, 0.2232]),
        np.array([0.5345, 0.2546, 0.4184, 0.5090, 0.2232]),
        ["$O_1$", "$O_1$", "$O_2$", "duplicate", "duplicate"],
        "Promotion tasks",
    )
    ax_promote.set_ylabel("loss after candidate")
    ax_promote.text(0.05, 0.90, "below diagonal = gain", fontsize=6.2, color="#3d6e44")

    # Transfer modules are disjoint from the source and promotion partitions.
    paired_scatter(
        ax_transfer,
        np.array([1.0, 1.0, 1.0, 0.3535]),
        np.array([0.3732, 0.2060, 0.6594, 0.2539]),
        ["$O_1$", "$O_1$", "$O_1$", "$O_2$"],
        "Disjoint transfer tasks",
    )
    ax_transfer.text(0.05, 0.90, "all four improve", fontsize=6.2, color="#3d6e44")

    labels = ["$O_1$\nmonomial", "$O_2$\nadditive", "duplicate\n$O_1$"]
    values = np.array([0.5988, 0.0082, -0.0067])
    ypos = np.arange(3)
    ax_gate.axvline(0, color="#737373", lw=0.7, zorder=0)
    ax_gate.hlines(ypos, 0, values, color=[palette["$O_1$"], palette["$O_2$"], palette["duplicate"]], linewidth=2.15, zorder=2)
    ax_gate.scatter(values, ypos, color=[palette["$O_1$"], palette["$O_2$"], palette["duplicate"]], s=34, zorder=3)
    for x, y in zip(values, ypos):
        ax_gate.text(x + (0.018 if x >= 0 else -0.018), y + 0.11, f"{x:+.3f}", ha="left" if x >= 0 else "right", fontsize=6.6)
    ax_gate.set_title("Promotion decision", fontweight="bold", pad=5)
    ax_gate.set_xlabel("net held-out utility")
    ax_gate.set_yticks(ypos)
    ax_gate.set_yticklabels(labels)
    ax_gate.set_xlim(-0.12, 0.70)
    ax_gate.set_ylim(-0.55, 2.55)
    ax_gate.grid(axis="x", color="#dddddd", linewidth=0.35, zorder=0)
    ax_gate.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.01,
        "NewtonBench diagnostic: source, promotion, and transfer modules use disjoint partitions; every point is one held-out task. Green shading indicates improvement over the frozen language.",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.38, w_pad=1.85)
    fig.savefig(OUT / "language_growth_audit.pdf", bbox_inches="tight")
    fig.savefig(OUT / "language_growth_audit.png", dpi=220, bbox_inches="tight")
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


def scientific_artifact_atlas():
    """Show concrete, logged artifacts rather than an abstract benchmark taxonomy."""
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.78), gridspec_kw={"wspace": 0.22})
    fig.text(
        0.5,
        0.985,
        "One typed state, three scientific artifact interfaces",
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "Examples are drawn from retained full-run outputs; each shows the task, executable evidence, and rendered artifact.",
        ha="center",
        va="top",
        fontsize=6.9,
        color="#444444",
    )

    panels = [
        {
            "title": "DiscoveryBench",
            "color": "#1e5e9e",
            "task": "Which century did axes become quantitatively most frequent?",
            "evidence": "Filtered the time series; selected the peak frequency window.",
            "artifact": "End of 4th millennium BCE: axes are quantitatively most frequent.",
            "score": "HMS = 100; C-HMS = 100",
            "tag": "tabular discovery",
        },
        {
            "title": "NewtonBench",
            "color": "#2f7d50",
            "task": "Infer the force law from mass1, mass2, and distance observations.",
            "evidence": "Log-coordinate probe closes exponents: (1, 1, -1.5).",
            "artifact": "F = 6.674e-5 * mass1 * mass2 * distance^-1.5",
            "score": "symbolic equivalence = true",
            "tag": "symbolic law",
        },
        {
            "title": "UltraHorizon",
            "color": "#a45b14",
            "task": "Commit a 50-step sequence rule from partial interaction traces.",
            "evidence": "Replay detects parity-controlled interleaving before reversal and shift.",
            "artifact": "Interleave main and vice; parity picks the leader. Reverse, then shift by step.",
            "score": "strict Seq task score = 100",
            "tag": "long-horizon rule",
        },
    ]

    for ax, panel in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        c = panel["color"]
        ax.add_patch(
            FancyBboxPatch(
                (0.03, 0.095),
                0.94,
                0.73,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=0.8,
                edgecolor=c,
                facecolor="#fbfbfb",
            )
        )
        ax.text(0.08, 0.89, panel["title"], color=c, fontsize=8.6, fontweight="bold", va="bottom")
        ax.text(0.08, 0.84, panel["tag"], color="#666666", fontsize=6.3, va="bottom")

        sections = [("Task", panel["task"]), ("Executed evidence", panel["evidence"]), ("Rendered artifact", panel["artifact"])]
        ys = [0.73, 0.50, 0.28]
        for (label, text), y in zip(sections, ys):
            ax.text(0.09, y, label.upper(), color=c, fontsize=5.9, fontweight="bold", va="top")
            ax.text(0.09, y - 0.057, fill(text, width=28), fontsize=6.35, va="top", color="#171717")
            if label != "Rendered artifact":
                ax.plot([0.09, 0.91], [y - 0.16, y - 0.16], color="#dddddd", lw=0.55)
        ax.text(0.09, 0.028, panel["score"], fontsize=5.9, color="#333333", fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.89), pad=0.18)
    fig.savefig(OUT / "scientific_artifact_atlas.pdf", bbox_inches="tight")
    fig.savefig(OUT / "scientific_artifact_atlas.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def discovery_case_trace():
    """Render one complete, logged DiscoveryBench chain for the first paper page."""
    fig, ax = plt.subplots(figsize=(3.35, 2.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, green, ink = "#235f9e", "#2f7d50", "#1b1b1b"
    ax.text(0.03, 0.975, "A logged discovery artifact", fontsize=9.25, fontweight="bold", va="top")
    ax.text(0.03, 0.925, "DiscoveryBench / archaeology task", fontsize=6.5, color="#555555", va="top")

    def card(y, h, title, body, color, body_size=7.0):
        ax.add_patch(
            FancyBboxPatch(
                (0.035, y),
                0.93,
                h,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=0.75,
                edgecolor=color,
                facecolor="#fbfcfe",
            )
        )
        ax.text(0.07, y + h - 0.052, title.upper(), fontsize=5.85, fontweight="bold", color=color, va="top")
        ax.text(0.07, y + h - 0.11, fill(body, width=46), fontsize=body_size, color=ink, va="top")

    card(0.675, 0.18, "Task", "In which century did the axes become quantitatively most frequent?", blue, 6.85)
    card(0.405, 0.205, "Executed evidence", "Filter the archaeology time series; locate the maximum of the axes-frequency series.", blue, 6.8)
    card(0.135, 0.205, "Rendered hypothesis", "At the end of the 4th millennium BCE, axes become quantitatively most frequent.", green, 6.8)
    ax.text(0.50, 0.062, "native evaluator: HMS = 100; consistency-HMS = 100", ha="center", fontsize=6.2, color="#333333", fontweight="bold")
    ax.text(0.50, 0.018, "The trace retains both the executable measurement and the final scientific claim.", ha="center", fontsize=5.8, color="#555555")
    fig.tight_layout(pad=0.10)
    fig.savefig(OUT / "discovery_case_trace.pdf", bbox_inches="tight")
    fig.savefig(OUT / "discovery_case_trace.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def discovery_score_profile():
    """Plot the retained per-task evaluator outputs for the full DiscoveryBench run."""
    path = (
        ROOT
        / "lmw/universal_discovery_real/"
        "aaai27_language_growth_protocol_v1_fixed_l0_test239/"
        "official_eval.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    raw = np.asarray([float(row["HMS_raw_100"]) for row in rows])
    consistent = np.asarray([float(row["HMS_consistency_100"]) for row in rows])
    rng = np.random.default_rng(27)
    jitter = rng.normal(0, 0.65, len(raw))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.92, 2.25), gridspec_kw={"width_ratios": [1.15, 0.85]})
    ax1.scatter(raw + jitter, consistent - jitter, s=13, color="#2d6da3", alpha=0.58, linewidth=0)
    ax1.plot([0, 100], [0, 100], "--", color="#666666", lw=0.75)
    ax1.set_xlim(-3, 103)
    ax1.set_ylim(-3, 103)
    ax1.set_xlabel("HMS")
    ax1.set_ylabel("consistency-HMS")
    ax1.set_title("Per-task evaluator agreement", fontweight="bold", pad=5)
    ax1.grid(color="#e0e0e0", linewidth=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(4, 93, f"n = {len(rows)} full-run rows", fontsize=6.6, color="#444444")

    labels = ["0", "(0, 50)", "[50, 100)", "100"]
    masks = [raw == 0, (raw > 0) & (raw < 50), (raw >= 50) & (raw < 100), raw == 100]
    counts = np.asarray([int(mask.sum()) for mask in masks])
    colors = ["#e6e6e6", "#c8d9ea", "#76a6c9", "#27618f"]
    bars = ax2.bar(labels, counts, color=colors, edgecolor="#444444", linewidth=0.35)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.025, str(count), ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel("number of tasks")
    ax2.set_title("HMS outcome distribution", fontweight="bold", pad=5)
    ax2.set_ylim(0, max(counts) * 1.18)
    ax2.grid(axis="y", color="#e0e0e0", linewidth=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "DiscoveryBench full run: recorded native HMS and consistency-HMS for all 239 evaluator rows.",
        ha="center",
        fontsize=6.55,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.35, w_pad=1.55)
    fig.savefig(OUT / "discovery_score_profile.pdf", bbox_inches="tight")
    fig.savefig(OUT / "discovery_score_profile.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def newton_coverage_heatmap():
    """Show the actual per-domain profile of the complete 324-task NewtonBench run."""
    path = ROOT / "lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/rows.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = [
        ("m0_gravity", "Gravity"),
        ("m1_coulomb_force", "Coulomb"),
        ("m2_magnetic_force", "Magnetic"),
        ("m3_fourier_law", "Fourier"),
        ("m4_snell_law", "Snell"),
        ("m5_radioactive_decay", "Decay"),
        ("m6_underdamped_harmonic", "Harmonic"),
        ("m7_malus_law", "Malus"),
        ("m8_sound_speed", "Sound"),
        ("m9_hooke_law", "Hooke"),
        ("m10_be_distribution", "Bose--Einstein"),
        ("m11_heat_transfer", "Heat"),
    ]
    difficulties = ["easy", "medium", "hard"]
    matrix = np.zeros((len(order), len(difficulties)))
    for i, (module, _) in enumerate(order):
        for j, difficulty in enumerate(difficulties):
            values = [float(row["SA"]) for row in rows if row["module"] == module and row["difficulty"] == difficulty]
            matrix[i, j] = np.mean(values)

    fig, ax = plt.subplots(figsize=(6.92, 3.44))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(difficulties)), [label.capitalize() for label in difficulties])
    ax.set_yticks(range(len(order)), [label for _, label in order])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)
    ax.set_title("NewtonBench symbolic accuracy across the complete 324-task run", fontsize=9.4, fontweight="bold", pad=23)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{100 * val:.0f}", ha="center", va="center", fontsize=7.4, color="#0f1b25" if val < 0.62 else "white", fontweight="bold")
    for y in np.arange(-0.5, len(order), 1):
        ax.axhline(y, color="white", lw=0.8)
    for x in np.arange(-0.5, len(difficulties), 1):
        ax.axvline(x, color="white", lw=0.8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("symbolic accuracy")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0", "50", "100"])
    fig.text(
        0.5,
        0.015,
        "Each cell averages nine configurations: three law versions and three system variants. Values are exact symbolic accuracy (percent).",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.38)
    fig.savefig(OUT / "newton_coverage_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "newton_coverage_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def newton_coverage_and_growth_composite():
    """A compact main-paper figure: full coverage next to the causal growth audit."""
    path = ROOT / "lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/rows.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = [
        ("m0_gravity", "Gravity"), ("m1_coulomb_force", "Coulomb"), ("m2_magnetic_force", "Magnetic"),
        ("m3_fourier_law", "Fourier"), ("m4_snell_law", "Snell"), ("m5_radioactive_decay", "Decay"),
        ("m6_underdamped_harmonic", "Harmonic"), ("m7_malus_law", "Malus"), ("m8_sound_speed", "Sound"),
        ("m9_hooke_law", "Hooke"), ("m10_be_distribution", "Bose--Einstein"), ("m11_heat_transfer", "Heat"),
    ]
    difficulties = ["easy", "medium", "hard"]
    matrix = np.zeros((len(order), len(difficulties)))
    for i, (module, _) in enumerate(order):
        for j, difficulty in enumerate(difficulties):
            values = [float(row["SA"]) for row in rows if row["module"] == module and row["difficulty"] == difficulty]
            matrix[i, j] = np.mean(values)

    fig = plt.figure(figsize=(7.05, 3.10))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.02, 0.98], height_ratios=[1.0, 0.74], wspace=0.42, hspace=0.58)
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_gate = fig.add_subplot(grid[1, 1])

    im = ax_heat.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(3), ["Easy", "Medium", "Hard"])
    ax_heat.xaxis.tick_top()
    ax_heat.set_yticks(range(len(order)), [label for _, label in order])
    ax_heat.tick_params(axis="both", length=0, labelsize=6.6)
    ax_heat.set_title("Complete NewtonBench coverage", fontsize=8.35, fontweight="bold", pad=17)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax_heat.text(j, i, f"{100 * val:.0f}", ha="center", va="center", fontsize=6.4, color="#12202a" if val < 0.62 else "white", fontweight="bold")
    for y in np.arange(-0.5, len(order), 1):
        ax_heat.axhline(y, color="white", lw=0.55)
    for x in np.arange(-0.5, 3, 1):
        ax_heat.axvline(x, color="white", lw=0.55)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.025)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0", "50", "100"])
    cbar.ax.tick_params(labelsize=6.2)

    palette = {"$O_1$": "#2f7d50", "$O_2$": "#2e6db4", "duplicate": "#c65a37"}
    before = np.array([1.0, 1.0, 0.4326, 0.5090, 0.2232, 1.0, 1.0, 1.0, 0.3535])
    after = np.array([0.5345, 0.2546, 0.4184, 0.5090, 0.2232, 0.3732, 0.2060, 0.6594, 0.2539])
    groups = ["$O_1$", "$O_1$", "$O_2$", "duplicate", "duplicate", "$O_1$", "$O_1$", "$O_1$", "$O_2$"]
    markers = ["o"] * 5 + ["s"] * 4
    for x, y, g, marker in zip(before, after, groups, markers):
        ax_scatter.scatter(x, y, marker=marker, color=palette[g], s=28, edgecolor="white", linewidth=0.45, zorder=3)
    ax_scatter.plot([0, 1.02], [0, 1.02], "--", color="#777777", lw=0.65, zorder=1)
    ax_scatter.fill_between([0, 1.02], [0, 1.02], [0, 0], color="#edf7ed", alpha=0.9)
    ax_scatter.set_xlim(0, 1.02)
    ax_scatter.set_ylim(0, 1.02)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_title("Held-out operator evidence", fontsize=8.35, fontweight="bold", pad=5)
    ax_scatter.set_xlabel("loss before", fontsize=6.8)
    ax_scatter.set_ylabel("loss after", fontsize=6.8)
    ax_scatter.tick_params(labelsize=6.4)
    ax_scatter.grid(color="#e0e0e0", linewidth=0.35)
    ax_scatter.spines[["top", "right"]].set_visible(False)
    ax_scatter.text(0.05, 0.91, "circle: promotion; square: transfer", fontsize=5.55, color="#3d6e44")

    transfer_path = (
        ROOT
        / "lmw/language_transfer/aaai27_newton_language_growth_control_v1/summary.json"
    )
    transfer = json.loads(transfer_path.read_text())
    module_labels = {
        "m2_magnetic_force": "Magnetic",
        "m3_fourier_law": "Fourier",
        "m5_radioactive_decay": "Decay",
        "m9_hooke_law": "Hooke",
    }
    module_colors = {
        "m2_magnetic_force": "#2e6db4",
        "m3_fourier_law": "#2f7d50",
        "m5_radioactive_decay": "#b57918",
        "m9_hooke_law": "#8a56a6",
    }
    for module in module_labels:
        task_rows = [row for row in transfer["task_rows"] if row["module"] == module]
        ax_gate.scatter(
            [row["control_gain"] for row in task_rows],
            [row["selected_gain"] for row in task_rows],
            color=module_colors[module],
            label=module_labels[module],
            s=22,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax_gate.plot([0, 0.86], [0, 0.86], "--", color="#777777", lw=0.65, zorder=1)
    ax_gate.fill_between([0, 0.86], [0, 0.86], [0.86, 0.86], color="#edf7ed", alpha=0.9)
    ax_gate.set_xlim(0.08, 0.86)
    ax_gate.set_ylim(0.08, 0.86)
    ax_gate.tick_params(labelsize=6.2)
    ax_gate.set_xlabel("type-compatible control gain", fontsize=6.8)
    ax_gate.set_ylabel("residual-selected gain", fontsize=6.8)
    ax_gate.set_title("Residual-conditioned selection", fontsize=8.35, fontweight="bold", pad=4)
    ax_gate.text(
        0.105,
        0.79,
        f"mean advantage {transfer['selection_advantage_mean']:+.3f}",
        fontsize=5.7,
        color="#2f6f44",
    )
    ax_gate.legend(
        loc="lower right",
        ncol=2,
        fontsize=5.25,
        frameon=False,
        borderpad=0.1,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    ax_gate.grid(color="#e0e0e0", linewidth=0.35)
    ax_gate.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.42)
    fig.savefig(OUT / "newton_coverage_growth_composite.pdf", bbox_inches="tight")
    fig.savefig(OUT / "newton_coverage_growth_composite.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    method_overview()
    concrete_operator_growth_example()
    language_growth_audit()
    discovery_ablation()
    discovery_external_controls()
    uh_breakdown()
    residual_language_induction_ravr_style()
    scientific_artifact_atlas()
    discovery_case_trace()
    discovery_score_profile()
    newton_coverage_heatmap()
    newton_coverage_and_growth_composite()
