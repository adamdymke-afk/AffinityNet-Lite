"""
visualize_results.py — publication-quality plots for Agentic-Q-Tuner.

Generates two figures and saves them to plots/ in the project root:

  1. convergence_plot.png
     Fitness (balanced accuracy) vs. generation number.
     Shaded bands mark every 5-generation chunk boundary; vertical dashed
     lines show exactly when the Agent increased the Quantum Rotation Angle,
     with an annotation of the new Δθ value.

  2. manhattan_plot.png
     All 5,000 features on the X-axis; Q-population mean selection
     probability (β²) on the Y-axis — styled as a Manhattan plot.
     Ground-truth informative genes are highlighted in a contrasting colour
     with labels for the top-ranked ones.

Usage
-----
    python src/visualize_results.py          # runs its own QEA internally

Or import and call directly with pre-computed results:

    from visualize_results import plot_convergence, plot_manhattan
    plot_convergence(history, angle_events, out_dir)
    plot_manhattan(selection_probs, feature_names, ground_truth_set, out_dir)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")                           # headless / non-interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Ensure src/ siblings are importable
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
PLOTS_DIR = Path(__file__).parent.parent / "plots"

COLOUR_NOISE      = "#A8C5DA"    # muted steel blue — noise features
COLOUR_SIGNAL     = "#E63946"    # vivid red        — ground-truth genes
COLOUR_FITNESS    = "#2B7BB9"    # dark blue        — fitness curve
COLOUR_ANGLE_LINE = "#E07800"    # amber            — rotation-angle events
COLOUR_CHUNK_BAND = "#F0F4F8"    # very light grey  — chunk shading

FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update({
    "font.family":       FONT_FAMILY,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# Plot 1 — Convergence
# ---------------------------------------------------------------------------
def plot_convergence(
    history:       list[float],
    angle_history: list[float],
    chunk_size:    int  = 5,
    out_dir:       Path = PLOTS_DIR,
) -> Path:
    """
    Fitness-vs-Generation convergence plot with Agent decision annotations.

    Parameters
    ----------
    history       : global-best fitness after every generation (length = total_gen)
    angle_history : Δθ value recorded at the *end* of each chunk
    chunk_size    : number of generations per analysis window
    out_dir       : directory in which to save the PNG

    Returns
    -------
    Path to the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    total_gen  = len(history)
    n_chunks   = len(angle_history)
    gens       = list(range(1, total_gen + 1))

    fig, ax = plt.subplots(figsize=(12, 5))

    # --- Alternating chunk bands -------------------------------------------
    for c in range(n_chunks):
        g_start = c * chunk_size + 1
        g_end   = min((c + 1) * chunk_size, total_gen)
        if c % 2 == 0:
            ax.axvspan(g_start - 0.5, g_end + 0.5, color=COLOUR_CHUNK_BAND, zorder=0)

    # --- Vertical lines for rotation-angle increases -----------------------
    prev_angle = angle_history[0]
    for c_idx, angle in enumerate(angle_history[1:], start=2):
        if angle > prev_angle + 1e-9:
            gen_event = (c_idx - 1) * chunk_size          # last gen of previous chunk
            ax.axvline(
                gen_event + 0.5,
                color=COLOUR_ANGLE_LINE,
                linestyle="--",
                linewidth=1.2,
                zorder=2,
            )
            ax.text(
                gen_event + 1.0,
                max(history) * 0.35,
                f"Δθ → {angle / np.pi:.3f}π",
                color=COLOUR_ANGLE_LINE,
                fontsize=8,
                rotation=90,
                va="bottom",
                ha="left",
            )
        prev_angle = angle

    # --- Fitness curve ------------------------------------------------------
    ax.plot(
        gens, history,
        color=COLOUR_FITNESS,
        linewidth=2.2,
        zorder=3,
        label="Global-best balanced accuracy",
    )
    ax.fill_between(gens, history, alpha=0.12, color=COLOUR_FITNESS, zorder=2)

    # --- Chunk boundary ticks -----------------------------------------------
    chunk_boundaries = [c * chunk_size for c in range(1, n_chunks + 1) if c * chunk_size <= total_gen]
    ax.set_xticks(sorted(set([1] + chunk_boundaries + [total_gen])))

    # --- Agent-escalation legend entry ------------------------------------
    angle_patch = mpatches.Patch(
        facecolor="none",
        edgecolor=COLOUR_ANGLE_LINE,
        linestyle="--",
        linewidth=1.2,
        label="Agent increases Δθ (stall detected)",
    )
    ax.legend(handles=[
        plt.Line2D([0], [0], color=COLOUR_FITNESS, linewidth=2.2,
                   label="Global-best balanced accuracy"),
        angle_patch,
        mpatches.Patch(color=COLOUR_CHUNK_BAND, label="5-generation analysis window"),
    ], loc="lower right")

    ax.set_xlim(0.5, total_gen + 0.5)
    ax.set_ylim(-0.02, 1.08)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Balanced Accuracy (3-fold CV)")
    ax.set_title(
        "QEA Convergence with Adaptive Rotation Angle (AgentManager)",
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout()
    out_path = out_dir / "convergence_plot.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved convergence plot → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plot 2 — Manhattan-style feature activity
# ---------------------------------------------------------------------------
def plot_manhattan(
    selection_probs: np.ndarray,
    feature_names:   Sequence[str],
    ground_truth:    set[str],
    top_label_n:     int  = 5,
    out_dir:         Path = PLOTS_DIR,
) -> Path:
    """
    Manhattan-style plot: all 5,000 features coloured by type and sorted by
    genomic index (preserving positional intuition like a real Manhattan plot).

    Parameters
    ----------
    selection_probs : mean β² per feature from the final Q-population
    feature_names   : ordered column names of the feature matrix
    ground_truth    : set of known informative feature names
    top_label_n     : how many ground-truth features to label by name
    out_dir         : directory in which to save the PNG

    Returns
    -------
    Path to the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "feature":   list(feature_names),
        "prob":      selection_probs,
        "index":     range(len(feature_names)),
        "is_signal": [f in ground_truth for f in feature_names],
    })

    fig, ax = plt.subplots(figsize=(14, 5))

    # --- Noise features (two alternating shades for readability) -----------
    noise = df[~df["is_signal"]]
    # Split into two halves by feature index parity for the alternating look
    noise_even = noise[noise["index"] % 2000 < 1000]
    noise_odd  = noise[noise["index"] % 2000 >= 1000]

    ax.scatter(
        noise_even["index"], noise_even["prob"],
        s=3, color=COLOUR_NOISE, alpha=0.55, linewidths=0, rasterized=True,
        label=f"Noise features (n={len(noise):,})",
    )
    ax.scatter(
        noise_odd["index"], noise_odd["prob"],
        s=3, color="#7EB5D0", alpha=0.55, linewidths=0, rasterized=True,
    )

    # --- Ground-truth informative features ---------------------------------
    signal = df[df["is_signal"]].sort_values("prob", ascending=False)

    ax.scatter(
        signal["index"], signal["prob"],
        s=55, color=COLOUR_SIGNAL, alpha=0.95, linewidths=0.5,
        edgecolors="white", zorder=5,
        label=f"Ground-truth informative genes (n={len(signal)})",
    )

    # --- Label the top-ranked ground-truth features -----------------------
    for _, row in signal.head(top_label_n).iterrows():
        ax.annotate(
            row["feature"],
            xy=(row["index"], row["prob"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8,
            color=COLOUR_SIGNAL,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COLOUR_SIGNAL, lw=0.8),
        )

    # --- Significance threshold line (Q-population-mean baseline) ----------
    baseline = selection_probs.mean()
    ax.axhline(baseline, color="grey", linestyle=":", linewidth=1.0, zorder=2)
    ax.text(
        len(feature_names) * 0.02, baseline + 0.005,
        f"Population mean β² = {baseline:.3f}",
        fontsize=8, color="grey",
    )

    ax.set_xlim(-50, len(feature_names) + 50)
    ax.set_ylim(-0.02, min(selection_probs.max() * 1.20, 1.05))
    ax.set_xlabel("Feature Index (gene_0000 → gene_4999)")
    ax.set_ylabel("Selection Probability  (mean β²)")
    ax.set_title(
        "Manhattan Plot — Q-Population Feature Activity Across 5,000 Genes",
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper right", markerscale=2.5)

    fig.tight_layout()
    out_path = out_dir / "manhattan_plot.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved Manhattan plot     → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point — runs the full pipeline and plots
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader   import generate_omics_dataset, get_ground_truth_features
    from agent_manager import AgentManager

    print("[visualize] Generating dataset …")
    X, y         = generate_omics_dataset()
    ground_truth = set(get_ground_truth_features())

    print("[visualize] Running AgentManager …")
    agent   = AgentManager(X, y, total_generations=60, chunk_size=5)
    results = agent.run()

    print("[visualize] Rendering plots …")
    plot_convergence(
        history       = results["history"],
        angle_history = results["angle_history"],
        chunk_size    = 5,
    )
    plot_manhattan(
        selection_probs = results["selection_probs"],
        feature_names   = X.columns,
        ground_truth    = ground_truth,
    )
    print("[visualize] Done.")
