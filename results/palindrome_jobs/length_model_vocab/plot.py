"""Plot palindrome length_model_vocab sweep results — all models x seq_lengths x num_data_tokens.

Reads:
  - data/<model>/<model>_seq_<N>_dtokens_<D>.csv       (per-epoch logs)
  - data/test_results.csv                               (merged generation results)

Produces:
  graphs/<model>/   — per-model heatmaps
  graphs/           — cross-model comparison heatmaps, line plots, difficulty surface

Usage:
    python plot.py /path/to/run_<ID>/
    python plot.py          # auto-detects newest run subdirectory
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = ["standard", "linear_attention", "gla", "retnet", "deltanet", "gated_deltanet"]

MODEL_COLORS = {
    "standard": "#1f77b4",
    "linear_attention": "#ff7f0e",
    "gla": "#2ca02c",
    "retnet": "#d62728",
    "deltanet": "#9467bd",
    "gated_deltanet": "#8c564b",
}


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")]
    if not subdirs:
        raise FileNotFoundError(f"No run_* subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def load_test_results(run_dir):
    """Load merged test_results.csv."""
    fp = os.path.join(run_dir, "data", "test_results.csv")
    if not os.path.isfile(fp):
        return None
    df = pd.read_csv(fp)
    if df.empty:
        return None
    for c in ["seq_length", "num_data_tokens"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["final_token_acc", "final_exact_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["seq_length", "num_data_tokens"]).reset_index(drop=True)


def load_model_epochs(run_dir, model):
    """Return dict mapping (seq_length, dtokens) -> DataFrame for a single model."""
    all_data = {}
    pattern = os.path.join(run_dir, "data", model, f"{model}_seq_*_dtokens_*.csv")
    for fp in sorted(glob.glob(pattern)):
        name = os.path.basename(fp).replace(".csv", "")
        # Exclude test_results files
        if "test_results" in name:
            continue
        # e.g. "standard_seq_16_dtokens_8" -> seq=16, dtokens=8
        parts = name.split("_seq_")
        if len(parts) != 2:
            continue
        rest = parts[1]  # "16_dtokens_8"
        rest_parts = rest.split("_dtokens_")
        if len(rest_parts) != 2:
            continue
        try:
            seq = int(rest_parts[0])
            dtokens = int(rest_parts[1])
        except ValueError:
            continue
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
            all_data[(seq, dtokens)] = df
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
    return all_data


def build_heatmap_matrix(tdf_model, metric, seq_lengths, dtokens_list):
    """Build a 2D numpy array (dtokens x seq_length) for the given metric."""
    matrix = np.full((len(dtokens_list), len(seq_lengths)), np.nan)
    for _, row in tdf_model.iterrows():
        seq = row["seq_length"]
        dt = row["num_data_tokens"]
        val = row[metric]
        if seq in seq_lengths and dt in dtokens_list:
            si = seq_lengths.index(seq)
            di = dtokens_list.index(dt)
            matrix[di, si] = val
    return matrix


def plot_single_heatmap(matrix, seq_lengths, dtokens_list, title, filepath,
                        vmin=0.0, vmax=1.0, cmap="RdYlGn", scale=100):
    """Plot a single heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix * scale, aspect="auto", cmap=cmap,
                   vmin=vmin * scale, vmax=vmax * scale, origin="lower")
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([str(s) for s in seq_lengths])
    ax.set_yticks(range(len(dtokens_list)))
    ax.set_yticklabels([str(d) for d in dtokens_list])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Num Data Tokens")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(dtokens_list)):
        for j in range(len(seq_lengths)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val * scale < (vmin + vmax) * scale / 2 else "black"
                ax.text(j, i, f"{val * scale:.1f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {filepath}")


def plot_comparison_heatmaps(tdf, graphs_dir, metric, suptitle, filename,
                             vmin=0.0, vmax=1.0, cmap="RdYlGn", scale=100):
    """Plot a 2x3 grid of heatmaps (one per model) with shared colorbar."""
    seq_lengths = sorted(tdf["seq_length"].unique().astype(int))
    dtokens_list = sorted(tdf["num_data_tokens"].unique().astype(int))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    norm = Normalize(vmin=vmin * scale, vmax=vmax * scale)

    for idx, model in enumerate(MODELS):
        r, c = divmod(idx, 3)
        ax = axes[r][c]
        mdf = tdf[tdf["model_type"] == model]
        matrix = build_heatmap_matrix(mdf, metric, seq_lengths, dtokens_list)

        im = ax.imshow(matrix * scale, aspect="auto", cmap=cmap,
                       norm=norm, origin="lower")
        ax.set_xticks(range(len(seq_lengths)))
        ax.set_xticklabels([str(s) for s in seq_lengths], fontsize=8)
        ax.set_yticks(range(len(dtokens_list)))
        ax.set_yticklabels([str(d) for d in dtokens_list], fontsize=8)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sequence Length", fontsize=9)
        ax.set_ylabel("Num Data Tokens", fontsize=9)

        # Annotate cells
        for i in range(len(dtokens_list)):
            for j in range(len(seq_lengths)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val * scale < (vmin + vmax) * scale / 2 else "black"
                    ax.text(j, i, f"{val * scale:.0f}", ha="center", va="center",
                            fontsize=6, color=text_color)

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_lines_by_seq(tdf, graphs_dir, metric, ylabel, suptitle, filename, scale=100):
    """For each seq_length, plot metric vs num_data_tokens, one line per model."""
    seq_lengths = sorted(tdf["seq_length"].unique().astype(int))
    ncols = 3
    nrows = math.ceil(len(seq_lengths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, seq in enumerate(seq_lengths):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sdf = tdf[tdf["seq_length"] == seq]
        for model in MODELS:
            mdf = sdf[sdf["model_type"] == model].sort_values("num_data_tokens")
            if mdf.empty or metric not in mdf.columns:
                continue
            ax.plot(mdf["num_data_tokens"], mdf[metric] * scale,
                    marker="o", markersize=4, color=MODEL_COLORS[model], label=model)
        ax.set_xscale("log", base=2)
        dtokens_vals = sorted(sdf["num_data_tokens"].unique())
        ax.set_xticks(dtokens_vals)
        ax.set_xticklabels([str(int(x)) for x in dtokens_vals])
        ax.set_title(f"seq_length={seq}")
        ax.set_xlabel("Num Data Tokens")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(seq_lengths), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_lines_by_dtokens(tdf, graphs_dir, metric, ylabel, suptitle, filename, scale=100):
    """For each num_data_tokens, plot metric vs seq_length, one line per model."""
    dtokens_list = sorted(tdf["num_data_tokens"].unique().astype(int))
    ncols = 3
    nrows = math.ceil(len(dtokens_list) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, dt in enumerate(dtokens_list):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ddf = tdf[tdf["num_data_tokens"] == dt]
        for model in MODELS:
            mdf = ddf[ddf["model_type"] == model].sort_values("seq_length")
            if mdf.empty or metric not in mdf.columns:
                continue
            ax.plot(mdf["seq_length"], mdf[metric] * scale,
                    marker="o", markersize=4, color=MODEL_COLORS[model], label=model)
        ax.set_xscale("log", base=2)
        seq_vals = sorted(ddf["seq_length"].unique())
        ax.set_xticks(seq_vals)
        ax.set_xticklabels([str(int(x)) for x in seq_vals])
        ax.set_title(f"num_data_tokens={dt}")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(dtokens_list), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_difficulty_surface(tdf, graphs_dir, metric, title, filename, scale=100):
    """Contour plot of accuracy averaged across all models."""
    seq_lengths = sorted(tdf["seq_length"].unique().astype(int))
    dtokens_list = sorted(tdf["num_data_tokens"].unique().astype(int))

    # Average across models
    avg = tdf.groupby(["seq_length", "num_data_tokens"])[metric].mean().reset_index()
    matrix = np.full((len(dtokens_list), len(seq_lengths)), np.nan)
    for _, row in avg.iterrows():
        si = seq_lengths.index(int(row["seq_length"]))
        di = dtokens_list.index(int(row["num_data_tokens"]))
        matrix[di, si] = row[metric]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix * scale, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100, origin="lower")
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([str(s) for s in seq_lengths])
    ax.set_yticks(range(len(dtokens_list)))
    ax.set_yticklabels([str(d) for d in dtokens_list])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Num Data Tokens")
    ax.set_title(title)

    # Annotate
    for i in range(len(dtokens_list)):
        for j in range(len(seq_lengths)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val * scale < 50 else "black"
                ax.text(j, i, f"{val * scale:.1f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_model_epoch_heatmaps(run_dir, model, model_graphs, seq_lengths, dtokens_list):
    """From per-epoch CSVs, build heatmaps of final-epoch teacher-forced accuracy."""
    all_data = load_model_epochs(run_dir, model)
    if not all_data:
        print(f"  No epoch data for {model}, skipping epoch heatmaps.")
        return

    for metric, label, fname in [
        ("token_acc", "Token Accuracy", "heatmap_epoch_token_acc.png"),
        ("exact_acc", "Exact Accuracy", "heatmap_epoch_exact_acc.png"),
    ]:
        matrix = np.full((len(dtokens_list), len(seq_lengths)), np.nan)
        for (seq, dt), df in all_data.items():
            if seq in seq_lengths and dt in dtokens_list:
                si = seq_lengths.index(seq)
                di = dtokens_list.index(dt)
                if metric in df.columns:
                    matrix[di, si] = df[metric].iloc[-1]

        filepath = os.path.join(model_graphs, fname)
        plot_single_heatmap(matrix, seq_lengths, dtokens_list,
                            f"{model} — {label} (final epoch, teacher-forced)",
                            filepath)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_newest_run_dir(BASE_DIR)
    print(f"Plotting from: {run_dir}")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    tdf = load_test_results(run_dir)
    if tdf is None:
        print("No test_results.csv found. Exiting.")
        return

    seq_lengths = sorted(tdf["seq_length"].unique().astype(int).tolist())
    dtokens_list = sorted(tdf["num_data_tokens"].unique().astype(int).tolist())

    print(f"Seq lengths: {seq_lengths}")
    print(f"Num data tokens: {dtokens_list}")

    for model in MODELS:
        print(f"\n--- {model} ---")
        model_graphs = os.path.join(graphs_dir, model)
        os.makedirs(model_graphs, exist_ok=True)

        mdf = tdf[tdf["model_type"] == model]
        if mdf.empty:
            print(f"  No data for {model}, skipping.")
            continue

        # Generation test heatmaps
        for metric, label, fname in [
            ("final_exact_acc", "Final Exact Accuracy (generate)", "heatmap_final_exact_acc.png"),
            ("final_token_acc", "Final Token Accuracy (generate)", "heatmap_final_token_acc.png"),
        ]:
            if metric in mdf.columns:
                matrix = build_heatmap_matrix(mdf, metric, seq_lengths, dtokens_list)
                filepath = os.path.join(model_graphs, fname)
                plot_single_heatmap(matrix, seq_lengths, dtokens_list,
                                    f"{model} — {label}", filepath)

        # Teacher-forced heatmaps from epoch data
        plot_model_epoch_heatmaps(run_dir, model, model_graphs, seq_lengths, dtokens_list)

    print("\n--- Cross-model comparison heatmaps ---")
    if "final_exact_acc" in tdf.columns:
        plot_comparison_heatmaps(tdf, graphs_dir, "final_exact_acc",
                                 "Palindrome — Model Comparison — Final Exact Accuracy (generate)",
                                 "comparison_heatmap_final_exact_acc.png")
    if "final_token_acc" in tdf.columns:
        plot_comparison_heatmaps(tdf, graphs_dir, "final_token_acc",
                                 "Palindrome — Model Comparison — Final Token Accuracy (generate)",
                                 "comparison_heatmap_final_token_acc.png")

    print("\n--- Line plots by seq_length ---")
    if "final_exact_acc" in tdf.columns:
        plot_lines_by_seq(tdf, graphs_dir, "final_exact_acc", "Exact Accuracy (%)",
                          "Palindrome — Exact Accuracy vs Num Data Tokens (by seq_length)",
                          "comparison_line_by_seq_exact_acc.png")
    if "final_token_acc" in tdf.columns:
        plot_lines_by_seq(tdf, graphs_dir, "final_token_acc", "Token Accuracy (%)",
                          "Palindrome — Token Accuracy vs Num Data Tokens (by seq_length)",
                          "comparison_line_by_seq_token_acc.png")

    print("\n--- Line plots by num_data_tokens ---")
    if "final_exact_acc" in tdf.columns:
        plot_lines_by_dtokens(tdf, graphs_dir, "final_exact_acc", "Exact Accuracy (%)",
                              "Palindrome — Exact Accuracy vs Seq Length (by num_data_tokens)",
                              "comparison_line_by_dtokens_exact_acc.png")
    if "final_token_acc" in tdf.columns:
        plot_lines_by_dtokens(tdf, graphs_dir, "final_token_acc", "Token Accuracy (%)",
                              "Palindrome — Token Accuracy vs Seq Length (by num_data_tokens)",
                              "comparison_line_by_dtokens_token_acc.png")

    print("\n--- Difficulty surface ---")
    if "final_exact_acc" in tdf.columns:
        plot_difficulty_surface(tdf, graphs_dir, "final_exact_acc",
                                "Palindrome — Difficulty Landscape\n(Exact Accuracy averaged across models)",
                                "difficulty_surface_exact_acc.png")
    if "final_token_acc" in tdf.columns:
        plot_difficulty_surface(tdf, graphs_dir, "final_token_acc",
                                "Palindrome — Difficulty Landscape\n(Token Accuracy averaged across models)",
                                "difficulty_surface_token_acc.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
