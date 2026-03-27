"""Plot length_model sweep results — all models x sequence lengths.

Reads:
  - data/<model>/<model>_seq_<N>.csv  (per-epoch logs)
  - data/test_results.csv             (merged generation results)

Produces:
  graphs/<model>/   — per-model plots (same as length/plot.py)
  graphs/           — cross-model comparison and convergence plots

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


def load_model_epochs(run_dir, model):
    """Return dict mapping seq_length -> DataFrame for a single model."""
    all_data = {}
    pattern = os.path.join(run_dir, "data", model, f"{model}_seq_*.csv")
    for fp in sorted(glob.glob(pattern)):
        name = os.path.basename(fp).replace(".csv", "")
        # e.g. "standard_seq_16" -> 16
        parts = name.split("_seq_")
        if len(parts) != 2:
            continue
        try:
            seq = int(parts[1])
        except ValueError:
            continue
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
            all_data[seq] = df
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
    return all_data


def load_final_from_epochs(all_data):
    """Return DataFrame with one row per seq_length from last epoch."""
    records = []
    for seq, df in sorted(all_data.items()):
        last = df.iloc[-1]
        records.append({
            "seq_length": seq,
            "train_loss": float(last.get("train_loss", np.nan)),
            "val_loss": float(last.get("val_loss", np.nan)),
            "token_acc": float(last.get("token_acc", np.nan)),
            "exact_acc": float(last.get("exact_acc", np.nan)),
        })
    return pd.DataFrame(records).sort_values("seq_length").reset_index(drop=True)


def load_test_results(run_dir):
    """Load merged test_results.csv."""
    fp = os.path.join(run_dir, "data", "test_results.csv")
    if not os.path.isfile(fp):
        return None
    df = pd.read_csv(fp)
    if df.empty:
        return None
    df["seq_length"] = pd.to_numeric(df["seq_length"], errors="coerce")
    for c in ["final_token_acc", "final_exact_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["seq_length"]).sort_values("seq_length").reset_index(drop=True)


def plot_metric(rdf, graphs_dir, xcol, ycol, ylabel, title, filename, scale=1.0, log2_x=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rdf[xcol], rdf[ycol] * scale, marker="o", markersize=5, color="steelblue")
    if log2_x:
        ax.set_xscale("log", base=2)
        xticks = sorted(pd.unique(rdf[xcol].dropna()))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) for x in xticks])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def plot_epoch_metric(all_data, graphs_dir, metric, ylabel, title, filename, scale=1.0):
    """Plot metric vs epoch, one line per sequence length."""
    fig, ax = plt.subplots(figsize=(10, 6))
    seq_lengths = sorted(all_data.keys())
    colors = cm.viridis(np.linspace(0, 1, len(seq_lengths)))
    for seq, color in zip(seq_lengths, colors):
        df = all_data[seq]
        if metric not in df.columns:
            continue
        ax.plot(df["epoch"], df[metric] * scale,
                marker="o", markersize=3, color=color, label=f"seq={seq}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_single_model(run_dir, model, graphs_dir):
    """Generate all per-model plots into graphs/<model>/."""
    model_graphs = os.path.join(graphs_dir, model)
    os.makedirs(model_graphs, exist_ok=True)

    all_data = load_model_epochs(run_dir, model)
    if not all_data:
        print(f"  No epoch data for {model}, skipping per-model plots.")
        return

    print(f"  {model}: loaded {len(all_data)} sequence lengths")
    rdf = load_final_from_epochs(all_data)

    # Epoch-level plots
    plot_epoch_metric(all_data, model_graphs, "token_acc", "Token Accuracy (%)",
                      f"{model} — Token Accuracy per Epoch (teacher-forced)", "epoch_token_acc.png", scale=100)
    plot_epoch_metric(all_data, model_graphs, "exact_acc", "Exact Accuracy (%)",
                      f"{model} — Exact Accuracy per Epoch (teacher-forced)", "epoch_exact_acc.png", scale=100)
    plot_epoch_metric(all_data, model_graphs, "train_loss", "Train Loss",
                      f"{model} — Train Loss per Epoch", "epoch_train_loss.png")
    plot_epoch_metric(all_data, model_graphs, "val_loss", "Val Loss",
                      f"{model} — Val Loss per Epoch", "epoch_val_loss.png")

    # Final-epoch vs seq_length plots
    plot_metric(rdf, model_graphs, "seq_length", "exact_acc", "Exact Accuracy (%)",
                f"{model} — Exact Accuracy vs Seq Length (teacher-forced)", "exact_acc.png", scale=100)
    plot_metric(rdf, model_graphs, "seq_length", "token_acc", "Token Accuracy (%)",
                f"{model} — Token Accuracy vs Seq Length (teacher-forced)", "token_acc.png", scale=100)
    plot_metric(rdf, model_graphs, "seq_length", "train_loss", "Train Loss",
                f"{model} — Train Loss vs Seq Length", "train_loss.png")
    plot_metric(rdf, model_graphs, "seq_length", "val_loss", "Val Loss",
                f"{model} — Val Loss vs Seq Length", "val_loss.png")

    # Test (generation) plots from per-model test results
    tdf = load_test_results(run_dir)
    if tdf is not None:
        mdf = tdf[tdf["model_type"] == model].copy()
        if not mdf.empty:
            if "final_exact_acc" in mdf.columns:
                plot_metric(mdf, model_graphs, "seq_length", "final_exact_acc", "Final Exact Accuracy (%)",
                            f"{model} — Final Exact Accuracy vs Seq Length (generate)", "final_exact_acc.png", scale=100)
            if "final_token_acc" in mdf.columns:
                plot_metric(mdf, model_graphs, "seq_length", "final_token_acc", "Final Token Accuracy (%)",
                            f"{model} — Final Token Accuracy vs Seq Length (generate)", "final_token_acc.png", scale=100)


def plot_comparison(all_model_finals, graphs_dir, ycol, ylabel, title, filename, scale=1.0):
    """One line per model, metric vs seq_length."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        if model not in all_model_finals:
            continue
        rdf = all_model_finals[model]
        if ycol not in rdf.columns or rdf[ycol].isna().all():
            continue
        ax.plot(rdf["seq_length"], rdf[ycol] * scale,
                marker="o", markersize=5, color=MODEL_COLORS[model], label=model)
    ax.set_xscale("log", base=2)
    all_seqs = sorted(set(s for rdf in all_model_finals.values() for s in rdf["seq_length"]))
    ax.set_xticks(all_seqs)
    ax.set_xticklabels([str(int(x)) for x in all_seqs])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_test_comparison(tdf, graphs_dir, ycol, ylabel, title, filename, scale=1.0):
    """Comparison plot from merged test_results.csv."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        mdf = tdf[tdf["model_type"] == model].sort_values("seq_length")
        if mdf.empty or ycol not in mdf.columns:
            continue
        ax.plot(mdf["seq_length"], mdf[ycol] * scale,
                marker="o", markersize=5, color=MODEL_COLORS[model], label=model)
    ax.set_xscale("log", base=2)
    all_seqs = sorted(tdf["seq_length"].unique())
    ax.set_xticks(all_seqs)
    ax.set_xticklabels([str(int(x)) for x in all_seqs])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_convergence(all_model_epochs, graphs_dir, metric, ylabel, title_prefix, filename, scale=1.0):
    """Grid of subplots: each subplot is a seq_length, lines are models."""
    # Collect all seq_lengths across models
    all_seqs = sorted(set(s for data in all_model_epochs.values() for s in data.keys()))
    if not all_seqs:
        return

    ncols = 3
    nrows = math.ceil(len(all_seqs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, seq in enumerate(all_seqs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        for model in MODELS:
            data = all_model_epochs.get(model, {})
            if seq not in data:
                continue
            df = data[seq]
            if metric not in df.columns:
                continue
            ax.plot(df["epoch"], df[metric] * scale,
                    marker="o", markersize=3, color=MODEL_COLORS[model], label=model)
        ax.set_title(f"seq={seq}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(len(all_seqs), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{title_prefix} — Convergence by Sequence Length", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_newest_run_dir(BASE_DIR)
    print(f"Plotting from: {run_dir}")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    all_model_epochs = {}   # model -> {seq -> df}
    all_model_finals = {}   # model -> final-epoch df

    for model in MODELS:
        print(f"\n--- {model} ---")
        plot_single_model(run_dir, model, graphs_dir)
        data = load_model_epochs(run_dir, model)
        if data:
            all_model_epochs[model] = data
            all_model_finals[model] = load_final_from_epochs(data)

    if all_model_finals:
        print("\n--- Cross-model comparison plots ---")
        plot_comparison(all_model_finals, graphs_dir, "token_acc", "Token Accuracy (%)",
                        "Model Comparison — Token Accuracy vs Seq Length (teacher-forced)",
                        "comparison_token_acc.png", scale=100)
        plot_comparison(all_model_finals, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                        "Model Comparison — Exact Accuracy vs Seq Length (teacher-forced)",
                        "comparison_exact_acc.png", scale=100)

    tdf = load_test_results(run_dir)
    if tdf is not None:
        print("\n--- Cross-model test comparison plots ---")
        if "final_token_acc" in tdf.columns:
            plot_test_comparison(tdf, graphs_dir, "final_token_acc", "Token Accuracy (%)",
                                 "Model Comparison — Token Accuracy vs Seq Length (generate)",
                                 "comparison_test_token_acc.png", scale=100)
        if "final_exact_acc" in tdf.columns:
            plot_test_comparison(tdf, graphs_dir, "final_exact_acc", "Exact Accuracy (%)",
                                 "Model Comparison — Exact Accuracy vs Seq Length (generate)",
                                 "comparison_test_exact_acc.png", scale=100)

    if all_model_epochs:
        print("\n--- Convergence plots ---")
        plot_convergence(all_model_epochs, graphs_dir, "token_acc", "Token Accuracy (%)",
                         "Token Accuracy", "convergence_token_acc.png", scale=100)
        plot_convergence(all_model_epochs, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                         "Exact Accuracy", "convergence_exact_acc.png", scale=100)

    print("\nDone.")


if __name__ == "__main__":
    main()
