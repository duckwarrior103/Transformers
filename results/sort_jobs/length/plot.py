"""Plot sweep-seq-length results.

Reads:
  - data/seq_<N>.csv files from a run directory (per-epoch logs)
  - data/test_results.csv (merged final generation results)

Produces graphs in a graphs/ subfolder.

Usage:
    python plot.py /path/to/run_<ID>/
    python plot.py          # auto-detects newest run subdirectory
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")]
    if not subdirs:
        raise FileNotFoundError(f"No run_* subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def load_all_epochs(run_dir):
    """Return dict mapping seq_length -> DataFrame with all epochs."""
    all_data = {}
    for fp in sorted(glob.glob(os.path.join(run_dir, "data", "seq_*.csv"))):
        name = os.path.basename(fp).replace(".csv", "").split("_")
        try:
            seq = int(name[1])
        except Exception:
            print(f"  Skipping (bad name): {fp}")
            continue
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
            all_data[seq] = df
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
    return all_data


def load_run_final_from_epochs(all_data):
    """Return DataFrame with one row per sequence length, final-epoch values."""
    records = []
    for seq, df in sorted(all_data.items()):
        last = df.iloc[-1]

        def g(k):
            return float(last[k]) if k in last.index else np.nan

        records.append({
            "seq_length": seq,
            "train_loss": g("train_loss"),
            "val_loss": g("val_loss"),
            "token_acc": g("token_acc"),
            "exact_acc": g("exact_acc"),
        })
    return pd.DataFrame(records).sort_values("seq_length").reset_index(drop=True)


def load_test_results(run_dir):
    """Load merged test_results.csv (generation metrics)."""
    fp = os.path.join(run_dir, "data", "test_results.csv")
    if not os.path.isfile(fp):
        return None

    df = pd.read_csv(fp)
    if df.empty:
        return None

    # Expected columns in your writer:
    # model_type, seq_length, final_token_acc, final_exact_acc, hidden_size, num_layers, num_heads
    if "seq_length" not in df.columns:
        print(f"  Warning: {fp} missing 'seq_length' column; columns={list(df.columns)}")
        return None

    # Ensure numeric
    df["seq_length"] = pd.to_numeric(df["seq_length"], errors="coerce")
    for c in ["final_token_acc", "final_exact_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seq_length"]).sort_values("seq_length").reset_index(drop=True)
    return df


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


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_newest_run_dir(BASE_DIR)
    print(f"Plotting from: {run_dir}")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    all_data = load_all_epochs(run_dir)
    if all_data:
        print(f"Loaded {len(all_data)} sequence lengths (epoch logs)")
        rdf_epochs = load_run_final_from_epochs(all_data)

        plot_epoch_metric(all_data, graphs_dir, "token_acc", "Token Accuracy (%)",
                          "Token Accuracy per Epoch (teacher-forced)", "epoch_token_acc.png", scale=100)
        plot_epoch_metric(all_data, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                          "Exact Accuracy per Epoch (teacher-forced)", "epoch_exact_acc.png", scale=100)
        plot_epoch_metric(all_data, graphs_dir, "train_loss", "Train Loss",
                          "Train Loss per Epoch", "epoch_train_loss.png")
        plot_epoch_metric(all_data, graphs_dir, "val_loss", "Val Loss",
                          "Val Loss per Epoch", "epoch_val_loss.png")

        plot_metric(rdf_epochs, graphs_dir, "seq_length", "exact_acc", "Exact Accuracy (%)",
                    "Seq-Length Sweep — Exact Accuracy vs Seq Length (teacher-forced)",
                    "exact_acc.png", scale=100)
        plot_metric(rdf_epochs, graphs_dir, "seq_length", "token_acc", "Token Accuracy (%)",
                    "Seq-Length Sweep — Token Accuracy vs Seq Length (teacher-forced)",
                    "token_acc.png", scale=100)
        plot_metric(rdf_epochs, graphs_dir, "seq_length", "train_loss", "Train Loss",
                    "Seq-Length Sweep — Train Loss vs Seq Length", "train_loss.png", scale=1.0)
        plot_metric(rdf_epochs, graphs_dir, "seq_length", "val_loss", "Val Loss",
                    "Seq-Length Sweep — Val Loss vs Seq Length", "val_loss.png", scale=1.0)
    else:
        print("No per-epoch result CSVs found (data/seq_*.csv). Skipping epoch/val-loss plots.")

    tdf = load_test_results(run_dir)
    if tdf is None:
        print("No merged test results found (data/test_results.csv).")
    else:
        print(f"Loaded test results: {len(tdf)} rows")
        if "final_exact_acc" in tdf.columns:
            plot_metric(
                tdf, graphs_dir, "seq_length", "final_exact_acc", "Final Exact Accuracy (%)",
                "Seq-Length Sweep — Final Exact Accuracy vs Seq Length (generate)",
                "final_exact_acc.png", scale=100
            )
        if "final_token_acc" in tdf.columns:
            plot_metric(
                tdf, graphs_dir, "seq_length", "final_token_acc", "Final Token Accuracy (%)",
                "Seq-Length Sweep — Final Token Accuracy vs Seq Length (generate)",
                "final_token_acc.png", scale=100
            )

    print("Done.")


if __name__ == "__main__":
    main()