"""Plot sweep-seq-length results.

Reads seq_<N>.csv files from a run directory (final epoch per file) and produces
four line graphs in a graphs/ subfolder, one line per sequence length:
  - exact_acc.png  : exact accuracy (%) vs sequence length
  - token_acc.png  : token accuracy (%) vs sequence length
  - train_loss.png : train loss vs sequence length
  - val_loss.png   : val loss vs sequence length

Usage:
    python plot.py results/sweep-seq-length/run_<ID>/
    python plot.py          # auto-detects newest run subdirectory
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def load_run(run_dir):
    """Return DataFrame with one row per sequence length, final-epoch values."""
    records = []
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
            last = df.iloc[-1]
            def g(k):
                return float(last[k]) if k in last.index else np.nan
            records.append({"seq_length": seq,
                             "train_loss": g("train_loss"), "val_loss": g("val_loss"),
                             "token_acc": g("token_acc"), "exact_acc": g("exact_acc")})
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
    return pd.DataFrame(records).sort_values("seq_length").reset_index(drop=True)


def plot_metric(rdf, graphs_dir, metric, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rdf["seq_length"], rdf[metric] * scale,
            marker="o", markersize=5, color="steelblue")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_newest_run_dir(BASE_DIR)
    print(f"Plotting from: {run_dir}")

    rdf = load_run(run_dir)
    if rdf.empty:
        print("No result CSVs found.")
        return
    print(f"Loaded {len(rdf)} sequence lengths")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    plot_metric(rdf, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                "Seq-Length Sweep — Exact Accuracy vs Seq Length", "exact_acc.png", scale=100)
    plot_metric(rdf, graphs_dir, "token_acc", "Token Accuracy (%)",
                "Seq-Length Sweep — Token Accuracy vs Seq Length", "token_acc.png", scale=100)
    plot_metric(rdf, graphs_dir, "train_loss", "Train Loss",
                "Seq-Length Sweep — Train Loss vs Seq Length", "train_loss.png")
    plot_metric(rdf, graphs_dir, "val_loss", "Val Loss",
                "Seq-Length Sweep — Val Loss vs Seq Length", "val_loss.png")

    print("Done.")


if __name__ == "__main__":
    main()