"""Plot smoke-test results: accuracy and loss vs epoch.

Reads results.csv from the newest (or specified) run directory.
Produces 4 graphs: token_acc, exact_acc, train_loss, val_loss vs epoch.

Usage:
    python plot.py /path/to/run_<ID>/
    python plot.py          # auto-detects newest run subdirectory
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")]
    if not subdirs:
        raise FileNotFoundError(f"No run_* subdirectories found in {base}")
    # Pick newest run that actually has a results.csv (skip in-progress runs)
    for d in sorted(subdirs, key=os.path.getmtime, reverse=True):
        if os.path.isfile(os.path.join(d, "data", "results.csv")):
            return d
    raise FileNotFoundError(f"No run with data/results.csv found in {base}")


def load_results(run_dir):
    fp = os.path.join(run_dir, "data", "results.csv")
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"No results.csv found in {run_dir}/data/")
    df = pd.read_csv(fp)
    if df.empty:
        raise ValueError(f"results.csv is empty in {run_dir}")
    return df


def plot_metric(df, graphs_dir, metric, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df[metric] * scale, marker="o", markersize=4, color="steelblue")
    ax.set_xlabel("Epoch")
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

    df = load_results(run_dir)
    info = f"model={df['model_type'].iloc[0]} seq={df['seq_length'].iloc[0]}"
    print(f"Loaded {len(df)} epochs — {info}")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    plot_metric(df, graphs_dir, "token_acc", "Token Accuracy (%)",
                f"MQAR Smoke — Token Accuracy vs Epoch\n({info})",
                "token_acc.png", scale=100)
    plot_metric(df, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                f"MQAR Smoke — Exact Accuracy vs Epoch\n({info})",
                "exact_acc.png", scale=100)
    plot_metric(df, graphs_dir, "train_loss", "Train Loss",
                f"MQAR Smoke — Train Loss vs Epoch\n({info})",
                "train_loss.png")
    plot_metric(df, graphs_dir, "val_loss", "Val Loss",
                f"MQAR Smoke — Val Loss vs Epoch\n({info})",
                "val_loss.png")

    print("Done.")


if __name__ == "__main__":
    main()
