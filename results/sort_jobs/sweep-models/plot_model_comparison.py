"""Plot model comparison results from a sweep-models run.

Reads one CSV per model from a run directory (model_<TYPE>.csv) and produces:
  - model_comparison_bar.png   : bar chart of final exact_acc per model
  - model_comparison_curves.png: train_loss + val_loss learning curves, one line per model

Usage:
    python plot_model_comparison.py --run_dir results/sweep-models/<RUN_ID>
    python plot_model_comparison.py          # uses newest subdirectory
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


MODEL_ORDER = ["standard", "linear_attention", "gla", "retnet", "deltanet", "gated_deltanet"]
COLOURS = cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
COLOUR_MAP = dict(zip(MODEL_ORDER, COLOURS))

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def load_run(run_dir):
    """Return dict: model_type -> DataFrame (all epochs)."""
    data = {}
    for fp in sorted(glob.glob(os.path.join(run_dir, "data", "model_*.csv"))):
        model = os.path.basename(fp).replace("model_", "").replace(".csv", "")
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                data[model] = df
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
    return data


def plot_bar(data, run_dir):
    """Bar chart of final exact_acc per model."""
    models, accs = [], []
    for m in MODEL_ORDER:
        if m in data:
            last = data[m].iloc[-1]
            accs.append(float(last.get("exact_acc", 0)) * 100)
            models.append(m)
    # also include any models not in the standard order
    for m in data:
        if m not in MODEL_ORDER:
            last = data[m].iloc[-1]
            accs.append(float(last.get("exact_acc", 0)) * 100)
            models.append(m)

    colours = [COLOUR_MAP.get(m, "grey") for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    bars = ax.bar(models, accs, color=colours, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, min(115, max(accs) * 1.2 + 5))
    ax.set_ylabel("Final Exact Accuracy (%)")
    ax.set_title("Model Comparison — Final Exact Accuracy")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    out = os.path.join(run_dir, "graphs", "model_comparison_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def plot_curves(data, run_dir):
    """Learning curves: train_loss and val_loss over epochs, one line per model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for m in list(MODEL_ORDER) + [k for k in data if k not in MODEL_ORDER]:
        if m not in data:
            continue
        df = data[m]
        if "epoch" not in df.columns:
            df = df.copy()
            df["epoch"] = range(1, len(df) + 1)
        colour = COLOUR_MAP.get(m, "grey")
        if "train_loss" in df.columns:
            axes[0].plot(df["epoch"], df["train_loss"].astype(float),
                         marker="o", markersize=4, label=m, color=colour)
        if "val_loss" in df.columns:
            axes[1].plot(df["epoch"], df["val_loss"].astype(float),
                         marker="o", markersize=4, label=m, color=colour)

    for ax, title, ylabel in zip(
        axes,
        ["Train Loss per Epoch", "Val Loss per Epoch"],
        ["Train Loss", "Val Loss"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle("Model Comparison — Learning Curves", fontsize=13)
    plt.tight_layout()
    out = os.path.join(run_dir, "graphs", "model_comparison_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def plot_acc_curves(data, run_dir):
    """Accuracy curves: exact_acc and token_acc over epochs, one line per model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for m in list(MODEL_ORDER) + [k for k in data if k not in MODEL_ORDER]:
        if m not in data:
            continue
        df = data[m]
        if "epoch" not in df.columns:
            df = df.copy()
            df["epoch"] = range(1, len(df) + 1)
        colour = COLOUR_MAP.get(m, "grey")
        if "exact_acc" in df.columns:
            axes[0].plot(df["epoch"], df["exact_acc"].astype(float) * 100,
                         marker="o", markersize=4, label=m, color=colour)
        if "token_acc" in df.columns:
            axes[1].plot(df["epoch"], df["token_acc"].astype(float) * 100,
                         marker="o", markersize=4, label=m, color=colour)

    for ax, title, ylabel in zip(
        axes,
        ["Exact Accuracy per Epoch", "Token Accuracy per Epoch"],
        ["Exact Accuracy (%)", "Token Accuracy (%)"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle("Model Comparison — Accuracy Curves", fontsize=13)
    plt.tight_layout()
    out = os.path.join(run_dir, "graphs", "model_comparison_acc_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to run directory. Defaults to newest subdir.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_newest_run_dir(BASE_DIR)
    print(f"Plotting from: {run_dir}")

    data = load_run(run_dir)
    if not data:
        print("No model CSVs found.")
        return
    print(f"Loaded models: {list(data.keys())}")

    os.makedirs(os.path.join(run_dir, "graphs"), exist_ok=True)
    plot_bar(data, run_dir)
    plot_curves(data, run_dir)
    plot_acc_curves(data, run_dir)
    print("Done.")


if __name__ == "__main__":
    main()
