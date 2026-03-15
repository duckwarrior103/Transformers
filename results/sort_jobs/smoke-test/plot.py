"""Plot smoke-test results for all models.

Reads model_<TYPE>.csv files from a run directory and produces four line graphs
and two bar charts in a graphs/ subfolder:
  - exact_acc.png       : exact accuracy (%) vs epoch, one line per model
  - token_acc.png       : token accuracy (%) vs epoch, one line per model
  - train_loss.png      : training loss vs epoch, one line per model
  - val_loss.png        : validation loss vs epoch, one line per model
  - comparison_bar.png  : final exact & token accuracy side by side per model

Usage:
    python plot.py results/smoke-test/run_20260310T172646/
    python plot.py          # auto-detects newest run subdirectory
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


MODEL_ORDER = ["standard", "linear_attention", "gla", "retnet", "deltanet", "gated_deltanet"]
COLOURS = cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
COLOUR_MAP = dict(zip(MODEL_ORDER, COLOURS))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def load_run(run_dir):
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


def plot_metric(data, graphs_dir, metric, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in list(MODEL_ORDER) + [k for k in data if k not in MODEL_ORDER]:
        if m not in data:
            continue
        df = data[m]
        if metric not in df.columns:
            continue
        epochs = df["epoch"].astype(int) if "epoch" in df.columns else range(1, len(df) + 1)
        ax.plot(epochs, df[metric].astype(float) * scale,
                marker="o", markersize=4, label=m, color=COLOUR_MAP.get(m, "grey"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def plot_bar_comparison(data, graphs_dir):
    """Grouped bar chart: exact accuracy and token accuracy side by side per model."""
    all_models = list(MODEL_ORDER) + [k for k in data if k not in MODEL_ORDER]
    models = [m for m in all_models if m in data
              and "exact_acc" in data[m].columns and "token_acc" in data[m].columns]
    if not models:
        return

    exact_vals = [float(data[m]["exact_acc"].iloc[-1]) * 100 for m in models]
    token_vals = [float(data[m]["token_acc"].iloc[-1]) * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    _, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, exact_vals, width, label="Exact Accuracy",
                   color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, token_vals, width, label="Token Accuracy",
                   color="#DD8452", edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars1, exact_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, token_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Smoke Test — Final Exact & Token Accuracy by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, max(exact_vals + token_vals) * 1.15 if (exact_vals or token_vals) else 1)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, "comparison_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_newest_run_dir(BASE_DIR)
    print(f"Plotting from: {run_dir}")

    data = load_run(run_dir)
    if not data:
        print("No model CSVs found.")
        return
    print(f"Loaded models: {list(data.keys())}")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    plot_metric(data, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                "Smoke Test — Exact Accuracy per Epoch", "exact_acc.png", scale=100)
    plot_metric(data, graphs_dir, "token_acc", "Token Accuracy (%)",
                "Smoke Test — Token Accuracy per Epoch", "token_acc.png", scale=100)
    plot_metric(data, graphs_dir, "train_loss", "Train Loss",
                "Smoke Test — Train Loss per Epoch", "train_loss.png")
    plot_metric(data, graphs_dir, "val_loss", "Val Loss",
                "Smoke Test — Val Loss per Epoch", "val_loss.png")

    plot_bar_comparison(data, graphs_dir)

    print("Done.")


if __name__ == "__main__":
    main()