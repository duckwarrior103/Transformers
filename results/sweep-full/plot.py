"""Plot sweep-full results (6 models × seq × vocab).

Reads model_<TYPE>/seq_<SEQ>_maxv_<MAXV>.csv files from a run directory
(final epoch per file) and produces line graphs in a graphs/ subfolder.

For each metric (exact_acc, token_acc, train_loss, val_loss):
  - <metric>_vs_seq_<model>.png : one line per vocab size, x=seq_length
  - <metric>_vs_seq_all.png     : all models on one plot (final vocab size only,
                                   as a quick comparison)

Usage:
    python plot.py results/sweep-full/run_<ID>/
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
MODEL_COLOURS = cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
MODEL_COLOUR_MAP = dict(zip(MODEL_ORDER, MODEL_COLOURS))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def load_run(run_dir):
    """Return long-form DataFrame: model, seq_length, vocab_size, metrics."""
    records = []
    for model_dir in sorted(glob.glob(os.path.join(run_dir, "model_*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir).replace("model_", "")
        for fp in sorted(glob.glob(os.path.join(model_dir, "seq_*_maxv_*.csv"))):
            name = os.path.basename(fp).replace(".csv", "").split("_")
            try:
                seq = int(name[1])
                vocab = int(name[3])
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
                records.append({"model": model, "seq_length": seq, "vocab_size": vocab,
                                 "train_loss": g("train_loss"), "val_loss": g("val_loss"),
                                 "token_acc": g("token_acc"), "exact_acc": g("exact_acc")})
            except Exception as e:
                print(f"  Warning: could not read {fp}: {e}")
    return pd.DataFrame(records)


def plot_per_model(df, graphs_dir, metric, ylabel, scale=1.0):
    """One plot per model: x=seq_length, one line per vocab size."""
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    models += [m for m in df["model"].unique() if m not in MODEL_ORDER]

    for model in models:
        sub = df[df["model"] == model]
        vocabs = sorted(sub["vocab_size"].unique())
        colours = cm.viridis(np.linspace(0.1, 0.9, max(len(vocabs), 1)))

        fig, ax = plt.subplots(figsize=(10, 5))
        for vocab, colour in zip(vocabs, colours):
            vsub = sub[sub["vocab_size"] == vocab].sort_values("seq_length")
            if vsub.empty:
                continue
            ax.plot(vsub["seq_length"], vsub[metric] * scale,
                    marker="o", markersize=4, label=f"vocab={vocab}", color=colour)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model} — {ylabel} vs Seq Length")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out = os.path.join(graphs_dir, f"{metric}_vs_seq_{model}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Wrote {out}")


def plot_all_models(df, graphs_dir, metric, ylabel, scale=1.0):
    """All models on one plot using the largest shared vocab size."""
    shared_vocabs = set.intersection(*[
        set(df[df["model"] == m]["vocab_size"].unique())
        for m in df["model"].unique()
    ]) if not df.empty else set()

    vocab = max(shared_vocabs) if shared_vocabs else df["vocab_size"].max()
    sub = df[df["vocab_size"] == vocab]

    models = [m for m in MODEL_ORDER if m in sub["model"].values]
    models += [m for m in sub["model"].unique() if m not in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        msub = sub[sub["model"] == model].sort_values("seq_length")
        if msub.empty:
            continue
        ax.plot(msub["seq_length"], msub[metric] * scale,
                marker="o", markersize=4, label=model,
                color=MODEL_COLOUR_MAP.get(model, "grey"))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.set_title(f"All Models — {ylabel} vs Seq Length (vocab={vocab})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, f"{metric}_vs_seq_all.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_newest_run_dir(BASE_DIR)
    print(f"Plotting from: {run_dir}")

    df = load_run(run_dir)
    if df.empty:
        print("No result CSVs found.")
        return
    print(f"Loaded {len(df)} records across models: {sorted(df['model'].unique())}")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    for metric, ylabel, scale in [
        ("exact_acc",  "Exact Accuracy (%)", 100),
        ("token_acc",  "Token Accuracy (%)", 100),
        ("train_loss", "Train Loss",         1.0),
        ("val_loss",   "Val Loss",           1.0),
    ]:
        plot_per_model(df, graphs_dir, metric, ylabel, scale)
        plot_all_models(df, graphs_dir, metric, ylabel, scale)

    print("Done.")


if __name__ == "__main__":
    main()