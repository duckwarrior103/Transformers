"""Aggregate results from a sweep-full 3D run and produce comparison plots.

Expected directory layout:
    results/sweep-full/<RUN_ID>/
        model_standard/seq_16_maxv_64.csv
        model_gla/seq_16_maxv_64.csv
        ...

Outputs (written into <RUN_ID>/):
    exact_acc_heatmap_<model>.png       — per-model (seq × vocab) heatmap
    exact_acc_comparison.png            — all models side-by-side, same colour scale
    exact_acc_diff_<A>_vs_<B>.png       — pairwise difference heatmaps
    sweep_full_combined.csv             — long-form table of all results

Usage:
    python aggregate_full.py --run_dir results/sweep-full/<RUN_ID>
    python aggregate_full.py            # uses newest subdirectory
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


MODEL_ORDER = ["standard", "linear_attention", "gla", "retnet", "deltanet", "gated_deltanet"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def parse_seq_maxv(filename):
    """Parse seq_<SEQ>_maxv_<MAXV>.csv -> (seq, maxv) ints."""
    name = os.path.basename(filename).replace(".csv", "")
    parts = name.split("_")
    try:
        seq = int(parts[1])
        maxv = int(parts[3])
        return seq, maxv
    except Exception:
        return None, None


def load_all_results(run_dir):
    """Return long-form DataFrame with columns: model, seq_length, vocab_size, + metrics."""
    records = []
    for model_dir in sorted(glob.glob(os.path.join(run_dir, "data", "model_*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir).replace("model_", "")
        for fp in sorted(glob.glob(os.path.join(model_dir, "seq_*_maxv_*.csv"))):
            seq, maxv = parse_seq_maxv(fp)
            if seq is None:
                print(f"  Skipping (bad name): {fp}")
                continue
            try:
                df = pd.read_csv(fp)
                if df.empty:
                    continue
                last = df.iloc[-1]
                def g(k):
                    return float(last[k]) if k in last.index else np.nan
                records.append({
                    "model": model,
                    "seq_length": seq,
                    "vocab_size": maxv,
                    "train_loss": g("train_loss"),
                    "val_loss": g("val_loss"),
                    "token_acc": g("token_acc"),
                    "exact_acc": g("exact_acc"),
                })
            except Exception as e:
                print(f"  Warning: could not read {fp}: {e}")

    return pd.DataFrame(records)


def make_pivot(df, model, value="exact_acc"):
    sub = df[df["model"] == model]
    if sub.empty:
        return None
    seqs = sorted(sub["seq_length"].unique())
    vocabs = sorted(sub["vocab_size"].unique())
    return sub.pivot(index="seq_length", columns="vocab_size", values=value).reindex(
        index=seqs, columns=vocabs
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_model_heatmaps(df, run_dir, value="exact_acc", to_percent=True, graphs_dir=None):
    """One heatmap per model."""
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    models += [m for m in df["model"].unique() if m not in MODEL_ORDER]

    # shared colour scale
    all_vals = df[value].dropna()
    vmin = float(all_vals.min()) * (100 if to_percent else 1)
    vmax = float(all_vals.max()) * (100 if to_percent else 1)

    for model in models:
        mat = make_pivot(df, model, value)
        if mat is None:
            continue
        if to_percent:
            mat = mat * 100

        nrows, ncols = mat.shape
        fig, ax = plt.subplots(figsize=(max(5, ncols * 0.55), max(4, nrows * 0.55)))
        sns.heatmap(mat, annot=True, fmt=".1f", cmap="viridis",
                    vmin=vmin, vmax=vmax, ax=ax,
                    cbar_kws={"label": f"{value} {'(%)' if to_percent else ''}"},
                    annot_kws={"fontsize": 7})
        ax.set_title(f"{model} — {value}{' (%)' if to_percent else ''}")
        ax.set_xlabel("vocab_size (max_value)")
        ax.set_ylabel("seq_length")
        plt.tight_layout()
        out = os.path.join(graphs_dir or run_dir, f"{value}_heatmap_{model}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Wrote {out}")


def plot_stacked_comparison(df, run_dir, value="exact_acc", to_percent=True, graphs_dir=None):
    """All model heatmaps in one figure, same colour scale."""
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    models += [m for m in df["model"].unique() if m not in MODEL_ORDER]
    if not models:
        return

    all_vals = df[value].dropna()
    vmin = float(all_vals.min()) * (100 if to_percent else 1)
    vmax = float(all_vals.max()) * (100 if to_percent else 1)

    ncols = min(3, len(models))
    nrows = (len(models) + ncols - 1) // ncols

    # Determine subplot size based on data dimensions
    sample_mat = make_pivot(df, models[0], value)
    if sample_mat is not None:
        r, c = sample_mat.shape
        cell_h, cell_w = max(0.4, 4.0 / max(r, 1)), max(0.4, 6.0 / max(c, 1))
        sub_h = max(3, r * cell_h)
        sub_w = max(4, c * cell_w)
    else:
        sub_h, sub_w = 4, 5

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * sub_w, nrows * sub_h),
                              squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        mat = make_pivot(df, model, value)
        if mat is None:
            ax.axis("off")
            continue
        if to_percent:
            mat = mat * 100
        sns.heatmap(mat, annot=True, fmt=".1f", cmap="viridis",
                    vmin=vmin, vmax=vmax, ax=ax,
                    cbar=(idx == len(models) - 1),
                    cbar_kws={"label": f"{value} {'(%)' if to_percent else ''}"},
                    annot_kws={"fontsize": 6})
        ax.set_title(model, fontsize=11)
        ax.set_xlabel("vocab_size")
        ax.set_ylabel("seq_length")

    # hide unused axes
    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.suptitle(f"{value} comparison across models (last epoch)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(graphs_dir or run_dir, f"{value}_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def plot_diff_heatmaps(df, run_dir, value="exact_acc", to_percent=True, graphs_dir=None):
    """Pairwise difference heatmaps: modelA - modelB."""
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    models += [m for m in df["model"].unique() if m not in MODEL_ORDER]

    pivots = {}
    for m in models:
        mat = make_pivot(df, m, value)
        if mat is not None:
            pivots[m] = mat * (100 if to_percent else 1)

    for i, a in enumerate(models):
        for b in models[i + 1:]:
            if a not in pivots or b not in pivots:
                continue
            diff = pivots[a].subtract(pivots[b])
            if diff.isnull().all().all():
                continue
            abs_max = float(diff.abs().max().max())
            fig, ax = plt.subplots(figsize=(max(5, diff.shape[1] * 0.55),
                                             max(4, diff.shape[0] * 0.55)))
            sns.heatmap(diff, annot=True, fmt=".1f", cmap="RdBu",
                        center=0, vmin=-abs_max, vmax=abs_max, ax=ax,
                        cbar_kws={"label": f"Δ{value} {'(pp)' if to_percent else ''}"},
                        annot_kws={"fontsize": 7})
            ax.set_title(f"{a} minus {b} ({value})")
            ax.set_xlabel("vocab_size")
            ax.set_ylabel("seq_length")
            plt.tight_layout()
            out = os.path.join(graphs_dir or run_dir, f"{value}_diff_{a}_vs_{b}.png")
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to sweep-full run directory. Defaults to newest.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_newest_run_dir(BASE_DIR)
    print(f"Aggregating from: {run_dir}")

    data_dir = os.path.join(run_dir, "data")
    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    df = load_all_results(run_dir)
    if df.empty:
        print("No results found.")
        return

    print(f"Loaded {len(df)} records across models: {sorted(df['model'].unique())}")

    # Save combined long-form CSV
    out_csv = os.path.join(data_dir, "sweep_full_combined.csv")
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Wrote {out_csv}")

    plot_per_model_heatmaps(df, run_dir, graphs_dir=graphs_dir)
    plot_stacked_comparison(df, run_dir, graphs_dir=graphs_dir)
    plot_diff_heatmaps(df, run_dir, graphs_dir=graphs_dir)

    print(f"Done. Plots in {graphs_dir}")


if __name__ == "__main__":
    main()
