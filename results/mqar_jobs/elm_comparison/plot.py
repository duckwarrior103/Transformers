"""Plot MQAR elm_comparison sweep — GDN baselines vs GEAD ELM variants.

Reads:
  - data/<model>/<model>_seq_<N>.csv  (per-epoch logs)
  - data/test_results.csv             (merged generation results)

Produces:
  graphs/<model>/   — per-model plots
  graphs/           — cross-model comparison, per-pair, and convergence plots

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
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASK = "MQAR"

BASELINES   = ["gdn_hd64", "gdn_hd128", "gdn_hd256"]
ELM_MODELS  = ["gead_elm64", "gead_elm128", "gead_elm256"]
ELM_ORTH    = ["gead_elm64_orth", "gead_elm128_orth", "gead_elm256_orth"]
MODELS = BASELINES + ELM_MODELS + ELM_ORTH

# Paired colours: same hue per K-size triple (GDN / ELM-rand / ELM-orth)
_PAIR_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]

MODEL_COLORS = {
    "gdn_hd64":          _PAIR_COLORS[0],
    "gead_elm64":       _PAIR_COLORS[0],
    "gead_elm64_orth":  _PAIR_COLORS[0],
    "gdn_hd128":          _PAIR_COLORS[1],
    "gead_elm128":      _PAIR_COLORS[1],
    "gead_elm128_orth": _PAIR_COLORS[1],
    "gdn_hd256":          _PAIR_COLORS[2],
    "gead_elm256":      _PAIR_COLORS[2],
    "gead_elm256_orth": _PAIR_COLORS[2],
}

MODEL_LABELS = {
    "gdn_hd64":          "GDN hd=64",
    "gdn_hd128":          "GDN hd=128",
    "gdn_hd256":          "GDN hd=256",
    "gead_elm64":       "ELM-64 rand (K=64)",
    "gead_elm128":      "ELM-128 rand (K=128)",
    "gead_elm256":      "ELM-256 rand (K=256)",
    "gead_elm64_orth":  "ELM-64 orth (K=64)",
    "gead_elm128_orth": "ELM-128 orth (K=128)",
    "gead_elm256_orth": "ELM-256 orth (K=256)",
}

# Linestyles: GDN=dashed, ELM-rand=solid, ELM-orth=dotted
MODEL_LINESTYLES = {
    "gdn_hd64":          "--",
    "gdn_hd128":          "--",
    "gdn_hd256":          "--",
    "gead_elm64":       "-",
    "gead_elm128":      "-",
    "gead_elm256":      "-",
    "gead_elm64_orth":  ":",
    "gead_elm128_orth": ":",
    "gead_elm256_orth": ":",
}

MODEL_MARKERS = {
    "gdn_hd64":          "o",
    "gdn_hd128":          "o",
    "gdn_hd256":          "o",
    "gead_elm64":       "s",
    "gead_elm128":      "s",
    "gead_elm256":      "s",
    "gead_elm64_orth":  "^",
    "gead_elm128_orth": "^",
    "gead_elm256_orth": "^",
}

# Legend handles explaining line-style convention
STYLE_LEGEND = [
    Line2D([0], [0], color="gray", linestyle="--", linewidth=2,   label="GDN (dashed)"),
    Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.5, label="ELM rand (solid)"),
    Line2D([0], [0], color="gray", linestyle=":",  linewidth=1.5, label="ELM orth (dotted)"),
]

# Head-to-head triples (GDN, ELM-rand, ELM-orth, label)
PAIRS = [
    ("gdn_hd64", "gead_elm64",  "gead_elm64_orth",  "K=64"),
    ("gdn_hd128", "gead_elm128", "gead_elm128_orth", "K=128"),
    ("gdn_hd256", "gead_elm256", "gead_elm256_orth", "K=256"),
]


def find_newest_run_dir(base):
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")]
    if not subdirs:
        raise FileNotFoundError(f"No run_* subdirectories found in {base}")
    return max(subdirs, key=os.path.getmtime)


def load_model_epochs(run_dir, model):
    all_data = {}
    pattern = os.path.join(run_dir, "data", model, f"{model}_seq_*.csv")
    for fp in sorted(glob.glob(pattern)):
        name = os.path.basename(fp).replace(".csv", "")
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
    records = []
    for seq, df in sorted(all_data.items()):
        last = df.iloc[-1]
        records.append({
            "seq_length": seq,
            "train_loss": float(last.get("train_loss", np.nan)),
            "val_loss":   float(last.get("val_loss", np.nan)),
            "token_acc":  float(df["token_acc"].max()) if "token_acc" in df.columns else np.nan,
            "exact_acc":  float(df["exact_acc"].max()) if "exact_acc" in df.columns else np.nan,
        })
    return pd.DataFrame(records).sort_values("seq_length").reset_index(drop=True)


def load_test_results(run_dir):
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


def plot_metric(rdf, graphs_dir, xcol, ycol, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rdf[xcol], rdf[ycol] * scale, marker="o", markersize=5, color="steelblue")
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


def _xcol(df):
    """Return 'step' if present, else 'epoch'."""
    return "step" if "step" in df.columns else "epoch"


def plot_epoch_metric(all_data, graphs_dir, metric, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(10, 6))
    seq_lengths = sorted(all_data.keys())
    colors = cm.viridis(np.linspace(0, 1, len(seq_lengths)))
    for seq, color in zip(seq_lengths, colors):
        df = all_data[seq]
        if metric not in df.columns:
            continue
        xc = _xcol(df)
        ax.plot(df[xc], df[metric] * scale,
                marker="o", markersize=3, color=color, label=f"seq={seq}")
    ax.set_xlabel("Step")
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
    model_graphs = os.path.join(graphs_dir, model)
    os.makedirs(model_graphs, exist_ok=True)

    all_data = load_model_epochs(run_dir, model)
    if not all_data:
        print(f"  No epoch data for {model}, skipping per-model plots.")
        return

    print(f"  {model}: loaded {len(all_data)} sequence lengths")
    rdf = load_final_from_epochs(all_data)
    label = MODEL_LABELS.get(model, model)

    plot_epoch_metric(all_data, model_graphs, "token_acc", "Token Accuracy (%)",
                      f"{label} — Token Accuracy per Epoch", "epoch_token_acc.png", scale=100)
    plot_epoch_metric(all_data, model_graphs, "exact_acc", "Exact Accuracy (%)",
                      f"{label} — Exact Accuracy per Epoch", "epoch_exact_acc.png", scale=100)
    plot_epoch_metric(all_data, model_graphs, "train_loss", "Train Loss",
                      f"{label} — Train Loss per Epoch", "epoch_train_loss.png")
    plot_epoch_metric(all_data, model_graphs, "val_loss", "Val Loss",
                      f"{label} — Val Loss per Epoch", "epoch_val_loss.png")

    plot_metric(rdf, model_graphs, "seq_length", "exact_acc", "Exact Accuracy (%)",
                f"{label} — Exact Accuracy vs Seq Length (teacher-forced)", "exact_acc.png", scale=100)
    plot_metric(rdf, model_graphs, "seq_length", "token_acc", "Token Accuracy (%)",
                f"{label} — Token Accuracy vs Seq Length (teacher-forced)", "token_acc.png", scale=100)
    plot_metric(rdf, model_graphs, "seq_length", "train_loss", "Train Loss",
                f"{label} — Train Loss vs Seq Length", "train_loss.png")
    plot_metric(rdf, model_graphs, "seq_length", "val_loss", "Val Loss",
                f"{label} — Val Loss vs Seq Length", "val_loss.png")

    tdf = load_test_results(run_dir)
    if tdf is not None:
        mdf = tdf[tdf["model_type"] == model].copy()
        if not mdf.empty:
            if "final_exact_acc" in mdf.columns:
                plot_metric(mdf, model_graphs, "seq_length", "final_exact_acc", "Final Exact Accuracy (%)",
                            f"{label} — Final Exact Accuracy vs Seq Length (generate)", "final_exact_acc.png", scale=100)
            if "final_token_acc" in mdf.columns:
                plot_metric(mdf, model_graphs, "seq_length", "final_token_acc", "Final Token Accuracy (%)",
                            f"{label} — Final Token Accuracy vs Seq Length (generate)", "final_token_acc.png", scale=100)


def _finalize_comparison_ax(ax, all_seqs, xlabel, ylabel, title):
    ax.set_xscale("log", base=2)
    if all_seqs:
        ax.set_xticks(all_seqs)
        ax.set_xticklabels([str(int(x)) for x in all_seqs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + STYLE_LEGEND, fontsize=9)
    ax.grid(True, alpha=0.4)


def plot_comparison(all_model_finals, graphs_dir, ycol, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODELS:
        rdf = all_model_finals.get(model)
        if rdf is None or ycol not in rdf.columns or rdf[ycol].isna().all():
            continue
        lw = 2.0 if model in BASELINES else 1.5

        ax.plot(rdf["seq_length"], rdf[ycol] * scale,
                marker=MODEL_MARKERS[model], markersize=5, linewidth=lw, linestyle=MODEL_LINESTYLES[model],
                color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
    all_seqs = sorted(set(s for rdf in all_model_finals.values() for s in rdf["seq_length"]))
    _finalize_comparison_ax(ax, all_seqs, "Sequence Length", ylabel, title)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_test_comparison(tdf, graphs_dir, ycol, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODELS:
        mdf = tdf[tdf["model_type"] == model].sort_values("seq_length")
        if mdf.empty or ycol not in mdf.columns:
            continue
        lw = 2.0 if model in BASELINES else 1.5

        ax.plot(mdf["seq_length"], mdf[ycol] * scale,
                marker=MODEL_MARKERS[model], markersize=5, linewidth=lw, linestyle=MODEL_LINESTYLES[model],
                color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
    all_seqs = sorted(tdf["seq_length"].unique())
    _finalize_comparison_ax(ax, all_seqs, "Sequence Length", ylabel, title)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_pair_comparison(all_model_finals, graphs_dir, ycol, ylabel, title_prefix, filename_prefix, scale=1.0):
    """One plot per K-size triple (GDN / ELM-rand / ELM-orth)."""
    for gdn_model, elm_model, elm_orth_model, pair_label in PAIRS:
        fig, ax = plt.subplots(figsize=(9, 5))
        pair_seqs = set()
        for model in (gdn_model, elm_model, elm_orth_model):
            rdf = all_model_finals.get(model)
            if rdf is None or ycol not in rdf.columns or rdf[ycol].isna().all():
                continue
            lw = 2.0 if model in BASELINES else 1.5
            ax.plot(rdf["seq_length"], rdf[ycol] * scale,
                    marker=MODEL_MARKERS[model], markersize=6, linewidth=lw,
                    linestyle=MODEL_LINESTYLES[model],
                    color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
            pair_seqs.update(rdf["seq_length"].tolist())
        _finalize_comparison_ax(ax, sorted(pair_seqs), "Sequence Length", ylabel,
                                f"{title_prefix} — {pair_label}")
        plt.tight_layout()
        out = os.path.join(graphs_dir, f"{filename_prefix}_{pair_label.replace('=', '')}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")


def plot_pair_test_comparison(tdf, graphs_dir, ycol, ylabel, title_prefix, filename_prefix, scale=1.0):
    """One plot per K-size triple from test results."""
    for gdn_model, elm_model, elm_orth_model, pair_label in PAIRS:
        fig, ax = plt.subplots(figsize=(9, 5))
        pair_seqs = set()
        for model in (gdn_model, elm_model, elm_orth_model):
            mdf = tdf[tdf["model_type"] == model].sort_values("seq_length")
            if mdf.empty or ycol not in mdf.columns:
                continue
            lw = 2.0 if model in BASELINES else 1.5
            ax.plot(mdf["seq_length"], mdf[ycol] * scale,
                    marker=MODEL_MARKERS[model], markersize=6, linewidth=lw,
                    linestyle=MODEL_LINESTYLES[model],
                    color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
            pair_seqs.update(mdf["seq_length"].tolist())
        _finalize_comparison_ax(ax, sorted(pair_seqs), "Sequence Length", ylabel,
                                f"{title_prefix} — {pair_label}")
        plt.tight_layout()
        out = os.path.join(graphs_dir, f"{filename_prefix}_{pair_label.replace('=', '')}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")


def plot_convergence(all_model_epochs, graphs_dir, metric, ylabel, title_prefix, filename, scale=1.0):
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
            lw = 2.0 if model in BASELINES else 1.5
            xc = _xcol(df)
            ax.plot(df[xc], df[metric] * scale,
                    marker=MODEL_MARKERS[model], markersize=3, linewidth=lw,
                    linestyle=MODEL_LINESTYLES[model],
                    color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
        ax.set_title(f"seq={seq}")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + STYLE_LEGEND, fontsize=7)
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

    all_model_epochs = {}
    all_model_finals = {}

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
                        f"{TASK} ELM Comparison — Token Accuracy vs Seq Length (teacher-forced)",
                        "comparison_token_acc.png", scale=100)
        plot_comparison(all_model_finals, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                        f"{TASK} ELM Comparison — Exact Accuracy vs Seq Length (teacher-forced)",
                        "comparison_exact_acc.png", scale=100)

        print("\n--- Per-pair comparison plots ---")
        plot_pair_comparison(all_model_finals, graphs_dir, "token_acc", "Token Accuracy (%)",
                             f"{TASK} GDN vs ELM — Token Accuracy (teacher-forced)",
                             "pair_token_acc", scale=100)
        plot_pair_comparison(all_model_finals, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                             f"{TASK} GDN vs ELM — Exact Accuracy (teacher-forced)",
                             "pair_exact_acc", scale=100)

    tdf = load_test_results(run_dir)
    if tdf is not None:
        print("\n--- Cross-model test comparison plots ---")
        if "final_token_acc" in tdf.columns:
            plot_test_comparison(tdf, graphs_dir, "final_token_acc", "Token Accuracy (%)",
                                 f"{TASK} ELM Comparison — Token Accuracy vs Seq Length (generate)",
                                 "comparison_test_token_acc.png", scale=100)
        if "final_exact_acc" in tdf.columns:
            plot_test_comparison(tdf, graphs_dir, "final_exact_acc", "Exact Accuracy (%)",
                                 f"{TASK} ELM Comparison — Exact Accuracy vs Seq Length (generate)",
                                 "comparison_test_exact_acc.png", scale=100)

        print("\n--- Per-pair test comparison plots ---")
        if "final_token_acc" in tdf.columns:
            plot_pair_test_comparison(tdf, graphs_dir, "final_token_acc", "Token Accuracy (%)",
                                      f"{TASK} GDN vs ELM — Token Accuracy (generate)",
                                      "pair_test_token_acc", scale=100)
        if "final_exact_acc" in tdf.columns:
            plot_pair_test_comparison(tdf, graphs_dir, "final_exact_acc", "Exact Accuracy (%)",
                                      f"{TASK} GDN vs ELM — Exact Accuracy (generate)",
                                      "pair_test_exact_acc", scale=100)

    if all_model_epochs:
        print("\n--- Convergence plots ---")
        plot_convergence(all_model_epochs, graphs_dir, "token_acc", "Token Accuracy (%)",
                         f"{TASK} ELM Comparison — Token Accuracy", "convergence_token_acc.png", scale=100)
        plot_convergence(all_model_epochs, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                         f"{TASK} ELM Comparison — Exact Accuracy", "convergence_exact_acc.png", scale=100)

    print("\nDone.")


if __name__ == "__main__":
    main()
