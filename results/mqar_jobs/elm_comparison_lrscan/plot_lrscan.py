"""Plot MQAR elm_comparison LR scan results.

Reads:
  data/<model>/lr_<lr>/<model>_seq_<N>.csv          (step-based training logs)
  data/<model>/lr_<lr>/<model>_test_results_seq_<N>.csv  (per-seq test results)
  data/test_results_lr_<lr>.csv                     (merged test results per LR)

Produces:
  graphs/lrscan_summary.png          — avg peak token_acc vs LR per model
  graphs/comparison_peak_token_acc.png — peak token acc vs seq (best LR)
  graphs/comparison_peak_exact_acc.png — peak exact acc vs seq (best LR)
  graphs/pair_peak_*_K*.png          — per-pair peak accuracy plots
  graphs/convergence_token_acc.png   — step vs token_acc at best LR
  graphs/comparison_test_*.png       — test (generate) accuracy at best LR

Usage:
    python plot_lrscan.py /path/to/run_<ID>/
    python plot_lrscan.py   # auto-detects newest run in elm_comparison_lrscan/
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
LRSCAN_DIR = BASE_DIR

TASK = "MQAR"

BASELINES   = ["gdn_hd64", "gdn_hd128", "gdn_hd256"]
ELM_MODELS  = ["gead_elm64", "gead_elm128", "gead_elm256"]
ELM_ORTH    = ["gead_elm64_orth", "gead_elm128_orth", "gead_elm256_orth"]
MODELS = BASELINES + ELM_MODELS + ELM_ORTH

LRS = [0.001, 0.0005, 0.0001, 0.00005]

_PAIR_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]

MODEL_COLORS = {
    "gdn_hd64": _PAIR_COLORS[0], "gead_elm64": _PAIR_COLORS[0], "gead_elm64_orth": _PAIR_COLORS[0],
    "gdn_hd128": _PAIR_COLORS[1], "gead_elm128": _PAIR_COLORS[1], "gead_elm128_orth": _PAIR_COLORS[1],
    "gdn_hd256": _PAIR_COLORS[2], "gead_elm256": _PAIR_COLORS[2], "gead_elm256_orth": _PAIR_COLORS[2],
}

MODEL_LABELS = {
    "gdn_hd64": "GDN hd=64", "gdn_hd128": "GDN hd=128", "gdn_hd256": "GDN hd=256",
    "gead_elm64": "ELM-64 rand (K=64)", "gead_elm128": "ELM-128 rand (K=128)", "gead_elm256": "ELM-256 rand (K=256)",
    "gead_elm64_orth": "ELM-64 orth (K=64)", "gead_elm128_orth": "ELM-128 orth (K=128)", "gead_elm256_orth": "ELM-256 orth (K=256)",
}

MODEL_LINESTYLES = {
    "gdn_hd64": "--", "gdn_hd128": "--", "gdn_hd256": "--",
    "gead_elm64": "-", "gead_elm128": "-", "gead_elm256": "-",
    "gead_elm64_orth": ":", "gead_elm128_orth": ":", "gead_elm256_orth": ":",
}

MODEL_MARKERS = {
    "gdn_hd64": "o", "gdn_hd128": "o", "gdn_hd256": "o",
    "gead_elm64": "s", "gead_elm128": "s", "gead_elm256": "s",
    "gead_elm64_orth": "^", "gead_elm128_orth": "^", "gead_elm256_orth": "^",
}

STYLE_LEGEND = [
    Line2D([0], [0], color="gray", linestyle="--", linewidth=2,   label="GDN (dashed)"),
    Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.5, label="ELM rand (solid)"),
    Line2D([0], [0], color="gray", linestyle=":",  linewidth=1.5, label="ELM orth (dotted)"),
]

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


def _xcol(df):
    return "step" if "step" in df.columns else "epoch"


def load_model_steps_lr(run_dir, model, lr):
    lr_str = f"{lr:g}"
    all_data = {}
    pattern = os.path.join(run_dir, "data", model, f"lr_{lr_str}", f"{model}_seq_*.csv")
    for fp in sorted(glob.glob(pattern)):
        name = os.path.basename(fp).replace(".csv", "")
        if "_test_results_" in name:
            continue
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


def compute_peak_acc(all_data):
    records = []
    for seq, df in sorted(all_data.items()):
        records.append({
            "seq_length": seq,
            "token_acc": float(df["token_acc"].max()) if "token_acc" in df.columns else np.nan,
            "exact_acc": float(df["exact_acc"].max()) if "exact_acc" in df.columns else np.nan,
        })
    return pd.DataFrame(records).sort_values("seq_length").reset_index(drop=True)


def select_best_lr(run_dir, model, lrs):
    best_lr = None
    best_score = -1
    for lr in lrs:
        data = load_model_steps_lr(run_dir, model, lr)
        if not data:
            continue
        peaks = compute_peak_acc(data)
        score = peaks["token_acc"].mean()
        if score > best_score:
            best_score = score
            best_lr = lr
    return best_lr


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


def plot_lrscan_summary(run_dir, graphs_dir, lrs):
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODELS:
        scores = []
        valid_lrs = []
        for lr in lrs:
            data = load_model_steps_lr(run_dir, model, lr)
            if not data:
                continue
            peaks = compute_peak_acc(data)
            scores.append(peaks["token_acc"].mean() * 100)
            valid_lrs.append(lr)
        if not valid_lrs:
            continue
        lw = 2.0 if model in BASELINES else 1.5
        ax.plot(valid_lrs, scores,
                marker=MODEL_MARKERS[model], markersize=6, linewidth=lw,
                linestyle=MODEL_LINESTYLES[model],
                color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Avg Peak Token Accuracy (%)")
    ax.set_title(f"{TASK} ELM Comparison — LR Scan Summary")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + STYLE_LEGEND, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(graphs_dir, "lrscan_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_peak_comparison(all_model_peaks, graphs_dir, ycol, ylabel, title, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODELS:
        rdf = all_model_peaks.get(model)
        if rdf is None or ycol not in rdf.columns or rdf[ycol].isna().all():
            continue
        lw = 2.0 if model in BASELINES else 1.5
        ax.plot(rdf["seq_length"], rdf[ycol] * scale,
                marker=MODEL_MARKERS[model], markersize=5, linewidth=lw,
                linestyle=MODEL_LINESTYLES[model],
                color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
    all_seqs = sorted(set(s for rdf in all_model_peaks.values() for s in rdf["seq_length"]))
    _finalize_comparison_ax(ax, all_seqs, "Sequence Length", ylabel, title)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_pair_peak(all_model_peaks, graphs_dir, ycol, ylabel, title_prefix, filename_prefix, scale=1.0):
    for gdn_model, elm_model, elm_orth_model, pair_label in PAIRS:
        fig, ax = plt.subplots(figsize=(9, 5))
        pair_seqs = set()
        for model in (gdn_model, elm_model, elm_orth_model):
            rdf = all_model_peaks.get(model)
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


def plot_convergence(all_model_steps, graphs_dir, metric, ylabel, title_prefix, filename, scale=1.0):
    all_seqs = sorted(set(s for data in all_model_steps.values() for s in data.keys()))
    if not all_seqs:
        return
    ncols = 3
    nrows = math.ceil(len(all_seqs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, seq in enumerate(all_seqs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        for model in MODELS:
            data = all_model_steps.get(model, {})
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
    fig.suptitle(f"{title_prefix} — Convergence by Sequence Length (best LR)", fontsize=14, y=1.01)
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
                marker=MODEL_MARKERS[model], markersize=5, linewidth=lw,
                linestyle=MODEL_LINESTYLES[model],
                color=MODEL_COLORS[model], label=MODEL_LABELS.get(model, model))
    all_seqs = sorted(tdf["seq_length"].unique())
    _finalize_comparison_ax(ax, all_seqs, "Sequence Length", ylabel, title)
    plt.tight_layout()
    out = os.path.join(graphs_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_newest_run_dir(LRSCAN_DIR)
    print(f"Plotting from: {run_dir}")

    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    available_lrs = set()
    for model in MODELS:
        model_dir = os.path.join(run_dir, "data", model)
        if not os.path.isdir(model_dir):
            continue
        for d in os.listdir(model_dir):
            if d.startswith("lr_"):
                try:
                    available_lrs.add(float(d[3:]))
                except ValueError:
                    pass
    lrs = sorted(available_lrs, reverse=True) if available_lrs else LRS
    print(f"LRs found: {lrs}")

    print("\n--- LR scan summary ---")
    plot_lrscan_summary(run_dir, graphs_dir, lrs)

    best_lrs = {}
    for model in MODELS:
        best = select_best_lr(run_dir, model, lrs)
        if best is not None:
            best_lrs[model] = best
            print(f"  {model}: best LR = {best}")

    all_model_peaks = {}
    all_model_steps = {}
    for model, lr in best_lrs.items():
        data = load_model_steps_lr(run_dir, model, lr)
        if data:
            all_model_steps[model] = data
            all_model_peaks[model] = compute_peak_acc(data)

    if all_model_peaks:
        print("\n--- Peak accuracy comparison (best LR) ---")
        plot_peak_comparison(all_model_peaks, graphs_dir, "token_acc", "Peak Token Accuracy (%)",
                             f"{TASK} ELM Comparison — Peak Token Accuracy vs Seq Length (best LR)",
                             "comparison_peak_token_acc.png", scale=100)
        plot_peak_comparison(all_model_peaks, graphs_dir, "exact_acc", "Peak Exact Accuracy (%)",
                             f"{TASK} ELM Comparison — Peak Exact Accuracy vs Seq Length (best LR)",
                             "comparison_peak_exact_acc.png", scale=100)

        print("\n--- Per-pair peak accuracy plots ---")
        plot_pair_peak(all_model_peaks, graphs_dir, "token_acc", "Peak Token Accuracy (%)",
                       f"{TASK} GDN vs ELM — Peak Token Accuracy (best LR)",
                       "pair_peak_token_acc", scale=100)
        plot_pair_peak(all_model_peaks, graphs_dir, "exact_acc", "Peak Exact Accuracy (%)",
                       f"{TASK} GDN vs ELM — Peak Exact Accuracy (best LR)",
                       "pair_peak_exact_acc", scale=100)

    if all_model_steps:
        print("\n--- Convergence plots (best LR) ---")
        plot_convergence(all_model_steps, graphs_dir, "token_acc", "Token Accuracy (%)",
                         f"{TASK} ELM Comparison — Token Accuracy", "convergence_token_acc.png", scale=100)
        plot_convergence(all_model_steps, graphs_dir, "exact_acc", "Exact Accuracy (%)",
                         f"{TASK} ELM Comparison — Exact Accuracy", "convergence_exact_acc.png", scale=100)

    test_frames = []
    for model, lr in best_lrs.items():
        lr_str = f"{lr:g}"
        fp = os.path.join(run_dir, "data", "test_results_lr_" + lr_str + ".csv")
        if os.path.isfile(fp):
            df = pd.read_csv(fp)
            mdf = df[df["model_type"] == model].copy()
            if not mdf.empty:
                test_frames.append(mdf)
        else:
            pattern = os.path.join(run_dir, "data", model, f"lr_{lr_str}",
                                   f"{model}_test_results_seq_*.csv")
            for tfp in sorted(glob.glob(pattern)):
                try:
                    tdf = pd.read_csv(tfp)
                    if not tdf.empty:
                        test_frames.append(tdf)
                except Exception:
                    pass

    if test_frames:
        tdf = pd.concat(test_frames, ignore_index=True)
        tdf["seq_length"] = pd.to_numeric(tdf["seq_length"], errors="coerce")
        for c in ["final_token_acc", "final_exact_acc"]:
            if c in tdf.columns:
                tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
        tdf = tdf.dropna(subset=["seq_length"]).sort_values("seq_length").reset_index(drop=True)

        print("\n--- Test comparison plots (best LR) ---")
        if "final_token_acc" in tdf.columns:
            plot_test_comparison(tdf, graphs_dir, "final_token_acc", "Token Accuracy (%)",
                                 f"{TASK} ELM Comparison — Token Accuracy vs Seq Length (generate, best LR)",
                                 "comparison_test_token_acc.png", scale=100)
        if "final_exact_acc" in tdf.columns:
            plot_test_comparison(tdf, graphs_dir, "final_exact_acc", "Exact Accuracy (%)",
                                 f"{TASK} ELM Comparison — Exact Accuracy vs Seq Length (generate, best LR)",
                                 "comparison_test_exact_acc.png", scale=100)

    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Best LR':<12} {'Avg Peak Token Acc':>20}")
    print(f"{'='*80}")
    for model in MODELS:
        lr = best_lrs.get(model)
        if lr is None:
            print(f"{model:<25} {'N/A':<12} {'N/A':>20}")
            continue
        peaks = all_model_peaks.get(model)
        avg = peaks["token_acc"].mean() * 100 if peaks is not None else float("nan")
        print(f"{model:<25} {lr:<12g} {avg:>19.2f}%")
    print(f"{'='*80}")

    print("\nDone.")


if __name__ == "__main__":
    main()
