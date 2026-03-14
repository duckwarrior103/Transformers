"""Aggregate sweep results and produce 2D tables and heatmaps.

This script looks for CSV output files produced by the sweep job
with filenames following the pattern: seq_<SEQ>_vocab_<VOCAB>.csv.
Each CSV is expected to contain per-epoch metrics (train_loss, val_loss,
token_acc, exact_acc, etc.). The aggregator:

- loads the final epoch row from each CSV
- builds pivot tables (rows=sequence length, cols=vocab size)
  for train_loss, val_loss, token_acc and exact_acc
- writes matrix CSVs (e.g. token_acc_matrix.csv)
- produces heatmap PNGs (percentages for accuracies)
- writes line plots comparing final accuracies across sequence lengths

Usage:
    python aggregate_results.py

Outputs are written into the same directory as this script.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil

# Directory containing this script
BASE_RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Find the newest directory in the results folder
def find_newest_results_dir(base_dir):
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_dir}")
    newest_dir = max(subdirs, key=os.path.getmtime)
    return newest_dir

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Aggregate sweep results and produce 2D tables and heatmaps.")
parser.add_argument(
    "--results_dir",
    type=str,
    default=None,
    help="Path to the results directory. If not specified, the newest directory will be used."
)
args = parser.parse_args()

# Determine the results directory
if args.results_dir:
    RESULTS_DIR = args.results_dir
    print(f"Processing specified results directory: {RESULTS_DIR}")
else:
    try:
        RESULTS_DIR = find_newest_results_dir(BASE_RESULTS_DIR)
        print(f"Processing newest results directory: {RESULTS_DIR}")
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

# If you want to override the directory manually, set RESULTS_DIR explicitly here
# RESULTS_DIR = "/path/to/specific/results"


def find_csv_files():
    pattern = os.path.join(RESULTS_DIR, "seq_*_maxv_*.csv")
    return sorted(glob.glob(pattern))


def safe_read_csv(fp):
    try:
        return pd.read_csv(fp)
    except Exception:
        return None


def parse_metadata_from_filename(fp):
    # expects seq_<SEQ>_maxv_<MAXV>.csv
    name = os.path.basename(fp)
    parts = name.replace('.csv','').split('_')
    try:
        seq = int(parts[1])
        vocab = int(parts[3])
        return seq, vocab
    except Exception:
        return None, None


def build_records(files):
    records = []
    for fp in files:
        df = safe_read_csv(fp)
        if df is None or df.empty:  # Check if the DataFrame is empty
            print(f"Skipping empty or unreadable file: {fp}")
            continue
        seq, vocab = parse_metadata_from_filename(fp)
        if seq is None or vocab is None:
            print(f"Skipping {fp}, missing metadata")
            continue
        # take last epoch row
        last = df.iloc[-1]
        def getval(k, default=np.nan):
            if k in last.index:
                try:
                    return float(last[k])
                except Exception:
                    # strip possible percent or formatting
                    try:
                        return float(str(last[k]).strip().replace('%', ''))
                    except Exception:
                        return default
            return default
        records.append({
            'seq_length': int(seq),
            'vocab_size': int(vocab),
            'train_loss': getval('train_loss'),
            'val_loss': getval('val_loss'),
            'token_acc': getval('token_acc'),
            'exact_acc': getval('exact_acc')
        })
    return pd.DataFrame(records)


def make_pivot_and_plot(rdf, value_col, cmap='viridis', fmt=':.3f', to_percent=False, vmin=None, vmax=None):
    seqs = sorted(rdf['seq_length'].unique())
    vocabs = sorted(rdf['vocab_size'].unique())
    mat = rdf.pivot(index='seq_length', columns='vocab_size', values=value_col).reindex(index=seqs, columns=vocabs)

    # Optionally convert to percent
    if to_percent:
        mat_display = mat * 100.0
    else:
        mat_display = mat

    out_csv = os.path.join(RESULTS_DIR, f'{value_col}_matrix.csv')
    mat.to_csv(out_csv, float_format='%.6f')
    print(f'Wrote {out_csv}')

    plt.figure(figsize=(max(6, len(vocabs)*0.25), max(4, len(seqs)*0.25)))
    sns.heatmap(mat_display, annot=True, fmt='.2f' if to_percent else '.4f', cmap=cmap, vmin=vmin, vmax=vmax, 
                cbar_kws={'label': (value_col + (' (%)' if to_percent else ''))}, annot_kws={"fontsize": 6})
    plt.title(f'{value_col} (last epoch) — rows=seq_length, cols=vocab_size')
    plt.xlabel('vocab_size')
    plt.ylabel('seq_length')
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f'{value_col}_heatmap.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f'Wrote {out_png}')
    return mat


def plot_final_accuracy_lines(rdf):
    # For each vocab, plot final token and exact acc vs seq_length
    vocabs = sorted(rdf['vocab_size'].unique())
    plt.figure(figsize=(10,6))
    for vocab in vocabs:
        sub = rdf[rdf['vocab_size']==vocab].sort_values('seq_length')
        if sub.empty:
            continue
        plt.plot(sub['seq_length'], sub['token_acc']*100, marker='o', label=f'vocab={vocab}')
    plt.xscale('log', base=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Token Accuracy (%)')
    plt.title('Final Token Accuracy vs Sequence Length (per vocab)')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, 'token_acc_vs_seq_by_vocab.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f'Wrote {out_png}')

    plt.figure(figsize=(10,6))
    for vocab in vocabs:
        sub = rdf[rdf['vocab_size']==vocab].sort_values('seq_length')
        if sub.empty:
            continue
        plt.plot(sub['seq_length'], sub['exact_acc']*100, marker='o', label=f'vocab={vocab}')
    plt.xscale('log', base=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Exact Accuracy (%)')
    plt.title('Final Exact Accuracy vs Sequence Length (per vocab)')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, 'exact_acc_vs_seq_by_vocab.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f'Wrote {out_png}')

def main():
    files = find_csv_files()
    if not files:
        print('No result CSV files found in', RESULTS_DIR)
        return
    print(f'Found {len(files)} CSV files')
    rdf = build_records(files)
    if rdf.empty:
        print('No valid records parsed')
        return

    rdf = rdf.astype({'seq_length': int, 'vocab_size': int})
    rdf = rdf.sort_values(['seq_length','vocab_size']).reset_index(drop=True)

    # save combined long form
    combined_csv = os.path.join(RESULTS_DIR, 'sweep_combined_long.csv')
    rdf.to_csv(combined_csv, index=False, float_format='%.6f')
    print(f'Wrote {combined_csv}')

    # generate matrices and heatmaps (train/val loss)
    make_pivot_and_plot(rdf, 'train_loss', cmap='magma', fmt='.4f')
    make_pivot_and_plot(rdf, 'val_loss', cmap='magma', fmt='.4f')

    # generate matrices and heatmaps for accuracies (as percentages)
    make_pivot_and_plot(rdf, 'token_acc', cmap='viridis', fmt='.2f', to_percent=True)
    make_pivot_and_plot(rdf, 'exact_acc', cmap='viridis', fmt='.2f', to_percent=True)

    # line plots of final accuracy vs sequence length for each vocab
    plot_final_accuracy_lines(rdf)

    print('Done. Matrices and plots written to:', RESULTS_DIR)

if __name__ == '__main__':
    main()
