# Experiments Reference

This document describes the experiment framework for benchmarking FLA (Flash Linear Attention) architectures on the **sequence sorting task**.

---

## 1. Model Catalogue

All models use `hidden_size=512, num_hidden_layers=6, num_heads=8` for fair comparison.

| `--model_type` key | Class | Description |
|--------------------|-------|-------------|
| `standard` | `TransformerForCausalLM` | Standard softmax attention (Flash Attention 2) |
| `linear_attention` | `LinearAttentionForCausalLM` | Linear attention with element-wise product feature map |
| `gla` | `GLAForCausalLM` | Gated Linear Attention — data-dependent decay gates |
| `retnet` | `RetNetForCausalLM` | Retention network — multi-scale exponential decay |
| `deltanet` | `DeltaNetForCausalLM` | Delta rule linear attention with short conv |
| `gated_deltanet` | `GatedDeltaNetForCausalLM` | Gated DeltaNet with improved gating |

All configs are defined in `utilities/models_configs.py`.

---

## 2. Task: Sequence Sorting

Given a sequence of `seq_length` random integers in `[0, max_value)`, the model must output them in sorted order.

**Input format:** `[x₁, x₂, ..., xₙ, EOS]`
**Target format:** `[x₍₁₎, x₍₂₎, ..., x₍ₙ₎, EOS]` (sorted)
**EOS token:** `max_value + 1`
**Vocab size:** `max_value + 2`

**Metrics logged per epoch (in CSV):**
- `train_loss`, `val_loss` — cross-entropy over output tokens only
- `token_acc` — fraction of output tokens predicted correctly
- `exact_acc` — fraction of full sequences predicted exactly correctly

---

## 3. Experiment Types

### 3a. Single Model Run — `jobs/single-model/`

Quick single run for testing a specific config. Edit flags directly in the job file.

```bash
sbatch jobs/single-model/single-model.job
```

Output: printed to `experiment_<JOBID>.out`

---

### 3b. Smoke Test — `jobs/smoke-test/`

**Purpose:** validate that all 6 model types train, log CSVs, and produce plots — before committing GPU hours.

Parameters: `seq=32, max_value=64, epochs=3, train=2000, val=500`
Expected time: ~15 min

```bash
sbatch jobs/smoke-test/smoke-test.job
```

**Pass criteria:** 6 CSVs in `results/sort_jobs/smoke-test/<TIMESTAMP>/`, each with 3 rows. Plots auto-generated.

---

### 3c. Model Comparison Sweep — `jobs/sweep-models/`

**Purpose:** compare all attention mechanisms at a fixed (seq_length, max_value).
Produces the primary model ranking figure.

Default parameters (edit at top of job file):
```
SEQ_LENGTH=64, MAX_VALUE=128, EPOCHS=8, TRAIN_EXAMPLES=50000
```

```bash
sbatch jobs/sweep-models/sweep-models.job
```

Outputs in `results/sort_jobs/sweep-models/<TIMESTAMP>/`:
- `model_<TYPE>.csv` — per-epoch results for each model
- `model_comparison_bar.png` — bar chart of final exact_acc
- `model_comparison_curves.png` — train/val loss learning curves
- `model_comparison_exact_acc.png` — exact accuracy over epochs

Re-run plots only:
```bash
python results/sort_jobs/sweep-models/plot_model_comparison.py --run_dir results/sort_jobs/sweep-models/<TIMESTAMP>
```

---

### 3d. Seq-Length Sweep — `jobs/sweep-seq-length/`

**Purpose:** vary sequence length for a single model type to see how performance scales.

```bash
sbatch jobs/sweep-seq-length/sweep-seq-length.job
```

Outputs in `results/sort_jobs/sweep-seq-length/`:
- `seq_<N>.csv` — results for each sequence length
- `loss_curves.png`, `accuracy_curves.png`, `final_vs_seq_length.png`

Re-run plots:
```bash
python results/sort_jobs/sweep-seq-length/plot_results.py
```

---

### 3e. Seq × Vocab 2D Sweep — `jobs/sweep-seq-vocab/`

**Purpose:** sweep over sequence length AND vocab size (for one model type).

```bash
sbatch jobs/sweep-seq-vocab/sweep-seq-vocab.job
```

Outputs in `results/sort_jobs/sweep-seq-vocab/<TIMESTAMP>/`:
- `seq_<SEQ>_maxv_<MAXV>.csv` — one CSV per (seq, vocab) pair

Re-run aggregation and heatmaps:
```bash
python results/sort_jobs/sweep-seq-vocab/aggregate_results.py --results_dir results/sort_jobs/sweep-seq-vocab/<TIMESTAMP>
```

---

### 3f. Full 3D Sweep — `jobs/sweep-full/`

**Purpose:** all models × all seq_lengths × all vocab_sizes. Most comprehensive experiment.

**Warning:** 864 runs (6 models × 12 seq × 12 vocab). Very long — run smoke test first.

```bash
sbatch jobs/sweep-full/sweep-full.job
```

Outputs in `results/sort_jobs/sweep-full/<TIMESTAMP>/`:
- `model_<TYPE>/seq_<SEQ>_maxv_<MAXV>.csv` — one CSV per training run
- `exact_acc_heatmap_<model>.png` — per-model heatmap
- `exact_acc_comparison.png` — all models side-by-side, same colour scale
- `exact_acc_diff_<A>_vs_<B>.png` — pairwise difference heatmaps
- `sweep_full_combined.csv` — long-form table of all final-epoch results

Re-run aggregation:
```bash
python results/sort_jobs/sweep-full/aggregate_full.py --run_dir results/sort_jobs/sweep-full/<TIMESTAMP>
```

---

## 4. Results Directory Structure

```
results/
  smoke-test/
    <TIMESTAMP>/
      model_standard.csv
      model_gla.csv
      ...
      model_comparison_bar.png

  sweep-models/
    <TIMESTAMP>/
      model_*.csv
      model_comparison_bar.png
      model_comparison_curves.png
      model_comparison_exact_acc.png
    plot_model_comparison.py

  sweep-seq-length/
    seq_512.csv
    seq_1024.csv
    ...
    plot_results.py
    loss_curves.png
    accuracy_curves.png
    final_vs_seq_length.png

  sweep-seq-vocab/
    <TIMESTAMP>/
      seq_*_maxv_*.csv
    aggregate_results.py

  sweep-full/
    <TIMESTAMP>/
      model_standard/seq_*_maxv_*.csv
      model_gla/seq_*_maxv_*.csv
      ...
      exact_acc_heatmap_*.png
      exact_acc_comparison.png
      exact_acc_diff_*.png
      sweep_full_combined.csv
    aggregate_full.py
```

---

## 5. Quick Reference — Recommended Workflow

```bash
# 1. Validate setup (always do this first)
sbatch jobs/smoke-test/smoke-test.job

# 2. Compare models at a fixed scale
sbatch jobs/sweep-models/sweep-models.job

# 3. (Optional) Sweep one model across scales
sbatch jobs/sweep-seq-vocab/sweep-seq-vocab.job

# 4. (Optional) Full 3D experiment — after smoke test passes
sbatch jobs/sweep-full/sweep-full.job
```

---

## 6. CSV Schema

Every result CSV has these columns:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number (1-indexed) |
| `model_type` | e.g. `standard`, `gla`, `deltanet` |
| `train_loss` | Mean training cross-entropy loss |
| `val_loss` | Mean validation cross-entropy loss |
| `token_acc` | Per-token accuracy on output positions |
| `exact_acc` | Fraction of sequences exactly correct |
| `epoch_time` | Seconds for this epoch |
| `seq_length` | Sequence length used |
| `vocab_size` | Vocabulary size (= max_value + 2) |
| `train_examples` | Number of training examples |
| `hidden_size` | Model hidden dimension |
| `num_layers` | Number of layers |
| `num_heads` | Number of attention heads |
| `lr` | Learning rate |
| `batch_size` | Batch size |
