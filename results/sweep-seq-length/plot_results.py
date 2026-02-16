import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

results_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = sorted(glob.glob(os.path.join(results_dir, "seq_*.csv")),
                   key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))

if not csv_files:
    print(f"No CSV files found in {results_dir}")
    exit(1)

# ── 1. Per-epoch loss curves (overlaid by seq_length) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for f in csv_files:
    df = pd.read_csv(f)
    seq = int(df["seq_length"].iloc[0])
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label=f"seq={seq}")
    axes[1].plot(df["epoch"], df["val_loss"], marker="o", label=f"seq={seq}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss"); axes[0].set_title("Train Loss vs Epoch")
axes[0].legend(); axes[0].grid(True)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss"); axes[1].set_title("Val Loss vs Epoch")
axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_curves.png"), dpi=150)
print("Saved loss_curves.png")

# ── 2. Per-epoch accuracy curves (overlaid by seq_length) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for f in csv_files:
    df = pd.read_csv(f)
    seq = int(df["seq_length"].iloc[0])
    axes[0].plot(df["epoch"], df["token_acc"] * 100, marker="o", label=f"seq={seq}")
    axes[1].plot(df["epoch"], df["exact_acc"] * 100, marker="o", label=f"seq={seq}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Token Acc (%)"); axes[0].set_title("Token Accuracy vs Epoch")
axes[0].legend(); axes[0].grid(True)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Exact Acc (%)"); axes[1].set_title("Exact Accuracy vs Epoch")
axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_curves.png"), dpi=150)
print("Saved accuracy_curves.png")

# ── 3. Final metrics vs sequence length ──
seq_lengths, final_train_loss, final_val_loss = [], [], []
final_token_acc, final_exact_acc = [], []

for f in csv_files:
    df = pd.read_csv(f)
    seq_lengths.append(int(df["seq_length"].iloc[0]))
    final_train_loss.append(df["train_loss"].iloc[-1])
    final_val_loss.append(df["val_loss"].iloc[-1])
    final_token_acc.append(df["token_acc"].iloc[-1] * 100)
    final_exact_acc.append(df["exact_acc"].iloc[-1] * 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss vs seq_length
axes[0].plot(seq_lengths, final_train_loss, marker="o", label="Train Loss")
axes[0].plot(seq_lengths, final_val_loss, marker="s", label="Val Loss")
axes[0].set_xlabel("Sequence Length"); axes[0].set_ylabel("Loss")
axes[0].set_title("Final Loss vs Sequence Length")
axes[0].set_xscale("log", base=2); axes[0].legend(); axes[0].grid(True)

# Accuracy vs seq_length
axes[1].plot(seq_lengths, final_token_acc, marker="o", label="Token Acc")
axes[1].plot(seq_lengths, final_exact_acc, marker="s", label="Exact Acc")
axes[1].set_xlabel("Sequence Length"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Final Accuracy vs Sequence Length")
axes[1].set_xscale("log", base=2); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_vs_seq_length.png"), dpi=150)
print("Saved final_vs_seq_length.png")