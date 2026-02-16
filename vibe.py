import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fla.models import GLAConfig, GLAForCausalLM
from fla.models import TransformerConfig, TransformerForCausalLM
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import random
import argparse
import time
import csv
import os
from utilities.models_configs import get_standard_config, get_linear_attention_config, get_gla_config

# ============================================================================
# Device
# ============================================================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================================
# Argument Parsing
# ============================================================================
parser = argparse.ArgumentParser(description="Train a transformer for sorting task.")
parser.add_argument("--seq_length", type=int, default=1024)
parser.add_argument("--max_value", type=int, default=4096)
parser.add_argument("--train_examples", type=int, default=50000)
parser.add_argument("--val_examples", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--results_file", type=str, default="", help="Path to CSV file for logging per-epoch results")
args = parser.parse_args()

MAX_VALUE = args.max_value
EOS_TOKEN = MAX_VALUE + 1
VOCAB_SIZE = EOS_TOKEN + 1

SEQUENCE_LENGTH = args.seq_length
TRAIN_EXAMPLES = args.train_examples
VAL_EXAMPLES = args.val_examples
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay

print(f"Config: seq_length={SEQUENCE_LENGTH}, vocab_size={VOCAB_SIZE}, "
      f"train_examples={TRAIN_EXAMPLES}, epochs={EPOCHS}, lr={LEARNING_RATE}, "
      f"hidden_size={args.hidden_size}")

# ============================================================================
# CSV Logging
# ============================================================================
def init_csv(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "token_acc", "exact_acc",
            "epoch_time", "seq_length", "vocab_size", "train_examples",
            "hidden_size", "num_layers", "num_heads", "lr", "batch_size"
        ])

def log_epoch(filepath, epoch, train_loss, val_loss, token_acc, exact_acc, epoch_time):
    if not filepath:
        return
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{token_acc:.6f}", f"{exact_acc:.6f}", f"{epoch_time:.2f}",
            SEQUENCE_LENGTH, VOCAB_SIZE, TRAIN_EXAMPLES,
            args.hidden_size, args.num_layers, args.num_heads,
            LEARNING_RATE, BATCH_SIZE
        ])

if args.results_file:
    init_csv(args.results_file)

# ============================================================================
# Dataset
# ============================================================================
class SortingDataset(Dataset):
    def __init__(self, num_examples, seq_length, max_value):
        self.num_examples = num_examples
        self.seq_length = seq_length
        self.max_value = max_value

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        numbers = torch.randint(0, self.max_value, (self.seq_length,))
        sorted_numbers = torch.sort(numbers).values
        input_seq = torch.cat([numbers, torch.tensor([EOS_TOKEN])])
        target_seq = torch.cat([sorted_numbers, torch.tensor([EOS_TOKEN])])
        full_seq = torch.cat([input_seq, target_seq])
        input_ids = full_seq.clone()
        labels = full_seq.clone()
        labels[:len(input_seq)] = -100
        return input_ids, labels

train_dataset = SortingDataset(TRAIN_EXAMPLES, SEQUENCE_LENGTH, MAX_VALUE)
val_dataset = SortingDataset(VAL_EXAMPLES, SEQUENCE_LENGTH, MAX_VALUE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ============================================================================
# Model
# ============================================================================
model = TransformerForCausalLM(get_standard_config(VOCAB_SIZE, SEQUENCE_LENGTH))
model = model.to(device=device, dtype=torch.bfloat16)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.05 * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch():
    model.train()
    total_loss = 0
    for input_ids, labels in tqdm(train_loader, desc="Training"):
        input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    total_loss = 0
    correct_sequences = 0
    total_sequences = 0
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, labels in tqdm(val_loader, desc="Evaluating"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            start_idx = SEQUENCE_LENGTH
            end_idx = start_idx + SEQUENCE_LENGTH
            predicted_sorted = predictions[:, start_idx:end_idx]
            expected_sorted = input_ids[:, start_idx + 1:end_idx + 1]
            correct_tokens += (predicted_sorted == expected_sorted).sum().item()
            total_tokens += predicted_sorted.numel()
            for i in range(predicted_sorted.size(0)):
                if torch.equal(predicted_sorted[i], expected_sorted[i]):
                    correct_sequences += 1
            total_sequences += predicted_sorted.size(0)
    return total_loss / len(val_loader), correct_sequences / total_sequences, correct_tokens / total_tokens

# ============================================================================
# Training Loop
# ============================================================================
best_accuracy = 0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch()
    val_loss, val_seq_acc, val_token_acc = evaluate()
    epoch_time = time.time() - epoch_start

    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val Exact Acc: {val_seq_acc*100:.2f}%, Val Token Acc: {val_token_acc*100:.2f}%, "
          f"Epoch Time: {epoch_time:.2f}s")

    log_epoch(args.results_file, epoch + 1, train_loss, val_loss, val_token_acc, val_seq_acc, epoch_time)

    if val_seq_acc > best_accuracy:
        best_accuracy = val_seq_acc
        print("✓ New best model.")

total_time = time.time() - start_time
print(f"\nTotal Training Time: {total_time:.2f}s")

# Final testing
print("\n" + "="*70)
print("TESTING ON RANDOM NEW SAMPLES")
print("="*70)
for _ in range(5):
    nums = [random.randint(0, MAX_VALUE) for _ in range(SEQUENCE_LENGTH)]
    input_ids = torch.tensor([nums + [EOS_TOKEN]], device=device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=SEQUENCE_LENGTH + 1, do_sample=False, eos_token_id=EOS_TOKEN)
    predicted = output[0, len(nums) + 1:].tolist()
    if EOS_TOKEN in predicted:
        predicted = predicted[:predicted.index(EOS_TOKEN)]
    expected = sorted(nums)
    token_acc = sum(p == t for p, t in zip(predicted, expected)) / SEQUENCE_LENGTH * 100
    exact = predicted == expected
    print(f"Input: {nums[:10]}..., Predicted: {predicted[:10]}..., Expected: {expected[:10]}..., Token Acc: {token_acc:.1f}%, Exact: {'✓' if exact else '✗'}")

print("\nDone.")
