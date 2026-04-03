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
import json
import os
from utilities.models_configs import get_models_creator_dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

parser = argparse.ArgumentParser(description="Train a transformer for sorting task.")
parser.add_argument("--model_type", type=str, default="standard",
                    choices=["standard", "linear_attention", "gla", "retnet", "deltanet", "gated_deltanet",
                             "gead", "gead_elm64", "gead_elm128", "gead_elm256",
                             "gead_elm64_orth", "gead_elm128_orth", "gead_elm256_orth",
                             "gdn_ek1", "gdn_ek2", "gdn_ek4"],
                    help="Type of model to train")
parser.add_argument("--seq_length", type=int, default=256)
parser.add_argument("--num_data_tokens", type=int, default=64)
parser.add_argument("--train_examples", type=int, default=50000)
parser.add_argument("--val_examples", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--target_params", type=int, default=None,
                    help="If set, override hidden_size so trainable params <= this value (iso-param mode)")
parser.add_argument("--test_results_file", type=str, default="", help="Path to CSV file for logging final test results")
parser.add_argument("--results_file", type=str, default="", help="Path to CSV file for logging per-epoch results")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

MODEL_TYPE = args.model_type
NUM_LAYERS = args.num_layers
NUM_HEADS = args.num_heads
HIDDEN_SIZE = args.hidden_size

TRAIN_EXAMPLES = args.train_examples
VAL_EXAMPLES = args.val_examples
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay

SEQUENCE_LENGTH = args.seq_length

NUM_DATA_TOKENS = args.num_data_tokens # e.g. 64 means integers in range 0-63, 64 tokens
SEP_TOKEN = NUM_DATA_TOKENS # 0-63 for data, 64 for separator, 65 for EOS
EOS_TOKEN = NUM_DATA_TOKENS + 1
VOCAB_SIZE = NUM_DATA_TOKENS + 2 # data tokens + separator + EOS

if args.target_params is not None:
    from utilities.models_configs import find_iso_hidden_size
    args.hidden_size = find_iso_hidden_size(
        args.model_type, args.target_params, VOCAB_SIZE,
        2 * SEQUENCE_LENGTH + 2, args.num_layers, args.num_heads,
    )
    HIDDEN_SIZE = args.hidden_size

print(f"Task: Sorting {SEQUENCE_LENGTH} integers in range [0, {NUM_DATA_TOKENS-1}])"
      f"Config: seq_length={SEQUENCE_LENGTH}, vocab_size={VOCAB_SIZE}, "
      f"train_examples={TRAIN_EXAMPLES}, epochs={EPOCHS}, lr={LEARNING_RATE}, "
      f"hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}, num_heads={NUM_HEADS}, model_type={MODEL_TYPE}")

def init_csv(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "model_type", "train_loss", "val_loss", "token_acc", "exact_acc",
            "epoch_time", "seq_length", "vocab_size", "train_examples",
            "hidden_size", "num_layers", "num_heads", "lr", "batch_size"
        ])

def log_epoch(filepath, epoch, train_loss, val_loss, token_acc, exact_acc, epoch_time):
    if not filepath:
        return
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, MODEL_TYPE, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{token_acc:.6f}", f"{exact_acc:.6f}", f"{epoch_time:.2f}",
            SEQUENCE_LENGTH, NUM_DATA_TOKENS, TRAIN_EXAMPLES,
            args.hidden_size, args.num_layers, args.num_heads,
            LEARNING_RATE, BATCH_SIZE
        ])

def log_test_results(filepath, gen_token_acc, gen_exact_acc):
    if not filepath:
        return
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model_type", "seq_length", "num_data_tokens", "final_token_acc", "final_exact_acc", 
                "hidden_size", "num_layers", "num_heads"
            ])
        writer.writerow([
            MODEL_TYPE, SEQUENCE_LENGTH, NUM_DATA_TOKENS, f"{gen_token_acc:.6f}", f"{gen_exact_acc:.6f}",
            args.hidden_size, args.num_layers, args.num_heads
        ])

def save_config_json(filepath, args, num_params, num_frozen=0, num_buffers=0):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    config = {
        "model_type": args.model_type,
        "model": {
            "hidden_size": args.hidden_size,
            "intermediate_size": 4 * args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "trainable_params": num_params,
            "frozen_params": num_frozen,
            "buffer_params": num_buffers,
        },
        "task": {
            "name": "sort",
            "seq_length": args.seq_length,
            "num_data_tokens": args.num_data_tokens,
            "vocab_size": VOCAB_SIZE,
            "train_examples": args.train_examples,
            "val_examples": args.val_examples,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }
    if args.target_params is not None:
        config["model"]["target_params"] = args.target_params
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)

if args.results_file:
    init_csv(args.results_file)

class SortingDataset(Dataset):
    def __init__(self, num_examples, seq_length, num_data_tokens):
        self.num_examples = num_examples
        self.seq_length = seq_length
        self.num_data_tokens = num_data_tokens

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        numbers = torch.randint(0, self.num_data_tokens, (self.seq_length,))
        sorted_numbers = torch.sort(numbers).values

        full_seq = torch.cat([numbers, torch.tensor([SEP_TOKEN]), sorted_numbers, torch.tensor([EOS_TOKEN])])
        input_ids = full_seq.clone()
        labels = full_seq.clone()
        
        # Mask out the input part of the sequence in the labels to ignore it during loss calculation
        labels[:SEQUENCE_LENGTH+1] = -100

        return input_ids, labels

train_dataset = SortingDataset(TRAIN_EXAMPLES, SEQUENCE_LENGTH, NUM_DATA_TOKENS)
val_dataset = SortingDataset(VAL_EXAMPLES, SEQUENCE_LENGTH, NUM_DATA_TOKENS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
_T = 2 * SEQUENCE_LENGTH + 2
eval_batch_size = min(BATCH_SIZE, max(1, (4 * 1024 * 1024 * 1024) // (_T * VOCAB_SIZE * 2)))
val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

model_creator_dict = get_models_creator_dict()
model_config, model_class = model_creator_dict[MODEL_TYPE]
model = model_class(model_config(VOCAB_SIZE, 2 * SEQUENCE_LENGTH + 2, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS))
model = model.to(device=device, dtype=torch.bfloat16)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
num_buffers = sum(b.numel() for b in model.buffers())
print(f"Model parameters: {num_params:,} trainable | {num_frozen:,} frozen | {num_buffers:,} buffers")

if args.results_file:
    save_config_json(args.results_file.replace(".csv", "_config.json"), args, num_params, num_frozen, num_buffers)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.05 * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

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

    # Slice definitions based on sequence layout:
    # [input (seq_length) | SEP | sorted (seq_length) | EOS]
    pred_slice = slice(SEQUENCE_LENGTH, 2 * SEQUENCE_LENGTH)      # model predicts next token
    target_slice = slice(SEQUENCE_LENGTH + 1, 2 * SEQUENCE_LENGTH + 1)  # ground truth sorted region

    with torch.no_grad():
        for input_ids, labels in tqdm(val_loader, desc="Evaluating"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(dim=-1)
            predicted_sorted = predictions[:, pred_slice]
            expected_sorted = input_ids[:, target_slice]

            correct_tokens += (predicted_sorted == expected_sorted).sum().item()
            total_tokens += predicted_sorted.numel()
            correct_sequences += (predicted_sorted == expected_sorted).all(dim=1).sum().item()
            total_sequences += predicted_sorted.size(0)

    return total_loss / len(val_loader), correct_sequences / total_sequences, correct_tokens / total_tokens

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

TEST_EXAMPLES = 5000
TEST_BATCH_SIZE = 128
DISPLAY_SAMPLES = 5

print("\n" + "="*70)
print(f"AUTOREGRESSIVE TEST ON {TEST_EXAMPLES} RANDOM SAMPLES")
print("="*70)

model.eval()
total_exact = 0
total_token_correct = 0
total_tokens = 0
samples_seen = 0

all_inputs = torch.randint(0, NUM_DATA_TOKENS, (TEST_EXAMPLES, SEQUENCE_LENGTH))
all_expected = torch.sort(all_inputs, dim=1).values
sep_col = torch.full((TEST_EXAMPLES, 1), SEP_TOKEN, dtype=torch.long)
all_prompts = torch.cat([all_inputs, sep_col], dim=1)

for start in tqdm(range(0, TEST_EXAMPLES, TEST_BATCH_SIZE), desc="Testing (generate)"):
    end = min(start + TEST_BATCH_SIZE, TEST_EXAMPLES)
    batch_prompts = all_prompts[start:end].to(device)
    batch_expected = all_expected[start:end]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch_prompts,
            attention_mask=torch.ones_like(batch_prompts),
            max_new_tokens=SEQUENCE_LENGTH + 1,
            do_sample=False,
            eos_token_id=EOS_TOKEN,
        )

    predicted_region = outputs[:, SEQUENCE_LENGTH + 1:]

    for j in range(end - start):
        pred = predicted_region[j].tolist()
        if EOS_TOKEN in pred:
            pred = pred[:pred.index(EOS_TOKEN)]
        expected = batch_expected[j].tolist()

        pred_padded = (pred + [-1] * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]
        token_matches = sum(p == t for p, t in zip(pred_padded, expected))
        exact = pred == expected

        total_token_correct += token_matches
        total_tokens += SEQUENCE_LENGTH
        total_exact += int(exact)

        if samples_seen < DISPLAY_SAMPLES:
            inp = all_inputs[start + j].tolist()
            token_acc = token_matches / SEQUENCE_LENGTH * 100
            print(f"\n  Sample {samples_seen+1}:")
            print(f"  Input:     {inp}")
            print(f"  Predicted: {pred}")
            print(f"  Expected:  {expected}")
            print(f"  Token Acc: {token_acc:.1f}%, Exact: {'✓' if exact else '✗'}")
        samples_seen += 1

gen_token_acc = total_token_correct / total_tokens
gen_exact_acc = total_exact / TEST_EXAMPLES
print(f"\n{'='*70}")
print(f"GENERATION RESULTS ({TEST_EXAMPLES} samples):")
print(f"  Token Accuracy: {gen_token_acc*100:.2f}%")
print(f"  Exact Accuracy: {gen_exact_acc*100:.2f}%")
print(f"{'='*70}")


print("LOGGING FINAL TEST RESULTS TO CSV: {args.test_results_file}")
log_test_results(args.test_results_file, gen_token_acc, gen_exact_acc)

print("\nDone.")