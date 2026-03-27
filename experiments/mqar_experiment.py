import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fla.models import GLAConfig, GLAForCausalLM
from fla.models import TransformerConfig, TransformerForCausalLM
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import random
import numpy as np
import argparse
import time
import csv
import os
from utilities.models_configs import get_models_creator_dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

parser = argparse.ArgumentParser(description="Train a transformer for MQAR (Multi-Query Associative Recall) task.")
parser.add_argument("--model_type", type=str, default="standard", choices=["standard", "linear_attention", "gla", "retnet", "deltanet", "gated_deltanet", "gead"], help="Type of model to train")
parser.add_argument("--seq_length", type=int, default=64)
parser.add_argument("--num_data_tokens", type=int, default=1024)
parser.add_argument("--train_examples", type=int, default=50000)
parser.add_argument("--val_examples", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--test_results_file", type=str, default="", help="Path to CSV file for logging final test results")
parser.add_argument("--results_file", type=str, default="", help="Path to CSV file for logging per-epoch results")
args = parser.parse_args()

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

NUM_DATA_TOKENS = args.num_data_tokens
NUM_KEYS = NUM_DATA_TOKENS // 2          # keys:   [0, NUM_KEYS)
NUM_VALUES = NUM_DATA_TOKENS - NUM_KEYS  # values: [NUM_KEYS, NUM_DATA_TOKENS)
SEP_TOKEN = NUM_DATA_TOKENS              # separator token
VOCAB_SIZE = NUM_DATA_TOKENS + 1         # data tokens + separator (no EOS)

# Derived MQAR dimensions
NUM_KV_PAIRS = SEQUENCE_LENGTH // 4
NUM_QUERIES = NUM_KV_PAIRS // 2
TOTAL_TOKENS = 2 * NUM_KV_PAIRS + 1 + 2 * NUM_QUERIES  # actual sequence length

assert NUM_KV_PAIRS > 0, f"seq_length={SEQUENCE_LENGTH} too small: need at least 4 for 1 KV pair"
assert NUM_QUERIES > 0, f"seq_length={SEQUENCE_LENGTH} too small: need at least 8 for 1 query"
assert NUM_KV_PAIRS <= NUM_KEYS, f"num_kv_pairs={NUM_KV_PAIRS} > num_keys={NUM_KEYS}: not enough unique keys"

print(f"Task: MQAR with seq_length={SEQUENCE_LENGTH} "
      f"(num_kv_pairs={NUM_KV_PAIRS}, num_queries={NUM_QUERIES}, total_tokens={TOTAL_TOKENS})\n"
      f"Vocab: {NUM_DATA_TOKENS} data tokens (keys [0,{NUM_KEYS}), values [{NUM_KEYS},{NUM_DATA_TOKENS})), "
      f"sep={SEP_TOKEN}, model_vocab={VOCAB_SIZE}\n"
      f"Config: train_examples={TRAIN_EXAMPLES}, epochs={EPOCHS}, lr={LEARNING_RATE}, "
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

if args.results_file:
    init_csv(args.results_file)

class MQARDataset(Dataset):
    def __init__(self, num_examples, num_kv_pairs, num_queries, num_keys, num_values, sep_token):
        self.num_examples = num_examples
        self.num_kv_pairs = num_kv_pairs
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.num_values = num_values
        self.sep_token = sep_token
        self.kv_len = 2 * num_kv_pairs

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        keys = torch.randperm(self.num_keys)[:self.num_kv_pairs]
        values = torch.randint(self.num_keys, self.num_keys + self.num_values, (self.num_kv_pairs,))

        kv_block = torch.stack([keys, values], dim=1).reshape(-1)

        query_indices = torch.randperm(self.num_kv_pairs)[:self.num_queries]
        query_keys = keys[query_indices]
        query_values = values[query_indices]

        qa_block = torch.stack([query_keys, query_values], dim=1).reshape(-1)

        full_seq = torch.cat([kv_block, torch.tensor([self.sep_token]), qa_block])

        # Labels: -100 everywhere except answer positions
        labels = torch.full_like(full_seq, -100)
        for i in range(self.num_queries):
            answer_pos = self.kv_len + 1 + 2 * i + 1
            labels[answer_pos] = full_seq[answer_pos]

        return full_seq, labels

train_dataset = MQARDataset(TRAIN_EXAMPLES, NUM_KV_PAIRS, NUM_QUERIES, NUM_KEYS, NUM_VALUES, SEP_TOKEN)
val_dataset = MQARDataset(VAL_EXAMPLES, NUM_KV_PAIRS, NUM_QUERIES, NUM_KEYS, NUM_VALUES, SEP_TOKEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
_T = TOTAL_TOKENS
eval_batch_size = min(BATCH_SIZE, max(1, (4 * 1024 * 1024 * 1024) // (_T * VOCAB_SIZE * 2)))
val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

model_creator_dict = get_models_creator_dict()
model_config, model_class = model_creator_dict[MODEL_TYPE]
model = model_class(model_config(VOCAB_SIZE, TOTAL_TOKENS, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS))
model = model.to(device=device, dtype=torch.bfloat16)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params:,}")

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

# Pre-compute answer and query positions for evaluation
KV_LEN = 2 * NUM_KV_PAIRS
ANSWER_POSITIONS = [KV_LEN + 1 + 2 * i + 1 for i in range(NUM_QUERIES)]
# In HuggingFace causal LM, logits[t] predicts token t+1, so to predict
# the answer at position answer_pos, we look at logits[answer_pos - 1] (the query position)
QUERY_POSITIONS = [ap - 1 for ap in ANSWER_POSITIONS]

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

            predictions = outputs.logits.argmax(dim=-1)
            pred_answers = predictions[:, QUERY_POSITIONS]
            true_answers = input_ids[:, ANSWER_POSITIONS]

            correct_tokens += (pred_answers == true_answers).sum().item()
            total_tokens += pred_answers.numel()
            correct_sequences += (pred_answers == true_answers).all(dim=1).sum().item()
            total_sequences += pred_answers.size(0)

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

for start in tqdm(range(0, TEST_EXAMPLES, TEST_BATCH_SIZE), desc="Testing (generate)"):
    end = min(start + TEST_BATCH_SIZE, TEST_EXAMPLES)
    batch_size = end - start

    batch_keys = []
    batch_values = []
    batch_query_keys = []
    batch_query_values = []

    for _ in range(batch_size):
        keys = torch.randperm(NUM_KEYS)[:NUM_KV_PAIRS]
        values = torch.randint(NUM_KEYS, NUM_KEYS + NUM_VALUES, (NUM_KV_PAIRS,))
        query_indices = torch.randperm(NUM_KV_PAIRS)[:NUM_QUERIES]
        batch_keys.append(keys)
        batch_values.append(values)
        batch_query_keys.append(keys[query_indices])
        batch_query_values.append(values[query_indices])

    prompts = []
    for i in range(batch_size):
        kv_block = torch.stack([batch_keys[i], batch_values[i]], dim=1).reshape(-1)
        prompt = torch.cat([kv_block, torch.tensor([SEP_TOKEN])])
        prompts.append(prompt)
    prompts = torch.stack(prompts).to(device)  # (batch, 2*NUM_KV_PAIRS + 1)

    pred_answers_list = []
    current_seq = prompts

    with torch.no_grad():
        for q_idx in range(NUM_QUERIES):
            q_key = torch.stack([batch_query_keys[i][q_idx] for i in range(batch_size)]).unsqueeze(1).to(device)
            current_seq = torch.cat([current_seq, q_key], dim=1)

            outputs = model.generate(
                input_ids=current_seq,
                attention_mask=torch.ones_like(current_seq),
                max_new_tokens=1,
                do_sample=False,
            )

            pred_answer = outputs[:, -1]  # (batch,)
            pred_answers_list.append(pred_answer)

            current_seq = outputs

    pred_answers = torch.stack(pred_answers_list, dim=1).cpu()
    expected_answers = torch.stack(batch_query_values)  # (batch, NUM_QUERIES)

    for j in range(batch_size):
        pred = pred_answers[j].tolist()
        expected = expected_answers[j].tolist()

        token_matches = sum(p == t for p, t in zip(pred, expected))
        exact = pred == expected

        total_token_correct += token_matches
        total_tokens += NUM_QUERIES
        total_exact += int(exact)

        if samples_seen < DISPLAY_SAMPLES:
            kv_block = torch.stack([batch_keys[j], batch_values[j]], dim=1).reshape(-1).tolist()
            queries = batch_query_keys[j].tolist()
            token_acc = token_matches / NUM_QUERIES * 100
            print(f"\n  Sample {samples_seen+1}:")
            print(f"  KV pairs: {list(zip(kv_block[0::2], kv_block[1::2]))}")
            print(f"  Queries:    {queries}")
            print(f"  Predicted:  {pred}")
            print(f"  Expected:   {expected}")
            print(f"  Token Acc: {token_acc:.1f}%, Exact: {'✓' if exact else '✗'}")
        samples_seen += 1

gen_token_acc = total_token_correct / total_tokens
gen_exact_acc = total_exact / TEST_EXAMPLES
print(f"\n{'='*70}")
print(f"GENERATION RESULTS ({TEST_EXAMPLES} samples):")
print(f"  Token Accuracy: {gen_token_acc*100:.2f}%")
print(f"  Exact Accuracy: {gen_exact_acc*100:.2f}%")
print(f"{'='*70}")

print(f"LOGGING FINAL TEST RESULTS TO CSV: {args.test_results_file}")
log_test_results(args.test_results_file, gen_token_acc, gen_exact_acc)

print("\nDone.")
