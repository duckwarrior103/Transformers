import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fla.models import TransformerConfig, TransformerForCausalLM
from fla.models import GLAConfig, GLAForCausalLM
from fla.models import LinearAttentionConfig, LinearAttentionForCausalLM
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import random

# ============================================================================
# Device
# ============================================================================

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================================
# Configuration
# ============================================================================

SEQUENCE_LENGTH = 1024
MAX_VALUE = 999

# Decreased from 500k to 50k for faster experimentation; can increase later if needed
#First Epoch 22.66% Token, 0.00% Exact on 50k examples
# With 500k examples, first epoch is Val Exact Accuracy: 39.88% Val Token Accuracy: 99.64%

TRAIN_EXAMPLES = 50_000
VAL_EXAMPLES = 20_000

BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

EOS_TOKEN = 1000
VOCAB_SIZE = 1001  # 0–999 + EOS

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
        numbers = torch.randint(0, self.max_value + 1, (self.seq_length,))
        sorted_numbers = torch.sort(numbers).values

        input_seq = torch.cat([numbers, torch.tensor([EOS_TOKEN])])
        target_seq = torch.cat([sorted_numbers, torch.tensor([EOS_TOKEN])])

        full_seq = torch.cat([input_seq, target_seq])

        input_ids = full_seq.clone()
        labels = full_seq.clone()

        # Mask input portion (unsorted numbers + EOS)
        labels[:len(input_seq)] = -100

        return input_ids, labels


train_dataset = SortingDataset(TRAIN_EXAMPLES, SEQUENCE_LENGTH, MAX_VALUE)
val_dataset = SortingDataset(VAL_EXAMPLES, SEQUENCE_LENGTH, MAX_VALUE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============================================================================
# Model
# ============================================================================

# config = LinearAttentionConfig(
#     vocab_size=VOCAB_SIZE,
#     hidden_size=512,
#     num_hidden_layers=6,
#     num_heads=8,
#     max_position_embeddings=SEQUENCE_LENGTH,
#     pad_token_id=EOS_TOKEN,
#     eos_token_id=EOS_TOKEN,
#     attn_mode="fused_chunk",  # Use fused_chunk for attention
#     expand_k=1.0,
#     expand_v=1.0,
#     feature_map="elementwise_product",
#     fuse_norm=True,
#     fuse_swiglu=True,
#     fuse_linear_cross_entropy=False,
# )

# model = LinearAttentionForCausalLM(config)

config = GLAConfig(  
    vocab_size=VOCAB_SIZE,  
    hidden_size=512,  
    num_hidden_layers=6,  
    num_heads=8,  
    max_position_embeddings=SEQUENCE_LENGTH,  
    pad_token_id=EOS_TOKEN,  
    eos_token_id=EOS_TOKEN,  
    # Fastest settings  
    attn_mode="chunk",  # fastest for inference; use "fused_chunk" for training  
    expand_k=1.0,  
    expand_v=1.0,  
    use_output_gate=True,  
    fuse_norm=True,               # enable fused RMSNorm  
    fuse_swiglu=True,             # enable fused SwiGLU  
    fuse_linear_cross_entropy=False,  # memory-efficient loss  
)  
model = GLAForCausalLM(config)

model = model.to(device=device, dtype=torch.bfloat16)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.05 * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ============================================================================
# Training
# ============================================================================

def train_epoch():
    model.train()
    total_loss = 0

    for input_ids, labels in tqdm(train_loader, desc="Training"):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Extract sorted output predictions and targets
            start_idx = SEQUENCE_LENGTH  # position 10
            end_idx = start_idx + SEQUENCE_LENGTH  # position 20

            predicted_sorted = predictions[:, start_idx:end_idx]
            expected_sorted = input_ids[:, start_idx + 1:end_idx + 1]

            # Token accuracy: compare token-by-token
            correct_tokens += (predicted_sorted == expected_sorted).sum().item()
            total_tokens += predicted_sorted.numel()

            # Sequence accuracy: compare entire sorted sequences
            for i in range(predicted_sorted.size(0)):
                if torch.equal(predicted_sorted[i], expected_sorted[i]):
                    correct_sequences += 1
                total_sequences += 1

    return (
        total_loss / len(val_loader),
        correct_sequences / total_sequences,
        correct_tokens / total_tokens
    )

# ============================================================================
# Training Loop
# ============================================================================

best_accuracy = 0

def display_sample():
    model.eval()
    input_ids, labels = next(iter(val_loader))
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

    # Extract a random sample from the batch
    idx = random.randint(0, input_ids.size(0) - 1)
    input_sample = input_ids[idx].tolist()
    predicted_sample = predictions[idx].tolist()
    expected_sample = labels[idx].tolist()

    # Extract the sorted portion for comparison
    start_idx = SEQUENCE_LENGTH
    end_idx = start_idx + SEQUENCE_LENGTH

    # Remove EOS_TOKEN from the predicted sequence
    predicted_sorted = [token for token in predicted_sample[start_idx:end_idx] if token != EOS_TOKEN]
    expected_sorted = input_sample[start_idx + 1:end_idx + 1]

    print("\nSample Validation Result:")
    print(f"Input:     {input_sample[:SEQUENCE_LENGTH]}")
    print(f"Predicted: {predicted_sorted}")
    print(f"Expected:  {expected_sorted}")
    print("\n")


for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_epoch()
    val_loss, val_seq_acc, val_token_acc = evaluate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Exact Accuracy: {val_seq_acc*100:.2f}%")
    print(f"Val Token Accuracy: {val_token_acc*100:.2f}%")

    if val_seq_acc > best_accuracy:
        best_accuracy = val_seq_acc
        torch.save(model.state_dict(), "best_model.pt")
        print("✓ New best model saved.")

    # Display a sample validation result
    display_sample()

# ============================================================================
# FINAL TESTING SECTION
# ============================================================================

def run_single_example(numbers):
    model.eval()

    input_ids = torch.tensor([numbers + [EOS_TOKEN]], device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=SEQUENCE_LENGTH + 1,
                do_sample=False,
                pad_token_id=EOS_TOKEN,
                eos_token_id=EOS_TOKEN  # Stop generating after EOS_TOKEN
            )

    # The output is: [input_numbers + EOS] + [sorted_numbers + EOS]
    # So, sorted output starts at position len(numbers) + 1
    start = len(numbers) + 1
    predicted = output[0, start:].tolist()

    # Remove any tokens after EOS_TOKEN
    if EOS_TOKEN in predicted:
        predicted = predicted[:predicted.index(EOS_TOKEN)]

    expected = sorted(numbers)

    token_correct = sum(p == t for p, t in zip(predicted, expected))
    token_acc = token_correct / SEQUENCE_LENGTH
    exact = predicted == expected

    print(f"\nInput:     {numbers}")
    print(f"Expected:  {expected}")
    print(f"Predicted: {predicted}")
    print(f"Token Acc: {token_acc*100:.1f}%")
    print(f"{'✓ EXACT MATCH' if exact else '✗ NOT EXACT'}")


print("\n" + "="*70)
print("TESTING ON RANDOM NEW SAMPLES")
print("="*70)

for _ in range(5):
    nums = [random.randint(0, MAX_VALUE) for _ in range(SEQUENCE_LENGTH)]
    run_single_example(nums)

print("\nDone.")
