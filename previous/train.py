import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

from models.model_config import Config
from models.encoder.encoder import Encoder
from data.data_generator import SortingDataset


class TransformerTrainer:
    def __init__(self, config, learning_rate=1e-3):
        self.config = config
        self.model = Encoder(config).to(config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.epoch_history = []
        self.loss_history = []
        self.val_loss_history = []
        self.token_accuracy_history = []
        self.sequence_accuracy_history = []

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(self.config.device), targets.to(self.config.device)

            self.optimizer.zero_grad()
            logits = self.model(sequences)  # (batch_size, seq_length, vocab_size)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch [{batch_idx + 1}/{num_batches}] Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")

        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        """Evaluate on test set and compute token and sequence accuracy."""
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        correct_sequences = 0
        total_sequences = 0

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences, targets = sequences.to(self.config.device), targets.to(self.config.device)
                logits = self.model(sequences)

                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                
                # Token-wise accuracy
                correct_tokens += (predictions == targets).sum().item()
                total_tokens += targets.numel()
                
                # Sequence-wise accuracy (entire sequence must be correct)
                correct_sequences += (predictions == targets).all(dim=1).sum().item()
                total_sequences += targets.size(0)

        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
        
        return total_loss / len(test_loader), token_accuracy, sequence_accuracy

    def train_model(self, train_loader, val_loader, epochs=100, verbose=True):
        """Full training loop."""
        self.epoch_history.clear()
        self.loss_history.clear()
        self.val_loss_history.clear()
        self.token_accuracy_history.clear()
        self.sequence_accuracy_history.clear()

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, token_accuracy, sequence_accuracy = self.evaluate(val_loader)

            self.epoch_history.append(epoch)
            self.loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.token_accuracy_history.append(token_accuracy)
            self.sequence_accuracy_history.append(sequence_accuracy)

            if verbose and (epoch + 1) % 1 == 0:  # Print every epoch
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Token Acc: {token_accuracy:.4f} | Seq Acc: {sequence_accuracy:.4f}\n")

    def plot_loss(self):
        """Plot training vs validation loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Train Loss', marker='o')
        plt.plot(self.val_loss_history, label='Val Loss', marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_accuracy(self):
        """Plot token and sequence accuracy over epochs."""
        plt.figure(figsize=(12, 5))
        
        # Token accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.token_accuracy_history, marker='o', color='blue', linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Token-wise Accuracy over Epochs")
        plt.grid(True)
        
        # Sequence accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.sequence_accuracy_history, marker='s', color='green', linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Sequence-wise Accuracy over Epochs")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))
        print(f"Model loaded from {path}")


def batch_test_model(trainer, test_loader, test_size=1000):
    """Test model on full test set and compute accuracy."""
    trainer.model.eval()
    correct = 0
    total = 0
    row_correct = 0
    num_rows = 0

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(trainer.config.device), targets.to(trainer.config.device)
            logits = trainer.model(sequences)
            predictions = logits.argmax(dim=-1)

            # Element-wise accuracy
            correct += (predictions == targets).sum().item()
            total += targets.numel()

            # Row-wise accuracy
            row_correct += (predictions == targets).all(dim=1).sum().item()
            num_rows += targets.size(0)

    element_accuracy = correct / total * 100 if total > 0 else 0
    row_accuracy = row_correct / num_rows * 100 if num_rows > 0 else 0

    print(f"Element-wise Accuracy: {element_accuracy:.2f}%")
    print(f"Row-wise Accuracy: {row_accuracy:.2f}%")

    return row_accuracy, element_accuracy


def sample_inference(trainer, num_samples=5, seq_length=3):
    """Run inference on random sorting examples and display predictions."""
    trainer.model.eval()
    
    # Generate small test dataset
    test_dataset = SortingDataset(num_samples=num_samples, seq_length=seq_length, max_value=50)
    
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80 + "\n")
    
    correct_count = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            sequence, target = test_dataset[i]
            sequence_input = sequence.unsqueeze(0).to(trainer.config.device)  # (1, seq_length)
            target = target.to(trainer.config.device)  # Move target to same device
            
            logits = trainer.model(sequence_input)  # (1, seq_length, vocab_size)
            predictions = logits.argmax(dim=-1).squeeze(0)  # (seq_length,)
            
            # Check if entire sequence is correct
            is_correct = torch.all(predictions == target).item()
            correct_count += int(is_correct)
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            
            print(f"Example {i + 1}:")
            print(f"  Input:      {sequence.tolist()}")
            print(f"  Predicted:  {predictions.cpu().tolist()}")
            print(f"  Expected:   {target.cpu().tolist()}")
            print(f"  Status:     {status}\n")
    
    accuracy = (correct_count / num_samples) * 100
    print("="*80)
    print(f"Sequence Accuracy: {accuracy:.2f}% ({correct_count}/{num_samples} correct)")
    print("="*80 + "\n")


def main():
    # Configuration parameters
    seq_length = 8  # Default sequence length for training
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "infer":
            # For inference: python train.py infer [infer_seq_length] [model_seq_length]
            infer_seq_length = 5  # Default inference sequence length
            model_seq_length = 5  # Default model sequence length
            
            if len(sys.argv) > 2:
                try:
                    infer_seq_length = int(sys.argv[2])
                except ValueError:
                    print(f"Invalid inference sequence length: {sys.argv[2]}. Using default: {infer_seq_length}")
            
            if len(sys.argv) > 3:
                try:
                    model_seq_length = int(sys.argv[3])
                except ValueError:
                    print(f"Invalid model sequence length: {sys.argv[3]}. Using default: {model_seq_length}")
            
            config = Config(
                d_model=128,
                num_heads=4,
                d_k=256,
                d_v=256,
                d_ff=256,
                num_layers=4,
                vocab_size=50,
                max_seq_length=512,
                pad_token_id=0,
                dropout=0.1,
                attn_dropout=0.1,
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            
            trainer = TransformerTrainer(config)
            save_dir = Path("saved_models")
            model_path = save_dir / f"transformer_sorting_model_seq{model_seq_length}.pt"
            
            if model_path.exists():
                trainer.load_model(model_path)
                print(f"Running inference with infer_seq_length={infer_seq_length} using model_seq{model_seq_length}...")
                sample_inference(trainer, num_samples=5, seq_length=infer_seq_length)
            else:
                print(f"Model not found at {model_path}")
                print(f"Available models in saved_models/:")
                save_dir.mkdir(exist_ok=True)
                models = list(save_dir.glob("transformer_sorting_model_seq*.pt"))
                if models:
                    for model in sorted(models):
                        print(f"  - {model.name}")
                else:
                    print("  No models found. Train first with: python train.py [seq_length]")
                print(f"\nUsage: python train.py infer [infer_seq_length] [model_seq_length]")
                print(f"Example: python train.py infer 3 5")
            return
        
        # For training: python train.py [seq_length]
        if len(sys.argv) > 1:
            try:
                seq_length = int(sys.argv[1])
            except ValueError:
                print(f"Invalid sequence length: {sys.argv[1]}. Using default: {seq_length}")
    
    config = Config(
        d_model=128,
        num_heads=4,
        d_k=256,
        d_v=256,
        d_ff=256,
        num_layers=4,
        vocab_size=50,
        max_seq_length=512,
        pad_token_id=0,
        dropout=0.1,
        attn_dropout=0.1,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Dataset
    train_dataset = SortingDataset(num_samples=50000, seq_length=seq_length, max_value=50)
    val_dataset = SortingDataset(num_samples=10000, seq_length=seq_length, max_value=50)
    test_dataset = SortingDataset(num_samples=10000, seq_length=seq_length, max_value=50)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / f"transformer_sorting_model_seq{seq_length}.pt"

    # Train
    print(f"Starting training with sequence length={seq_length}...")
    trainer = TransformerTrainer(config, learning_rate=5e-4)
    trainer.train_model(train_loader, val_loader, epochs=5, verbose=True)

    # Test
    print("\nTesting on test set...")
    batch_test_model(trainer, test_loader)

    # Save
    trainer.save_model(model_path)

    # Plot
    trainer.plot_loss()
    trainer.plot_accuracy()


if __name__ == "__main__":
    main()