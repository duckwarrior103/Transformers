import torch
import torch.nn as nn
from models.model_config import Config
from models.encoder.block import Block
from models.components.positional_embedding.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


class Encoder(nn.Module):
    def __init__(self, model_config: Config):
        super().__init__()
        self.config = model_config
        self.embed = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.pos_encoding = SinusoidalPositionalEmbedding(model_config.d_model, model_config.max_seq_length)
        self.layers = nn.ModuleList([Block(model_config) for _ in range(model_config.num_layers)])
        self.unembed = nn.Linear(model_config.d_model, model_config.vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embed(x)  # (batch_size, seq_length, d_model)
        x = x + self.pos_encoding(x)  # Add positional embeddings

        # Pass through each transformer block
        for layer in self.layers:
            x = layer(x)

        # Project to vocabulary size
        logits = self.unembed(x)  # (batch_size, seq_length, vocab_size)
        return logits

    @torch.no_grad()
    def predict(self, x):
        """Get predictions using argmax."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds

    @torch.no_grad()
    def predict_top_k(self, x, k=3):
        """Get top-k predictions."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        return top_k_indices, top_k_probs