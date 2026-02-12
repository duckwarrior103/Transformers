import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's saved but not trained
        self.register_buffer('pe', pe)  # shape [max_len, d_model]

    def forward(self, X):
        """
        X: [batch, seq_len, d_model]
        """
        seq_len = X.size(1)
        # slice positional encoding to match seq_len
        pe_slice = self.pe[:seq_len, :].unsqueeze(0)  # shape [1, seq_len, d_model]
        return pe_slice  # broadcast across batch