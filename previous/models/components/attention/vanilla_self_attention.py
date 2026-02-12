import torch
import torch.nn as nn
import numpy as np


class VanillaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.num_heads = config.num_heads

        self.W_k = nn.Linear(self.d_model, self.d_k)
        self.W_q = nn.Linear(self.d_model, self.d_k)
        self.W_v = nn.Linear(self.d_model, self.d_v)
        self.output_linear = nn.Linear(self.d_v, self.d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config.attn_dropout)

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        K = self.W_k(x)  # (batch_size, seq_length, d_k)
        Q = self.W_q(x)  # (batch_size, seq_length, d_k)
        V = self.W_v(x)  # (batch_size, seq_length, d_v)

        # Scaled dot-product attention
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.d_k)
        probs = self.softmax(attention_scores)
        probs = self.attn_dropout(probs)
        attention_output = torch.bmm(probs, V)

        # Pass through output linear layer
        output = self.output_linear(attention_output)
        return output