import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        
        # Ensure d_model is divisible by num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = self.d_model // self.num_heads
        
        # We project all heads at once for efficiency
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        
        self.output_linear = nn.Linear(self.d_model, self.d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config.attn_dropout)

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()

        # 1. Linear Projections
        # (batch_size, seq_length, d_model)
        K = self.W_k(x) 
        Q = self.W_q(x)
        V = self.W_v(x)

        # 2. Split into heads 
        # (batch_size, seq_length, num_heads, d_k) -> (batch_size, num_heads, seq_length, d_k)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention
        # Q (B, H, L, d_k) @ K.T (B, H, d_k, L) -> scores (B, H, L, L)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        probs = self.softmax(attention_scores)
        probs = self.attn_dropout(probs)
        
        # probs (B, H, L, L) @ V (B, H, L, d_k) -> output (B, H, L, d_k)
        attention_output = torch.matmul(probs, V)

        # 4. Concatenate heads
        # (B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

        # 5. Final output linear layer
        return self.output_linear(attention_output)