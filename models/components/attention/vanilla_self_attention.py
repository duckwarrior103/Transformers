import torch 
import torch.nn as nn

class VanillaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads

        assert self.head_dim * self.num_heads == self.d_model, "d_model must be divisible by num_heads"

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, 3 * d_model)
        qkv = qkv.view(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, 3 * head_dim)

        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each is (batch_size, num_heads, seq_length, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = self.attn_dropout(attn_weights)  # Apply dropout to attention weights

        # attn_output: (batch_size, num_heads, seq_length, head_dim)
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)

        # Move seq_length to dim=1 (batch, seq_length, num_heads, head_dim) for concatenation
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_length, num_heads, head_dim)

        # Concatenate heads
        attn_output = attn_output.view(batch_size, seq_length, self.d_model)  # (batch_size, seq_length, d_model)

        output = self.out_proj(attn_output)  # (batch_size, seq_length, d_model)

        return output