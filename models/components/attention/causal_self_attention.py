import torch.nn as nn
import torch

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = config.num_heads
        self.head_dim = config.n_embd // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Scale from X n * n_embd to n * 3*n_embd for q, k, v
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # Causal mask to ensure that attention is only applied to previous tokens
        self.register_buffer("mask", torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)