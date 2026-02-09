import torch.nn as nn
from models.components.attention.vanilla_self_attention import VanillaSelfAttention
from models.components.attention.neural_net.ffn import FFN


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = VanillaSelfAttention(config)
        self.mlp = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x