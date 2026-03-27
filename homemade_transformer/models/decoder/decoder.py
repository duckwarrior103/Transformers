import torch.nn as nn
from block import Block
from positional_embedding import SinusoidalPositionalEmbedding

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.pos_embed = SinusoidalPositionalEmbedding()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, idx):
        pe = self.pos_embed(idx)
        x = self.token_embed(idx)
        x = x + pe

        for block in self.blocks:
            x = block(x)

        logits = self.ln_f(x)
        return logits