import torch.nn as nn
from block import Block
from positional_embedding import SinusoidalPositionalEmbedding

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.pos_embed = SinusoidalPositionalEmbedding()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

    # idx refers to index of token in vocab
    def forward(self, idx):
        
        # Get positional embeddings 
        pe = self.pos_embed(idx)
        
        # Get token embeddings for all tokens
        x = self.token_embed(idx)

        # Combine token and positional embeddings
        x = x + pe

        # Pass through to all Decoder Blocks
        for block in self.blocks:
            x = block(x)

        # Prediction Layer
        logits = self.ln_f(x)
        return logits