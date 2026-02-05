import torch.nn as nn
from attention import CausalSelfAttention
from neural_net import MLP

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Layer normalisation 
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        # Multi-headed, causal self-attention layer
        self.attn = CausalSelfAttention(config)

        # Multi-layer Perceptron, a two layer fully-connected neural net
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x