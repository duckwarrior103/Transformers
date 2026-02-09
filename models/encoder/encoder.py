import torch 
import torch.nn as nn
from models.model_config import Config
from models.encoder.block import Block
from models.components.positional_embedding.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

class Encoder(nn.Module):
    def __init__(self, model_config: Config):
        super(Encoder, self).__init__()
        self.config = model_config
        self.layers = nn.ModuleList([
            Block(model_config) for _ in range(model_config.num_layers)
        ])

        self.token_embedding = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.pos_embedding = SinusoidalPositionalEmbedding(model_config.d_model, model_config.max_seq_length)

        # map from seq_length, d_model to vocab_size for output logits
        self.output_projection = nn.Linear(model_config.d_model, model_config.vocab_size)

    # x shape: (batch_size, seq_length)
    def forward(self, x):
        # Embed tokens and add positional embeddings
        x = self.token_embedding(x) 

        # Add positional embeddings
        x = x + self.pos_embedding(x)

        # Pass through each layer of the encoder
        for layer in self.layers:
            x = layer(x)

        # map x to vocab size for output logits
        logits = self.output_projection(x) # (batch_size, seq_length, vocab_size)

        return logits