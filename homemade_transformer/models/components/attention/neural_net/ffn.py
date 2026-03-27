import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(config.d_model, config.d_ff)
        self.layer2 = nn.Linear(config.d_ff, config.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x