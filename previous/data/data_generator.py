import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SortingDataset(Dataset):
    def __init__(self, num_samples=10000, seq_length=10, max_value=50):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.max_value = max_value
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequence
        sequence = torch.randint(0, self.max_value, (self.seq_length,))
        # Ground truth: sorted sequence
        target = torch.sort(sequence)[0]
        
        return sequence, target

