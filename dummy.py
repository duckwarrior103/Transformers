import torch

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
	device = "mps"

x = torch.randn(3, 3)
print(f"x is on {x.device}.")

print(f"Moving x from {x.device} to {device}.")

x = x.to(device)

print(f"x is on {x.device}.")