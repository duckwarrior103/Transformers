import torch  
from fla.models import TransformerConfig, TransformerForCausalLM  
  
# Select device explicitly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 1. Config  
config = TransformerConfig(  
    vocab_size=1001,  
    hidden_size=512,  
    num_hidden_layers=6,  
    num_heads=8,  
    max_position_embeddings=128,  
    pad_token_id=0,  
    eos_token_id=0,  
)  
model = TransformerForCausalLM(config).eval()

# Move model to GPU and convert to bfloat16
model = model.to(device=device, dtype=torch.bfloat16)
print(f"Model is on: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
  
# 2. Prepare integer input (batch=1, seq_len=10)  
input_ids = torch.tensor([[42, 7, 999, 3, 15, 0, 0, 0, 0, 0]], device=device)
print(f"Input is on: {input_ids.device}")
  
# 3. Inference: forward logits  
with torch.inference_mode():  
    outputs = model(input_ids=input_ids)  
logits = outputs.logits  # shape (1, 10, 1001)  
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
  
# 4. Inference: generate sorted sequence  
generated = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=False)
print(f"Generated shape: {generated.shape}")
print(f"Generated tokens: {generated}")