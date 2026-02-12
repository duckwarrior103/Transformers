import torch  
from fla.models import TransformerConfig, TransformerForCausalLM  
  
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
model = TransformerForCausalLM(config).eval()  # or .cuda()/.to(dtype)  
  
# 2. Prepare integer input (batch=1, seq_len=10)  
input_ids = torch.tensor([[42, 7, 999, 3, 15, 0, 0, 0, 0, 0]], device=model.device)  
  
# 3. Inference: forward logits  
with torch.inference_mode():  
    outputs = model(input_ids=input_ids)  
logits = outputs.logits  # shape (1, 10, 1001)  
  
# 4. Inference: generate sorted sequence  
generated = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=False)  
# generated tokens are integer IDs; map back to numbers if needed