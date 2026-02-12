import torch
import torch.nn.functional as F

from models.model_config import Config
from models.encoder.encoder import Encoder

def predict(model, sequences):
    """
    Get predicted tokens from the model using greedy decoding.
    
    Args:
        model: Encoder model
        sequences: (batch_size, seq_length) - input token indices
    
    Returns:
        predictions: (batch_size, seq_length) - predicted token indices
    """
    model.eval()
    with torch.no_grad():
        logits = model(sequences)
        predictions = logits.argmax(dim=-1)
    return predictions


def get_probabilities(model, sequences):
    """
    Get probability distribution over vocabulary for each position.
    
    Args:
        model: Encoder model
        sequences: (batch_size, seq_length) - input token indices
    
    Returns:
        probabilities: (batch_size, seq_length, vocab_size)
    """
    model.eval()
    with torch.no_grad():
        logits = model(sequences)
        probabilities = F.softmax(logits, dim=-1)
    return probabilities


def predict_top_k(model, sequences, k=5):
    """
    Get top-k predictions for each position.
    
    Args:
        model: Encoder model
        sequences: (batch_size, seq_length)
        k: number of top predictions to return
    
    Returns:
        top_k_tokens: (batch_size, seq_length, k)
        top_k_probs: (batch_size, seq_length, k)
    """
    model.eval()
    with torch.no_grad():
        logits = model(sequences)
        probabilities = F.softmax(logits, dim=-1)
        top_k_probs, top_k_tokens = torch.topk(probabilities, k, dim=-1)
    return top_k_tokens, top_k_probs


def __main__():
    # Example usage
    config = Config(
        d_model=128,
        num_heads=8,
        d_ff=256,
        num_layers=6,
        vocab_size=50,
        max_seq_length=512,
        pad_token_id=0,
        dropout=0.1,
        attn_dropout=0.1,
        device="cpu"
    )
    model = Encoder(config)
    
    # Dummy input (batch_size=2, seq_length=10)
    sequences = torch.randint(0, config.vocab_size, (2, 10))
    print(sequences)

    predictions = predict(model, sequences)
    probabilities = get_probabilities(model, sequences)
    top_k_tokens, top_k_probs = predict_top_k(model, sequences, k=3)
    
    print("Predictions:\n", predictions)
    print("Probabilities:\n", probabilities)
    print("Top-k Tokens:\n", top_k_tokens)
    print("Top-k Probabilities:\n", top_k_probs)

if __name__ == "__main__":
    __main__()