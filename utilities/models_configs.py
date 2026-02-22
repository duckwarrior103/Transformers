from fla.models import GLAConfig
from fla.models import LinearAttentionConfig
from fla.models import RetNetConfig
from fla.models import TransformerConfig

# Importing model classes
from fla.models import TransformerForCausalLM  
from fla.models import GLAForCausalLM  
from fla.models import RetNetForCausalLM  
from fla.models import LinearAttentionForCausalLM

# Config for standard attention model (e.g. FlashAttention2)

def get_standard_config(vocab_size, seq_length):  
    return TransformerConfig(  
        vocab_size=vocab_size,  
        hidden_size=512,  
        num_hidden_layers=6,  
        num_heads=8,  
        max_position_embeddings=seq_length,  
        pad_token_id=vocab_size - 1,  
        eos_token_id=vocab_size - 1,  
        qkv_bias=False,  
        qk_norm=False,  
        window_size=None,  
        rope_theta=10000.0,  
        fuse_norm=True,  
        fuse_swiglu=True,  
        fuse_cross_entropy=True,  
        fuse_linear_cross_entropy=False,  
    )

# Config for linear attention model (e.g. FlashAttention2)
def get_linear_attention_config(vocab_size, seq_length):  
    return LinearAttentionConfig(  
        vocab_size=vocab_size,  
        hidden_size=512,  
        num_hidden_layers=6,  
        num_heads=8,  
        max_position_embeddings=seq_length,  
        pad_token_id=vocab_size - 1,  
        eos_token_id=vocab_size - 1,  
        attn_mode="chunk",  
        expand_k=1.0,  
        expand_v=1.0,  
        feature_map="elementwise_product",  
        fuse_norm=True,  
        fuse_swiglu=True,  
        fuse_cross_entropy=True,  
        fuse_linear_cross_entropy=False,  
    )

def get_gla_config(vocab_size, seq_length):
    return GLAConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_heads=8,
        max_position_embeddings=seq_length,
        pad_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        attn_mode="chunk",
        expand_k=1.0,
        expand_v=1.0,
        use_output_gate=True,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_linear_cross_entropy=False,
    )

def get_retnet_config(vocab_size, seq_length):  

    return RetNetConfig(  
        vocab_size=vocab_size,  
        hidden_size=512,  
        num_hidden_layers=6,  
        num_heads=8,  
        max_position_embeddings=seq_length,  
        pad_token_id=vocab_size - 1,  
        eos_token_id=vocab_size - 1,  
        attn_mode="chunk",  
        expand_k=1.0,  
        expand_v=1.0,  
        use_output_gate=True,  
        fuse_norm=True,  
        fuse_swiglu=True,  
        fuse_cross_entropy=True,  
        fuse_linear_cross_entropy=False,  
    )


def get_models_creator_dict():
    return {
        "standard": (get_standard_config, TransformerForCausalLM),
        "linear_attention": (get_linear_attention_config, LinearAttentionForCausalLM),
        "gla": (get_gla_config, GLAForCausalLM),
        "retnet": (get_retnet_config, RetNetForCausalLM ),
    }

