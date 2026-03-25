from fla.models import GLAConfig
from fla.models import LinearAttentionConfig
from fla.models import RetNetConfig
from fla.models import TransformerConfig
from fla.models import DeltaNetConfig
from fla.models import GatedDeltaNetConfig

# Importing model classes
from fla.models import TransformerForCausalLM
from fla.models import GLAForCausalLM
from fla.models import RetNetForCausalLM
from fla.models import LinearAttentionForCausalLM as _LinearAttentionForCausalLM
from fla.layers.linear_attn import LinearAttention


class LinearAttentionForCausalLM(_LinearAttentionForCausalLM):
    """Switch to fused_recurrent during generate() (handles seq_len=1 token-by-token
    decoding), then restore chunk mode afterwards for training efficiency."""

    def generate(self, *args, **kwargs):
        layers = [m for m in self.modules() if isinstance(m, LinearAttention)]
        original_modes = [m.mode for m in layers]
        for m in layers:
            m.mode = 'fused_recurrent'
        try:
            return super().generate(*args, **kwargs)
        finally:
            for m, mode in zip(layers, original_modes):
                m.mode = mode
from fla.models import DeltaNetForCausalLM
from fla.models import GatedDeltaNetForCausalLM

# Config for standard attention model (e.g. FlashAttention2)

def get_standard_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):  
    return TransformerConfig(  
        vocab_size=vocab_size,  
        hidden_size=hidden_size,  
        num_hidden_layers=num_hidden_layers,  
        num_heads=num_heads,  
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
def get_linear_attention_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return LinearAttentionConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,  
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

def get_gla_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return GLAConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
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

def get_retnet_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return RetNetConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,  
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


def get_deltanet_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return DeltaNetConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        max_position_embeddings=seq_length,
        pad_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        attn_mode="chunk",
        expand_k=1.0,
        expand_v=1.0,
        use_short_conv=True,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
        fuse_linear_cross_entropy=False,
    )


def get_gated_deltanet_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return GatedDeltaNetConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        max_position_embeddings=seq_length,
        pad_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        attn_mode="chunk",
        use_short_conv=True,
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
        "retnet": (get_retnet_config, RetNetForCausalLM),
        "deltanet": (get_deltanet_config, DeltaNetForCausalLM),
        "gated_deltanet": (get_gated_deltanet_config, GatedDeltaNetForCausalLM),
    }

