from fla.models import GLAConfig
from fla.models import LinearAttentionConfig
from fla.models import RetNetConfig
from fla.models import TransformerConfig
from fla.models import DeltaNetConfig
from fla.models import GatedDeltaNetConfig
from utilities.gead import GEADConfig, GEADForCausalLM
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

def get_standard_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
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
def get_linear_attention_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return LinearAttentionConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
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
        intermediate_size=4 * hidden_size,
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
        intermediate_size=4 * hidden_size,
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
        intermediate_size=4 * hidden_size,
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


def get_gated_deltanet_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2, expand_k=1.0, head_dim=None):
    if head_dim is None:
        head_dim = hidden_size // num_heads
    return GatedDeltaNetConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_length,
        pad_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        attn_mode="chunk",
        expand_k=expand_k,
        expand_v=1.0,
        use_short_conv=True,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
        fuse_linear_cross_entropy=False,
    )


def get_gead_config(vocab_size, seq_length, hidden_size=128, num_hidden_layers=2, num_heads=2):
    return GEADConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        max_position_embeddings=seq_length,
        pad_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        attn_mode="chunk",
        expand_k=1.0,
        expand_v=1.0,
        use_short_conv=True,
        use_elm=True,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
        fuse_linear_cross_entropy=False,
    )


def _make_gead_elm_config_fn(head_dim: int, elm_orthogonal: bool = False,
                              num_hidden_layers: int = 2, num_heads: int = 2):
    """Factory: GEAD with square W1. hidden_size = num_heads * head_dim.
    elm_dim = head_dim (square W1)."""
    elm_dim = head_dim
    hidden_size = num_heads * head_dim
    def get_config(vocab_size, seq_length, _hidden_size=None, _num_layers=None, _num_heads=None, **_kwargs):
        return GEADConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_position_embeddings=seq_length,
            pad_token_id=vocab_size - 1,
            eos_token_id=vocab_size - 1,
            attn_mode="chunk",
            expand_v=1.0,
            use_short_conv=True,
            use_elm=True,
            elm_dim=elm_dim,
            elm_orthogonal=elm_orthogonal,
            fuse_norm=True,
            fuse_swiglu=True,
            fuse_cross_entropy=True,
            fuse_linear_cross_entropy=False,
        )
    return get_config


def _make_gdn_ek_config_fn(head_dim: int,
                            num_hidden_layers: int = 2, num_heads: int = 2):
    """Factory: GatedDeltaNet. hidden_size = num_heads * head_dim."""
    hidden_size = num_heads * head_dim
    def get_config(vocab_size, seq_length, _hidden_size=None, _num_layers=None, _num_heads=None, **_kwargs):
        return GatedDeltaNetConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_position_embeddings=seq_length,
            pad_token_id=vocab_size - 1,
            eos_token_id=vocab_size - 1,
            attn_mode="chunk",
            expand_v=1.0,
            use_short_conv=True,
            fuse_norm=True,
            fuse_swiglu=True,
            fuse_cross_entropy=True,
            fuse_linear_cross_entropy=False,
        )
    return get_config


def get_models_creator_dict():
    d = {
        "standard": (get_standard_config, TransformerForCausalLM),
        "linear_attention": (get_linear_attention_config, LinearAttentionForCausalLM),
        "gla": (get_gla_config, GLAForCausalLM),
        "retnet": (get_retnet_config, RetNetForCausalLM),
        "deltanet": (get_deltanet_config, DeltaNetForCausalLM),
        "gated_deltanet": (get_gated_deltanet_config, GatedDeltaNetForCausalLM),
        "gead": (get_gead_config, GEADForCausalLM),
    }
    for head_dim in [64, 128, 256]:
        d[f"gead_elm{head_dim}"]      = (_make_gead_elm_config_fn(head_dim, elm_orthogonal=False), GEADForCausalLM)
        d[f"gead_elm{head_dim}_orth"] = (_make_gead_elm_config_fn(head_dim, elm_orthogonal=True),  GEADForCausalLM)
        d[f"gdn_hd{head_dim}"]       = (_make_gdn_ek_config_fn(head_dim), GatedDeltaNetForCausalLM)
    return d


def find_iso_hidden_size(model_type, target_params, vocab_size, seq_length,
                         num_hidden_layers, num_heads):
    """Return largest hidden_size (multiple of num_heads) whose trainable
    param count does not exceed target_params."""
    config_fn, model_cls = get_models_creator_dict()[model_type]

    def count_params(h):
        cfg = config_fn(vocab_size, seq_length, h, num_hidden_layers, num_heads)
        m = model_cls(cfg)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        del m
        return n

    lo, hi, best = num_heads, 2048, num_heads
    while lo <= hi:
        mid = ((lo + hi) // (2 * num_heads)) * num_heads
        if mid == 0:
            break
        if count_params(mid) <= target_params:
            best = mid
            lo = mid + num_heads
        else:
            hi = mid - num_heads
    return best

