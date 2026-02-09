class Config:
    """
    Simple config container for spawning an encoder.
    Usage:
        cfg = Config(d_model=512, num_heads=8, d_ff=2048, num_layers=6)
    """

    def __init__(self, **kwargs):
        # core model sizes
        self.d_model = kwargs.pop("d_model", 512)
        # keep both names since some modules expect `embedding_dim` and others `d_model`
        self.embedding_dim = kwargs.pop("embedding_dim", self.d_model)

        self.num_heads = kwargs.pop("num_heads", 8)
        self.d_ff = kwargs.pop("d_ff", 2048)  # hidden dim in feed-forward
        self.num_layers = kwargs.pop("num_layers", 6)

        # runtime / tokenization
        self.vocab_size = kwargs.pop("vocab_size", 30522)
        self.max_seq_length = kwargs.pop("max_seq_length", 512)
        self.pad_token_id = kwargs.pop("pad_token_id", 0)

        # regularisation / misc
        self.dropout = kwargs.pop("dropout", 0.1)
        self.attn_dropout = kwargs.pop("attn_dropout", self.dropout)
        self.device = kwargs.pop("device", "cpu")

        # derived values / sanity checks
        assert self.embedding_dim == self.d_model, "embedding_dim must equal d_model"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads

    def to_dict(self):
        return self.__dict__.copy()

    def __repr__(self):
        return f"Config(d_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}, num_layers={self.num_layers})"

'''
Example usage:

model_config = Config(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    vocab_size=30522,
    max_seq_length=512,
    pad_token_id=0,
    dropout=0.1,
    attn_dropout=0.1,
    device="cpu"
)
print(model_config)
'''
