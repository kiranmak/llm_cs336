import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from torch import Tensor
from cs336_basics.model import ModelEmbeddings, ModelSwiGLU, LinearTransform
from cs336_basics.model import ModelRMS
from cs336_basics.attention import MultiheadSelfAttentionWithRoPE

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int,
                 theta: int, num_heads: int, d_ff: int,
                 dtype=None, device=None):
        super().__init__()
        # Fallback to CPU or let PyTorch handle it if None is passed
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = ModelRMS(d_model=d_model,
                            eps=1e-5,
                            device=self.device,
                            dtype=self.dtype)

        self.ln2 = ModelRMS(d_model=d_model,
                            eps=1e-5,
                            device=self.device,
                            dtype=self.dtype)

        self.attn = MultiheadSelfAttentionWithRoPE(d_model, max_seq_len,
                                                   theta, num_heads,
                                                   dtype=self.dtype,
                                                   device=self.device)
        self.ffn = ModelSwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]):
        out_norm_1 = self.ln1(x)
        token_positions = torch.arange(0, x.size(1), device=x.device)
        token_positions = rearrange(token_positions, "s -> 1 s")

        attention = self.attn(out_norm_1, token_positions)
        x = x + attention

        out_norm_2 = self.ln2(x)
        out_ff = self.ffn(out_norm_2)
        return x + out_ff

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int,
                 theta: int, num_heads: int, d_ff: int, num_layers: int,
                 device=None,
                 dtype=None):
        super().__init__()

        # Fallback to CPU or let PyTorch handle it if None is passed
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.token_embeddings = ModelEmbeddings(vocab_size, d_model,
                                                self.device,
                                                self.dtype)
        self.ln_final = ModelRMS(d_model=d_model,
                            eps=1e-5,
                            device=self.device,
                            dtype=self.dtype)
        self.layers = nn.ModuleList([TransformerBlock(
                                                    d_model, max_seq_len,
                                                    theta, num_heads, d_ff,
                                                    dtype=self.dtype,
                                                    device=self.device)
                                                for _ in range(num_layers)])
        self.lm_head = LinearTransform(d_model, vocab_size,
                                       dtype=self.dtype,
                                       device=self.device)

    def forward(self, x: Float[Tensor, " ... sequence_length"]):
        embeddings = self.token_embeddings(x)
        for layer in self.layers:
            embeddings = layer(embeddings)
        final_norm = self.ln_final(embeddings)
        logits = self.lm_head(final_norm)
        return logits

