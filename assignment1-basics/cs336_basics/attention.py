import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from torch import Tensor
from cs336_basics.nn_utils import softmax_fn
from cs336_basics.model import RotaryPositionalEmbedding

"""
Q = tensor([[[-1.5256e+00, -7.5023e-01, -6.5398e-01,  ..., -8.6959e-01,
          -3.3312e+00, -7.4787e-01],
         [-2....3.2825e-01],
         [-1.8315e-01,  2.0009e+00,  1.4760e-01,  ...,  1.2993e+00,
          -5.7777e-01,  1.4148e+00]]])
K = tensor([[[-1.0408,  0.9166, -1.3042,  ...,  0.7787, -0.7749, -0.1398],
         [ 1.1414, -0.6354, -1.4702,  ...,  1.8...43,  0.2841,  ..., -0.6094,  0.1403,  1.3990],
         [-0.1150,  0.0779,  1.3394,  ..., -0.3242, -0.8369,  0.8859]]])
V = tensor([[[-0.0766,  0.3599, -0.7820,  ..., -0.5296,  1.3544,  1.3778],
         [-0.0752, -0.4233,  0.4217,  ...,  2.3...59,  1.1937,  ...,  0.4603, -0.9189,  0.2698],
         [-0.7093, -1.5744,  1.5026,  ..., -1.7090,  0.2552,  0.8367]]])
mask = tensor([[[ True,  True, False, False,  True, False, False, False, False, False,
          False, False,  True, False, ...,  True, False, False, False,  True, False,  True, False,  True,
          False, False, False, False, False, False]]])
"""

"""scaled_dot_product_attention:
    Implement the scaled dot-product attention function.
    1. handle keys and queries of shape (batch_size, ..., seq_len, d_k) and
       values of shape (batch_size, ..., seq_len, d_v),
       where ... represents any number of other batch-like dimensions (if provided).
    2. return an output with the shape (batch_size, ..., seq_len, d_v).
       See Section 3.2 for a discussion on batch-like dimensions.
    3. support an optional user-provided boolean mask of shape (seq_len, seq_len).
    4. The attention probabilities of positions with a mask value of True should
        collectively sum to 1,
    5. the attention probabilities of positions with a mask value of False should
       be zero.
    To test your implementation against our provided tests, implement the test
    adapter at [adapters.run_scaled_dot_product_attention] .
    uv run pytest -k test_scaled_dot_product_attention tests
        - on third-order input tensors,
    uv run pytest -k test_4d_scaled_dot_product_attention tests
       - on fourth- order input tensors
"""
def scaled_dot_product_attention(
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None
    ) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # implement softmax(QK^T/sqrt(d_k))V
    sqrt_d_k = K.shape[-1]**0.5
    qks = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') / sqrt_d_k
    if mask is not None:
        qks = qks.masked_fill(~mask, float('-inf'))
    attention_weights = softmax_fn(qks, dim=-1)
    output = einsum(attention_weights, V, '... queries keys, ... keys values -> ... queries values')
    return output

"""
Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implementation should accept (at least) the following parameters:
d_model: int Dimensionality of the Transformer block inputs.
num_heads: int Number of heads to use in multi-head self-attention.
Following A. Vaswani et al. [8], set 𝑑𝑘=𝑑𝑣=𝑑modelℎ. To test your implementation against our provided tests, implement the test adapter at [adapters.run_multihead_self_attention] . Then, run uv run pytest -k test_multihead_self_attention to test your implementation.
"""
class MultiheadSelfAttention(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int,
                 device=None, dtype=None):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def set_weights(self, q_proj, k_proj, v_proj, o_proj):

        """
        Args:
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        output_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        """
        self.q_proj.weight.data = q_proj.clone().to(device=self.device, dtype=self.dtype)
        self.k_proj.weight.data = k_proj.clone().to(device=self.device, dtype=self.dtype)
        self.v_proj.weight.data = v_proj.clone().to(device=self.device, dtype=self.dtype)
        self.output_proj.weight.data = o_proj.clone().to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:

        """
        Args:
            x: in_features (Float[Tensor, "... sequence_length d_model"])
        Returns:
            Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of
            running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        """
        Q = self.q_proj(x) # (..., seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # split into hes
        """
        Example:
        If d_model = 768, heads = 12, and d_k = 64:
            x has shape (batch_size, seq_len, 768).
            Q = self.q_proj_wt(x) has shape (batch_size, seq_len, 768).
            rearrange transforms Q into shape (batch_size, 12, seq_len, 64).
        """
        Q = rearrange(Q, '... seq_len (heads d_k) -> ... heads seq_len d_k', heads=self.num_heads)
        K = rearrange(K, '... seq_len (heads d_k) -> ... heads seq_len d_k', heads=self.num_heads)
        V = rearrange(V, '... seq_len (heads d_v) -> ... heads seq_len d_v', heads=self.num_heads)

        # --- APPLY RoPE HERE if token_positions is provided ---
        if token_positions is not None and hasattr(self, 'rope_model'):
            # RoPE expects (..., seq_len, d_k)
            # So we apply RoPE to Q and K
            Q = self.rope_model(Q, token_positions)
            K = self.rope_model(K, token_positions)

        # Causal masking 1. Create a matrix of ones
        # diagonal=1 excludes the main diagonal (j == i)
        seq_len = x.shape[-2]
        # 2d to 4d change for broadcasting across batch and heads
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        attn_out = scaled_dot_product_attention(Q, K, V, causal_mask)
        attn_out = rearrange(attn_out,
                             '... heads seq_len d_v -> ... seq_len (heads d_v)',
                             heads=self.num_heads)
        final_tensor = self.output_proj(attn_out)
        return final_tensor


class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttention):

    def __init__(self, d_model: int, max_seq_len: int, theta: float, num_heads: int, device=None, dtype=None):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
        super().__init__(d_model, num_heads, device, dtype)
        self.max_seq_len = max_seq_len
        self.theta = theta

        d_k = d_model // num_heads
        self.rope_model = RotaryPositionalEmbedding(theta, d_k, max_seq_len)

    def forward(self, x, token_positions):
        attn_out = super().forward(x, token_positions)
        return attn_out


