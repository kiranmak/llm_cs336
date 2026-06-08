import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from torch import Tensor
from cs336_basics.nn_utils import softmax_fn

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
    5. the attention probabilities of positions with a mask value of False should be zero.

    To test your implementation against our provided tests, you will need to implement the test
    adapter at [adapters.run_scaled_dot_product_attention] .
    uv run pytest -k test_scaled_dot_product_attention tests on third-order input tensors,
    uv run pytest -k test_4d_scaled_dot_product_attention tests on fourth- order input tensors
"""
def scaled_dot_product_attention(
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
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
