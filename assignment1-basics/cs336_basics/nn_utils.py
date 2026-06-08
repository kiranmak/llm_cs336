import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
import einx
from torch import Tensor
from math import exp

def softmax_fn(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_val = in_features.max(dim=dim, keepdim=True)[0] # [0] because max returns a tuple.
    exps = torch.exp(in_features - max_val)
    return exps / exps.sum(dim=dim, keepdim=True)
