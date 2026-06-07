import torch
import torch.nn as nn
#import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
import einx
from torch import Tensor
import numpy as np

class LinearTransform(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 weights: Float[Tensor, " d_out d_in"] | None = None,
                 device=None,
                 dtype=None):
        """
        in_dim: int final dimension of the input
        out_dim: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super(LinearTransform, self).__init__()
        # construct and store your parameter as 𝑊 (not 𝑊 ⊤), putting it in an nn.Parameter
        if weights is not None:
            self.weight = nn.Parameter(
                        weights.clone().to(device=device, dtype=dtype),
                        requires_grad=True
                    )
        else:
            w = torch.empty(out_dim, in_dim, device=device, dtype=dtype)
            nn.init.trunc_normal_(w)
            self.weight = nn.Parameter(w, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the
        x = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return x

class ModelEmbeddings(nn.Module):

    def __init__(self,
        num_embeddings: int,
        embedding_dim: int,
        device=None, dtype=None):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., 𝑑model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super(ModelEmbeddings, self).__init__()
        self.device = device
        self.dtype  = dtype

        w = torch.empty(num_embeddings, embedding_dim,
                        device=device, dtype=dtype)
        nn.init.trunc_normal_(w)
        self.weight = nn.Parameter(w, requires_grad=True)

    def set_weights(self, wts: Float[Tensor, " num_embeddings  embedding_dim"]):
        self.weight = nn.Parameter(wts.clone().to(device=self.device, dtype=self.dtype),requires_grad=True)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weight[token_ids]

class ModelRMS(nn.Module):

    def __init__(self, d_model: int,
                 eps: float = 1e-5,
                 device=None, dtype=None):
        """
        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super(ModelRMS, self).__init__()
        self.device = device
        self.dtype  = dtype
        self.d_model = d_model # model dimensionality
        self.eps = eps # epsilon value for numerical stability

        w = torch.empty(d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(w)
        self.weight = nn.Parameter(w, requires_grad=True)

    def set_weights(self, wts: Float[Tensor, " d_model"]):
        """ weights (Float[Tensor, "d_model"]): RMSNorm weights."""

        self.weight = nn.Parameter(wts.clone().to(device=self.device,
                                                  dtype=self.dtype),
                                   requires_grad=True)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Process an input tensor of shape(batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        Note: your input to torch.float32 before performing the normalization
        """
        in_dtype = x.dtype
        d = x.shape[-1] # this is same as self.d_model
        x = x.to(torch.float32)
        rms = torch.sqrt((1/d * einsum(x, x, '... d, ... d -> ...')) + self.eps)
        g = self.weight
        rms_norm = einsum(x, g, '... d, d -> ... d') / rms.unsqueeze(-1)

        return rms_norm.to(in_dtype)

class ModelSwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        """
        super(ModelSwiGLU, self).__init__()
        self.d_model = d_model # model dimensionality
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # nn.Linear stores weight as (out, in), so shapes match:
        #   w1, w3: (d_ff, d_model)  → Linear(d_model, d_ff)
        #   w2:     (d_model, d_ff)  → Linear(d_ff, d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)

    def set_weights(self,
                    w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor):

        """
         Args:
            w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
            w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
            w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        """
        self.w1.weight.data = w1_weight.clone().to(device=self.device, dtype=self.dtype)
        self.w2.weight.data = w2_weight.clone().to(device=self.device, dtype=self.dtype)
        self.w3.weight.data = w3_weight.clone().to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Process an input tensor of shape(batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        Note: your input to torch.float32 before performing the normalization
        """

        w1 = self.w1.weight
        w2 = self.w2.weight
        w3 = self.w3.weight
        # W1x.sizgmoid(W1.x) -- A
        w1x = einsum(w1, x, 'd_ff d_model, b seq d_model -> b seq d_ff')
        silux = w1x * torch.sigmoid(w1x)

        # W3x -- B
        w3x = einsum(w3, x, 'd_ff d_model, b seq d_model -> b seq d_ff')
        # matmul AB or SiLU(W1x) * W3x
        sw1_w3x = silux * w3x
        # W2.matmul result
        result = einsum(w2, sw1_w3x, 'd_model d_ff, b seq d_ff -> b seq d_model')

        return result
