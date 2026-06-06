import torch
import torch.nn as nn
#import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
import einx
from torch import Tensor

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
