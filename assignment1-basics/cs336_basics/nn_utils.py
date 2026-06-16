import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
import einx
from torch import Tensor
import torch.nn.functional as F

def softmax_fn(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_val = in_features.max(dim=dim, keepdim=True)[0] # [0] because max returns a tuple.
    exps = torch.exp(in_features - max_val)
    return exps / exps.sum(dim=dim, keepdim=True)

def silu_fn(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
  return x * torch.sigmoid(x)

def cross_entropy_loss(logits, targets):
    """
    Deliverable: cross-entropy loss, which takes in predicted logits
    (logits) and targets (𝑥𝑖+1) and computes the cross-entrop
        ℓ𝑖=−log softmax(𝑜𝑖)[𝑥𝑖+1].
        = 1/D -log (nomin/denom)
        nomin = x_logits, denom = log_sum_exp
        x_logits = logit*one_hot(targets)
        log_sum = sum(exp(logits))
    Your function should handle the following:
    • Subtract the largest element for numerical stability.
    • Cancel out log and exp whenever possible. - open flatten softmax
    • Handle any additional batch dimensions and return the average
      across the batch. -- squeeze
      As with Section 3.2, we assume batch-like dimensions
      always come first, before the vocabulary size dimension.
    adapters.run_cross_entropy run uv run pytest -k test_cross_entropy
    """
    # Cross Entropy = -x_logits + log_sum_exp
    # 1. Log-Sum-Exp Trick for numerical stability
    max_val = torch.max(logits, dim=-1, keepdim=True)[0]
    sum_exp = torch.sum(torch.exp(logits - max_val), dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp) + max_val

    # 2. Convert integer targets to a one-hot encoded matrix
    # If logits is (B, C), targets_one_hot becomes (B, C)
    x = F.one_hot(targets, num_classes=logits.size(-1))

    # 3. Use einsum to extract the correct target logits
    x_logits = einsum(logits, x.float(), 'b c, b c -> b')

    # 4. flatten: log_sum_exp frpm (B, 1) to (B,)
    loss = -x_logits + log_sum_exp.squeeze(-1)
    #perplexity = torch.exp(loss.mean())

    #print()
    #print(f"Cross-Entropy Loss: {loss.mean().item():.4f}")
    #print(f"Perplexity:         {perplexity.item():.4f}")

    return loss.mean()

