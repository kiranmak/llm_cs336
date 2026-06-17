from typing import Iterable
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
import einx
from torch import Tensor
import torch.nn.functional as F
import math

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

def learning_rate_schedule(t, lr_max, lr_min, tw, tc):
    """
    Write a function that takes 𝑡, 𝛼max, 𝛼min, 𝑇𝑤 and 𝑇𝑐, and returns the
    learning rate 𝛼𝑡 according to the scheduler defined above. Then
    implement [adapters.get_lr_cosine_schedule] and make sure it passes uv
    run pytest -k test_get_lr_cosine_schedule.
    """
    lr_t = 0
    if t < tw:
        lr_t = (t/tw) * lr_max
    elif t <= tc:
        theta = ((t - tw)/ (tc - tw)) * math.pi
        lr_t = lr_min + 0.5 * (1 + math.cos(theta)) * (lr_max - lr_min)
    else:
        lr_t = lr_min
    return lr_t

### GPT code. Just to understand. I dint write it. 
def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
                      max_l2_norm: float) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # Calculate the norm for each parameter's gradient
    norms = [torch.norm(p.grad.detach()) for p in params_with_grad]
    
    # Stack the individual norms into a tensor
    # For MPS device, ensure tensor operations stay on MPS for efficiency
    if torch.backends.mps.is_available():
        stacked_norms = torch.stack(norms)
        total_norm = torch.linalg.norm(stacked_norms)
    else:
        total_norm = torch.norm(torch.stack(norms))

    # Calculate the clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    # Apply the clipping coefficient to each parameter's gradient
    # Ensure the coefficient is on the same device as the gradient
    if torch.backends.mps.is_available():
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef_clamped)
    else:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

