from typing import Iterable
import torch
import torch.nn as nn
import numpy.typing as npt
from jaxtyping import Float
from einops import einsum
from torch import Tensor
import torch.nn.functional as F
import math

def softmax_fn(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_val = in_features.max(dim=dim, keepdim=True)[0] # [0] because max returns a tuple.
    exps = torch.exp(in_features - max_val)
    return exps / exps.sum(dim=dim, keepdim=True)

def silu_fn(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
  return x * torch.sigmoid(x)

def cross_entropy_loss_orig(logits, targets):
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

    """
    x = F.one_hot(targets, num_classes=logits.size(-1))
    x_logits = einsum(logits, x.float(), 'b c, b c -> b')
    #Replace above one-hot + einsum with gather (same math, much faster,
    # still passes your tests):
    """
    x_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 4. flatten: log_sum_exp frpm (B, 1) to (B,)
    loss = -x_logits + log_sum_exp.squeeze(-1)
    #perplexity = torch.exp(loss.mean())

    #print()
    #print(f"Cross-Entropy Loss: {loss.mean().item():.4f}")
    #print(f"Perplexity:         {perplexity.item():.4f}")

    return loss.mean()

def cross_entropy_loss(logits_flat, targets_flat, chunk_size=1000):
    """
    Memory-efficient custom cross entropy using chunked processing
    to prevent MPS/CUDA Out-Of-Memory errors.
    """
    total_rows = logits_flat.size(0)
    total_loss = 0.0

    # Process the large tensor in smaller, manageable vertical slices
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)

        # Slice out a micro-chunk of logits and matching targets
        logits_chunk = logits_flat[i:end_idx]
        targets_chunk = targets_flat[i:end_idx]

        # Safe Log-Sum-Exp trick applied ONLY to this chunk
        max_val = torch.max(logits_chunk, dim=-1, keepdim=True)[0]
        shifted_logits = logits_chunk - max_val

        sum_exp = torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)
        log_softmax = shifted_logits - torch.log(sum_exp + 1e-9)

        # Gather the log-probabilities of the true target tokens
        target_log_probs = log_softmax.gather(dim=-1,
                                  index=targets_chunk.unsqueeze(-1)).squeeze(-1)

        # Accumulate the sum of losses for this chunk
        total_loss += -torch.sum(target_log_probs)

    # Return the average loss across the entire flattened batch
    return total_loss / total_rows


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

"""
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
"""

def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
                      max_l2_norm: float) -> float:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return 0.0

    # Calculate the norm for each parameter's gradient
    norms = [torch.norm(p.grad.detach()) for p in params_with_grad]

    # Stack the individual norms into a tensor
    if torch.backends.mps.is_available():
        stacked_norms = torch.stack(norms)
        total_norm = torch.linalg.norm(stacked_norms)
    else:
        total_norm = torch.norm(torch.stack(norms))

    # Calculate the clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # Apply the clipping coefficient to each parameter's gradient
    if torch.backends.mps.is_available():
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef_clamped)
    else:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

    return total_norm.item()

def get_batch(dataset: npt.NDArray, batch_size: int,
              context_length: int, device: str
              ) -> tuple[torch.Tensor, torch.Tensor]:

    import numpy as np
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be 1D, got shape {dataset.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")


    potential_start_indices = len(dataset) - context_length
    start_indices = np.random.randint(0, potential_start_indices, size=batch_size)

    indices = start_indices[:, None] + np.arange(context_length)

    X = torch.as_tensor(dataset[indices], dtype=torch.int64, device=device)
    Y = torch.as_tensor(dataset[indices + 1], dtype=torch.int64, device=device)

    return X, Y


#Implement the following two functions to load and save checkpoints:
def save_checkpoint(model, optimizer, iteration, out):
    """ This function should dump all the state from the
        model, optimizer and iteration into the file-like object out.
        * use the state_dict method of model and the optimizer
        * torch.save(obj, out) to dump obj into out
        * A typical choice is to have obj be a dictionary, but you can use whatever format
          you want as long as you can load your checkpoint later.
        This function expects the following parameters:
            model: torch.nn.Module
            optimizer: torch.optim.Optimizer
            iteration: int
            out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
     """
    checkpoint_state = {}
    checkpoint_state["iteration"] = iteration
    checkpoint_state["model"] = model.state_dict()
    checkpoint_state["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint_state, out)



def load_checkpoint(src, model, optimizer):
    """should load a checkpoint from src (path or file-like
       object), and then recover the model and optimizer states from that checkpoint. Your function
        should return the iteration number that was saved to the checkpoint.
        You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
        load_state_dict method in both the model and optimizer to return them to their previous
        states.
        This function expects the following parameters:
            src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
            model: torch.nn.Module
            optimizer: torch.optim.Optimizer
    """
    current_device = next(model.parameters()).device
    print(f" -> Mapping tensors to current host device: {current_device}")

    checkpoint_state = torch.load(src, map_location=current_device)
    model.load_state_dict(checkpoint_state["model"])
    optimizer.load_state_dict(checkpoint_state["optimizer"])
    return checkpoint_state["iteration"]

