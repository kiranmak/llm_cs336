from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iter number from the state, or 0.
                grad = p.grad.data # Get the grad of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


"""
init(𝜃) ▷ Initialize learnable parameters
𝑚←0 ▷ Initial value of the first moment vector; same shape as 𝜃
𝑣←0 ▷ Initial value of the second moment vector; same shape as 𝜃
for 𝑡=1,…,𝑇 do
    Sample batch of data 𝐵𝑡
    𝑔←∇𝜃ℓ(𝜃;𝐵𝑡) ▷ Compute the gradient of the loss
    𝛼𝑡←𝛼√1−𝛽𝑡21−𝛽𝑡1
    ▷ Compute adjusted 𝛼 for iteration 𝑡
    𝜃←𝜃−𝛼𝜆𝜃 ▷ Apply weight decay
    𝑚←𝛽1𝑚+(1−𝛽1)𝑔 ▷ Update the first moment estimate
    𝑣←𝛽2𝑣+(1−𝛽2)𝑔2 ▷ Update the second moment estimate
    𝜃←𝜃−𝛼𝑡𝑚√𝑣+𝜀 ▷ Apply moment-adjusted weight updates
end for
Algorithm 1: AdamW
"""
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01,
                    betas=(0.9, 0.999), eps=1e-8,):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        self.beta_1,self.beta_2 = betas[0], betas[1]

        defaults = {"lr": lr,
                    "epsilon": eps,
                    "lambda": weight_decay,
                    }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            _lambda = group["lambda"]
            _epsilon = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data # Get the grad of loss with respect to p.
                state = self.state[p] # Get state associated with p.
                # in the very first training step, t' won't exist. Initialize it:
                if "t" not in state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # 1. Fetch and increment 't' for the current step
                t = state["t"] + 1

                m, v = state["m"], state["v"]

                m.mul_(self.beta_1).add_(grad, alpha=1 - self.beta_1)
                v.mul_(self.beta_2).addcmul_(grad, grad, value=1 - self.beta_2)

                p.data -= lr * _lambda * p.data

                bias_correction1 = 1 - self.beta_1 ** t
                bias_correction2 = 1 - self.beta_2 ** t
                lr_t = lr * (math.sqrt(bias_correction2) / bias_correction1)

                p.data -= lr_t * (m/v.sqrt().add_(_epsilon))
                state["t"] = t  # update state
        return loss

"""
Following code is to understand nuances of optimizer skeleton code
"""
def sgd_tester():
    import copy
    weights_x = torch.nn.Parameter(5 * torch.randn((10, 10)))
    for lr in [1e-1, 1e-2, 1e-3]:
        weights = copy.deepcopy(weights_x.data)
        weights.requires_grad = True
        opt = SGD([weights], lr=lr)
        losses = []
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            loss.backward() # Run backward pass, which computes gradients.
            losses.append(loss.cpu().item())
            opt.step() # Run optimizer step
        print("with LR=", lr, "losses",
              [round(l, 4) for l in losses[:9]], "...")

if __name__ == "__main__":
    sgd_tester()
