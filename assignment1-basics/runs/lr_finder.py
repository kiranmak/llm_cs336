from runs.train_helper import open_memmap_1d
import torch
import math
import matplotlib.pyplot as plt

from runs.train_helper import training_initializer, conditional_compile
from cs336_basics.configs import TrainingConfig
from cs336_basics.paths import  DATA_PATH, OUT_PATH

from cs336_basics.nn_utils import (
        get_batch,
        cross_entropy_loss,
 )

class LRFinder:
    def __init__(self, model, optimizer, criterion ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.optimizer_state = optimizer.state_dict()

    def range_test(self, cfg, start_lr=1e-7, end_lr=10.0, num_iter=100, smooth_beta=0.98, device=None):
        self.model.train()
        # Compile model for kernel fusion
        #self.model = conditional_compile(self.model, device)

        # Calculate multiplier for exponential increment
        lr_multiplier = (end_lr / start_lr) ** (1.0 / num_iter)

        # Set initial learning rate in optimizer
        for pg in self.optimizer.param_groups:
            pg['lr'] = start_lr

        lrs = []
        losses = []

        best_loss = float('inf')
        running_loss = 0.0
        current_lr = start_lr

        train_mm = open_memmap_1d(cfg.dataset, np_dtype = cfg.np_dtype)

        for i in range(num_iter):
            X, Y = get_batch(train_mm,
                         cfg.batch_size,
                         cfg.context_length,
                         device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            Xf, Yf = logits.view(-1, cfg.vocab_size), Y.view(-1)
            loss =  cross_entropy_loss(Xf, Yf)

            # Stop if loss explodes (divergence)
            if i > 0 and loss.item() > 4 * best_loss:
                print(f"Stopping early: Loss diverged at LR {current_lr:.2e}")
                break

            loss.backward()
            self.optimizer.step()

            # Track best loss
            if loss.item() < best_loss or i == 0:
                best_loss = loss.item()

            # Apply EMA smoothing to the loss
            running_loss = smooth_beta * running_loss + (1 - smooth_beta) * loss.item()
            smoothed_loss = running_loss / (1 - smooth_beta ** (i + 1)) # Bias correction
            if (i % 10) == 0:
                msg =  f"step: {i} lr: {current_lr:.2e} loss: {loss.item():.4f} "
                msg += f"running loss: {running_loss:.4f} smoothed_loss: {smoothed_loss:.4f}"
                print(msg)

            lrs.append(current_lr)
            losses.append(smoothed_loss)

            # Step the learning rate exponentially
            current_lr *= lr_multiplier
            for pg in self.optimizer.param_groups:
                pg['lr'] = current_lr

        # Restore original weights and optimizer states
        self.model.load_state_dict({k: v.to(device) for k, v in self.model_state.items()})
        self.optimizer.load_state_dict(self.optimizer_state)
        print("Model states restored successfully!")

        return lrs, losses

    def plot(self, lrs, losses):
        plt.figure(figsize=(8, 5))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate (Log Scale)')
        plt.ylabel('Smoothed Loss')
        plt.title('Learning Rate Range Test')
        plt.grid(True, which="both", ls="-")
        plt.show()

if __name__ == "__main__":
    # Initialize your config (overriding only what you need)
    config = TrainingConfig(
        input_src_file = None,
        vocab_file = None,
        merge_file = None,
        exp_name="sweep",
        dataset = str(OUT_PATH   / "TinyStoriesV2-GPT4-train.bin"),
        valid_set = None,
        batch_size=64,  # custom override
        resume = False,
    )

    model, optimizer, criterion, device = training_initializer(config)
    lr_finder = LRFinder(model, optimizer, criterion )
    lrs, losses = lr_finder.range_test(
                                    cfg=config,
                                    start_lr=3e-4,
                                    end_lr=3e-3,
                                    num_iter=150,
                                    device=device
                                )
    lr_finder.plot(lrs, losses)
