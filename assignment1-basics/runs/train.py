import os
import time
import math
import torch
import json
from torch import nn
import numpy as np
from pathlib import Path
from cs336_basics.nn_utils import (
        get_batch,
        cross_entropy_loss,
        gradient_clipping,
        learning_rate_schedule,
        save_checkpoint, load_checkpoint,
 )
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel
from cs336_basics.paths import (
        OUT_PATH, EXP_PATH, set_device, get_amptype,
        DATA_PATH
        )
from cs336_basics.checkpoints import checkpoint_resume, checkpoint_sync
from cs336_basics.experiments_tracker import ExperimentTracker

from runs.train_helper import (
    open_memmap_1d, torch_dtype_from_string, estimate_loss,
    conditional_compile
    )
from cs336_basics.configs import CheckPtConfig, TrainingConfig


def print_msg(step, loss,tok_s, lr, fwd_pass_time):
    msg = f"[train] step={step+1} loss={loss.item():.4f} lr={lr:.3e}"
    msg += f" tok/s={tok_s:.1f}"
    msg += f" fp_tm={(fwd_pass_time):.2f}"
    print(msg)

def validation_step(cfg, best_val, step, exp, model, optimizer, device):
    valid_mm = open_memmap_1d(cfg.valid_set, cfg.np_dtype)
    val_t = time.time()
    val_loss = estimate_loss(model,
                            valid_mm,
                            cfg.eval_batches,
                            cfg.batch_size,
                            cfg.context_length,
                            device)

    val_pplex = float(math.exp(val_loss))
    val_t = time.time() - val_t
    print(f"[ eval] step={step+1} val_loss={val_loss:.4f}",
          f" val_perplexity={val_pplex:.2f}",
          f" val_tm(s)={val_t:.2f}")

    metrics={"val/loss": float(val_loss),
             "val/ppl": float(val_pplex)}
    exp.log(step+1, metrics)

    # Save the best-performing checkpoint
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(model, optimizer, step+1, cfg.best_chkpt_file)

    return best_val, model, optimizer


def main_training_loop(cfg: TrainingConfig):

    print("Training max steps: ", cfg.max_steps)

    torch.manual_seed(0)
    np.random.seed(0)

    print(78 * "=")
    print("Starting Training...")
    print(78 * "=")
    start = time.time()

    # harware initial setup
    device = set_device(None)
    amp_dtype, device_type = get_amptype()
    model_dtype = torch_dtype_from_string(cfg.model_dtype)

    os.makedirs(cfg.chkpt_dir, exist_ok=True)

    sname = str(Path(cfg.input_src_file).stem)
    exp = ExperimentTracker(sname, mode="train")

    # initialized weights 0:
    model = TransformerModel(cfg.vocab_size,
                            cfg.d_model,
                            cfg.context_length,
                            cfg.rope_theta,
                            cfg.num_heads,
                            cfg.d_ff,
                            cfg.num_layers,
                            device,
                            dtype=model_dtype).to(device)

    optimizer = AdamW(model.parameters(),
                      cfg.lr_max,
                      cfg.weight_decay,
                      (cfg.beta1,cfg.beta2),
                      cfg.eps)

    # Setup runtime tracking variables
    start_time = time.time()
    running_loss = 0.0
    # param count
    params = sum(p.numel() for p in model.parameters())
    print(f"Params:              {params/1e6:.2f}M")

    chkpt = CheckPtConfig()
    chkpt.interval = cfg.chkpt_interval
    starting_step = 0
    if cfg.resume:
        starting_step = checkpoint_resume(model, optimizer, chkpt)

    model.train() # Set model to training mode
    best_val      = float("inf")
    train_mm = open_memmap_1d(cfg.dataset, np_dtype = cfg.np_dtype)
    print("Train Corpus size: ", len(train_mm))
    # Compile model for kernel fusion
    model = conditional_compile(model, device)
    model.train()

    print(f"curr_start/total steps: {starting_step}/{cfg.max_steps}\n")

    for step in range(starting_step, cfg.max_steps):
        step_start = time.time()

        # 7.1 Update learning rate according to schedule
        lr = learning_rate_schedule(
            step,
            cfg.lr_max,
            cfg.lr_min,
            cfg.warmup_iters,
            cfg.cosine_cycle_iters
        )
        for group in optimizer.param_groups: group["lr"] = lr

        # 7.2 sample a batch from training data
        X, Y = get_batch(train_mm,
                         cfg.batch_size,
                         cfg.context_length,
                         device)

        optimizer.zero_grad(set_to_none=True)

        # 7.3. Forward pass: compute predicted logits from inputs
        fp_start = time.time()
        with torch.amp.autocast(device_type, amp_dtype):
            # penalize predictions that are wrong
            logits = model(X)
            Xf, Yf = logits.view(-1, cfg.vocab_size), Y.view(-1)
            loss =  cross_entropy_loss(Xf, Yf)

        # 7.4. Calculate parameter gradients - back propagation
        loss.backward()
        fp_end = time.time()

        # 7.5 Gradient clipping for training stability
        grad_norm = gradient_clipping(model.parameters(), cfg.grad_clip)

        # 7.6. Optimization step: update weights using AdamW equations
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % chkpt.interval == 0:
            checkpoint_sync(model, optimizer, step, chkpt)

        # 7.7 Periodic logging
        if (step+1) % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            # step * B * T
            tok_s = ((step - starting_step) *\
                       cfg.batch_size * cfg.context_length)/ elapsed

            print_msg(step, loss, tok_s, lr, fp_end - fp_start)

            # Pack metrics dictionary
            metrics = {
                "train/loss_avg": running_loss / cfg.log_interval,
                "train/perplexity": torch.exp(torch.tensor(
                          running_loss / cfg.log_interval)).item(),
                "train/grad_norm": grad_norm,
                "train/curr_step": step,
                "perf/hours": elapsed / 3600.0,
                "perf/tokens_per_sec": tok_s,
                "perf/sec_per_step": time.time() - step_start
            }
            exp.log(step+1, metrics)


        # 7.8 Periodic evaluation on validation set
        if cfg.valid_set and (step + 1) % cfg.eval_interval == 0:
            best_val, model, optimizer = validation_step(cfg, best_val, step, exp, model, optimizer, device)
            # Reset accumulation buffer
            running_loss = 0.0

    print(f"Final Checkpoint Done")
    checkpoint_sync(model, optimizer, cfg.max_steps, chkpt)
    cfg.save("hyperparams.json")
    exp.close()

    print("took {:.2f} seconds\n".format(time.time() - start))
    print(78 * "=")
    print("Training Finished...")
    print(78 * "=")


if __name__ == "__main__":
    # Initialize your config (overriding only what you need)
    config = TrainingConfig(
        input_src_file = str(DATA_PATH /"TinyStoriesV2-GPT4-train.txt"),
        vocab_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_vocab.json"),
        merge_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_merges.txt"),
        dataset = str(OUT_PATH   / "TinyStoriesV2-GPT4-train.bin"),
        valid_set = str(OUT_PATH / "TinyStoriesV2-GPT4-valid.bin"),
        batch_size=64,  # custom override
        resume = False,
    )

    main_training_loop(config)
