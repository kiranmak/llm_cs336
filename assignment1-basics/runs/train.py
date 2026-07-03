import argparse
import sys
import os
import time
import math
import torch
import json
from torch import nn
import numpy as np
from tqdm import tqdm
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
from cs336_basics.paths import OUT_PATH, EXP_PATH, set_device, get_amptype
from cs336_basics.tokenizer_exp import file_encode_bin_from_vocab_merges
from cs336_basics.checkpoints import (
        checkpoint_resume,
        checkpoint_sync,
        checkpoint_hyperparams)
from cs336_basics.configs import ConfigParams, get_preset_cfg
from cs336_basics.experiments_tracker import ExperimentTracker


def torch_dtype_from_string(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype string: {s}")


def open_memmap_1d(token_npy_path: str, np_dtype: str) -> np.memmap:
    """
    Open a 1D token memmap file. The file is npy
    """
    dtype = np.dtype(np_dtype)
    itemsize = dtype.itemsize
    nbytes = os.path.getsize(token_npy_path)
    length = nbytes // itemsize

    dataset = np.memmap(token_npy_path, mode="r", dtype=dtype,
                            shape=(length,))
    return dataset

@torch.no_grad()
def estimate_loss(model: torch.nn.Module,
                  data: np.memmap,
                  eval_batches,
                  batch_size,
                  context_length,
                  device) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        X, Y = get_batch(
                   data,
                   batch_size,
                   context_length,
                   device)

        logits = model(X)  # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy_loss(
                logits.reshape(B * S, V),
                Y.reshape(B * S))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))

def conditional_compile(model, device):
    """Compiles the model ONLY if running on CUDA to prevent Mac crashes."""
    if device.type == "cuda":
        print("CUDA detected: Compiling model graph...")
        return torch.compile(model)
    else:
        print("MPS/CPU detected: Skipping compilation step (using Eager mode).")
        return model


def training_together(dataset, validation_dataset,
                      config_params, presets,
                      exp, device, dtype):
    # Initialize data pipeline, model, and optimizer
    batch_size     = config_params.batch_size
    context_length = config_params.context_length
    vocab_size     = config_params.vocab_size
    d_model        = config_params.d_model
    d_ff           = config_params.d_ff
    num_layers     = config_params.num_layers
    num_heads      = config_params.num_heads
    rope_theta     = config_params.rope_theta
    resume         = config_params.resume
    chkpt_interval = config_params.checkpoint.interval

    # initialized weights 0:
    model = TransformerModel(vocab_size,
                            d_model,
                            context_length,
                            rope_theta,
                            num_heads,
                            d_ff,
                            num_layers,
                            device=device,
                            dtype=dtype).to(device)

    optimizer = AdamW(model.parameters(),
                    lr           = presets.optim.lr_max,
                    weight_decay = presets.optim.weight_decay,
                    betas  =(presets.optim.beta1, presets.optim.beta2),
                    eps          =presets.optim.eps)

    # Setup runtime tracking variables
    global_step = 0
    start_time = time.time()
    running_loss = 0.0

    chkpt = config_params.checkpoint
    best_chkpt_path = os.path.join(chkpt.dir, "best_validation.pt")
    if resume:
        global_step = checkpoint_resume(model, optimizer, chkpt)


    model.train() # Set model to training mode
    best_val = float("inf")
    total_steps = presets.train.max_steps
    starting_step = global_step
    print(f"total steps: {total_steps}\nstarting step {global_step}")
    # Compile model for kernel fusion
    model = conditional_compile(model, device)
    amp_dtype, device_type = get_amptype()
    model.train()

    for step in range(global_step, total_steps):
        step_start = time.time()

        # 7.1 Update learning rate according to schedule
        lr = learning_rate_schedule(
            t=step,
            lr_max=presets.optim.lr_max,
            lr_min=presets.optim.lr_min,
            tw = presets.optim.warmup_iters,
            tc = presets.optim.cosine_cycle_iters
        )
        for group in optimizer.param_groups: group["lr"] = lr

        # 7.2 sample a batch from training data
        X, Y = get_batch(dataset,
                         batch_size,
                         context_length,
                         device,)
        # Clear out previously accumulated gradients
        optimizer.zero_grad(set_to_none=True)

        # 7.3. Forward pass: compute predicted logits from inputs
        # X shape: (B, S) -> Logits shape: (B, S, Vocab_Size)
        fp_start = time.time()
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
            logits = model(X)

            # Flatten
            loss =  cross_entropy_loss(
                        logits.view(-1, vocab_size),Y.view(-1)
                    )

        # 7.4. Calculate parameter gradients - back propagation
        loss.backward()
        fp_end = time.time()

        # 7.5 Gradient clipping for training stability
        if presets.optim.grad_clip > 0:
            grad_norm = gradient_clipping(model.parameters(),
                              presets.optim.grad_clip)

        # 7.6. Optimization step: update weights using AdamW equations
        optimizer.step()

        global_step += 1
        running_loss += loss.item()

        if global_step % chkpt_interval == 0:
            checkpoint_sync(model, optimizer,
                                    global_step, chkpt)

        # 7.7 Periodic logging
        if global_step % presets.train.log_interval == 0:
            elapsed = time.time() - start_time
            # step * B * T
            tokens_processed = (step - starting_step) *\
                                batch_size * context_length
            tok_s = tokens_processed/elapsed
            msg = f"[train] step={step+1} loss={loss.item():.4f} lr={lr:.3e}"
            msg += f" tok/s={tok_s:.1f}"
            msg += f" fp_tm={(fp_end-fp_start):.2f}"
            print(msg)

            # Pack metrics dictionary
            metrics = {
                "train/loss_avg": running_loss / presets.train.log_interval,
                "train/perplexity": torch.exp(torch.tensor(
                          running_loss / presets.train.log_interval)).item(),
                "train/grad_norm": grad_norm,
                "train/global_step": global_step,
                "perf/wall_clock_hours": elapsed / 3600.0,
                "perf/tokens_per_sec": tok_s,
                "perf/sec_per_step": time.time() - step_start
            }
            exp.log(step+1, metrics)


        # 7.8 Periodic evaluation on validation set
        if validation_dataset and (step + 1) % presets.train.eval_interval == 0:
            val_t = time.time()
            val_loss = estimate_loss(model, validation_dataset,
                                    eval_batches=presets.train.eval_batches,
                                    batch_size=batch_size,
                                    context_length=context_length,
                                    device=device)
            val_ppl = float(math.exp(val_loss))
            val_t = time.time() - val_t
            print(f"[ eval] step={step+1} val_loss={val_loss:.4f}",
                  f" val_ppl={val_ppl:.2f}",
                  f" val_tm(s)={val_t:.2f}")
            metrics={"val/loss": float(val_loss),
                     "val/ppl": float(val_ppl)}
            exp.log(step+1, metrics)

            # Save the best-performing checkpoint
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step+ 1, best_chkpt_path)

            # Reset accumulation buffer
            running_loss = 0.0

    print(f"=== Final Checkpoint ===")
    checkpoint_sync(model, optimizer, global_step, chkpt)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params:  {params/1e6:.2f}M")
    exp.close()


def parse_user_params():


    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--debug', action='store_true', default=False)
    args_parser.add_argument("--device", type=str, default="gpu")

    args_parser.add_argument('--vocab_size', type=int, default=10000)
    args_parser.add_argument('--batch_size', type=int, default=16)
    args_parser.add_argument('--contextlen', type=int, default=256)
    args_parser.add_argument('--d_model',    type=int, default=512)
    args_parser.add_argument('--d_ff',       type=int, default=1344)
    args_parser.add_argument('--theta',      type=int, default=10000)
    args_parser.add_argument('--num_layers', type=int, default=4)
    args_parser.add_argument('--num_heads',  type=int, default=16)
    args_parser.add_argument('--resume', action='store_true', default=False)
    args_parser.add_argument('--tokenfile',  type=str, default=None)
    args_parser.add_argument('--maxsteps',  type=int, default=5000)

    if len(sys.argv)==1:
        args_parser.print_help(sys.stderr)
        sys.exit(1)
    args = args_parser.parse_args()
    if args.tokenfile == None:
        print("Token file is needed, please provide the token file.")
        sys.exit(1)

    hyper_params = ConfigParams(args.batch_size, args.contextlen,
                                args.vocab_size, args.d_model,
                                args.d_ff, args.num_layers,
                                args.num_heads, args.theta, args.resume)
    hyper_params.show()
    print("Training file: ", args.tokenfile)
    return hyper_params, args.tokenfile, args.maxsteps

def run_main():

    hyper_params, tokenfile, maxsteps = parse_user_params()
    presets = get_preset_cfg()
    presets.train.max_steps = maxsteps
    print("Training max steps: ", maxsteps)

    device = set_device(None)
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() - 2)

    torch.manual_seed(presets.train.seed)
    np.random.seed(presets.train.seed)

    exp = ExperimentTracker(
         log_dir=EXP_PATH,
         service_name=tokenfile,
         config=presets,  # dataclass will be serialized
         mode="train",
     )

    # --- override params only ----
    print(78 * "=")
    print("Starting Training...")
    print(78 * "=")
    start = time.time()

    token_train_path = OUT_PATH / f"{tokenfile}-train.bin"
    train_mm = open_memmap_1d(token_train_path, np_dtype = "uint16")

    token_valid_path = OUT_PATH / f"{tokenfile}-valid.bin"
    valid_mm = open_memmap_1d(token_valid_path, np_dtype = "uint16")

    model_dtype = torch_dtype_from_string("float32")
    #presets.train.max_steps = len(trainmm) // (batch_size)
    print("Max possible steps", len(train_mm)//hyper_params.batch_size)
    presets.train.eval_interval = 100
    presets.train.log_interval == 200

    print("Train Corpus size: ", len(train_mm))
    training_together(train_mm, valid_mm, hyper_params,
                      presets,
                      exp,
                      device,
                      model_dtype)

    checkpoint_hyperparams(hyper_params, tokenfile)
    exp.close()

    print("took {:.2f} seconds\n".format(time.time() - start))
    print(78 * "=")
    print("Finished Training...")
    print(78 * "=")

if __name__ == "__main__":
    run_main()
