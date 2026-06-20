import argparse
import sys
import os
import time
from datetime import datetime

import math
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from cs336_basics.nn_utils import get_batch, cross_entropy_loss
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel
from cs336_basics.nn_utils import save_checkpoint, load_checkpoint

class CheckPtConfig:
    # Other chackpointing Configuration
    def __init__(self):
        self.dir = "./checkpoints"
        os.makedirs(self.dir, exist_ok=True)
        self.interval = 5000 # Save every 5,000 steps
        self.max_keep = 3    # Keep only the 3 most recent full states
        self.saved_paths = []

class ConfigParams:
    def __init__(self, batch_size, context_length, vocab_size, d_model,
                       d_ff, num_layers, num_heads, theta, epochs=3):
        self.batch_size     = batch_size
        self.context_length = context_length
        self.vocab_size     = vocab_size
        self.d_model        = d_model
        self.d_ff           = d_ff
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.rope_theta     = theta
        self.epochs         = epochs
        self.checkpoint     = CheckPtConfig()

    def show(self):
        print("     batch_size     =", self.batch_size)
        print("     context_length =", self.context_length)
        print("     vocab_size     =", self.vocab_size)
        print("     d_model        =", self.d_model)
        print("     d_ff           =", self.d_ff)
        print("     num_layers     =", self.num_layers)
        print("     num_heads      =", self.num_heads)
        print("     rope_theta     =", self.rope_theta)
        print("     epochs         =", self.epochs)
        print("     Checkpoints:");
        print("         dir      =", self.checkpoint.dir)
        print("         interval =", self.checkpoint.interval)
        print("         max_save =", self.checkpoint.max_keep)
        print("         interval =", self.checkpoint.saved_paths)


def checkpoint_sync(model, optimizer, global_step, chkpt):
    checkpoint_path = os.path.join(chkpt.dir, f"checkpoint_step_{global_step}.pt")

    save_checkpoint(model, optimizer, global_step, checkpoint_path)

    max_chkpts     = chkpt.max_keep

    print(f"\n[CHECKPOINT SYNC] iteration {global_step} ({checkpoint_path})")

    # Manage rotating history to prevent storage exhaustion
    chkpt.saved_paths.append(checkpoint_path)
    if len(chkpt.saved_paths) > max_chkpts:
        oldest_checkpoint = chkpt.saved_paths.pop(0)
        if os.path.exists(oldest_checkpoint):
            os.remove(oldest_checkpoint)
            print(f"[CLEANUP] Deleted old checkpoint: {oldest_checkpoint}")

def checkpoint_resume(model, optimizer, chkpt):
    from pathlib import Path
    checkpoint_dir = Path(chkpt.dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {chkpt.dir} does not exist.")
        return 0

    # Filter for files only, then pick the one with the maximum modification time
    files = [f for f in checkpoint_dir.iterdir() if f.is_file() and f.name.endswith(".pt")]
    if not files:
        print(f"No checkpoints found in {chkpt.dir}.")
        return 0

    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    global_step = load_checkpoint(latest_file, model, optimizer)

    print(f"\n[CHECKPOINT-RESUME] training iteration {global_step} ({latest_file})")

    # Manage rotating history to prevent storage exhaustion
    chkpt.saved_paths.append(str(latest_file))
    return global_step

def training_together(dataset, config_params, device, resume):
    # Initialize data pipeline, model, and optimizer
    batch_size     = config_params.batch_size
    context_length = config_params.context_length
    vocab_size     = config_params.vocab_size
    d_model        = config_params.d_model
    d_ff           = config_params.d_ff
    num_layers     = config_params.num_layers
    num_heads      = config_params.num_heads
    rope_theta     = config_params.rope_theta
    total_epochs   = config_params.epochs
    chkpt_interval = config_params.checkpoint.interval

    total_batches = len(dataset) // (batch_size)
    # initialized weights 0:
    model = TransformerModel(vocab_size, d_model,
                             context_length, rope_theta,
                             num_heads, d_ff, num_layers).to(device)
    model.to(device)
    optimizer = AdamW(model.parameters(),
                      lr=1e-3,
                      weight_decay=0.01,
                      betas=(0.9, 0.999), eps=1e-8,)

    global_step = 0
    chkpt = config_params.checkpoint
    if resume:
        global_step = checkpoint_resume(model, optimizer, chkpt)

    # Loss function (CrossEntropyLoss expects raw logits)

    for epoch in range(total_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0

        for step in range(total_batches):
            X, Y = get_batch( dataset, batch_size, context_length, device,)
            X, Y = X.to(device), Y.to(device)

            # 1. Clear out previously accumulated gradients
            optimizer.zero_grad()

            # 2. Forward pass: compute predicted logits from inputs
            # X shape: (B, T) -> Logits shape: (B, T, Vocab_Size)
            logits = model(X)

            # 3. Flatten tensors for CrossEntropyLoss evaluation
            # CrossEntropy expects predictions: (N, C) and targets: (N)
            # where N = B * T, and C = Vocab_Size
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = Y.view(-1)

            loss =  cross_entropy_loss(logits_flat, targets_flat)

            # 4. Backward pass: Calculate parameter gradients via backpropagation
            loss.backward()

            # 5. Optimization step: update weights using AdamW equations
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            if global_step % chkpt_interval == 0:
                checkpoint_sync(model, optimizer,
                                    global_step, chkpt)


            if step % 10 == 0:
                # Calculate Perplexity dynamically from our loss value
                perplexity = torch.exp(torch.tensor(loss.item())).item()
                print(f"Epoch {epoch+1}/{total_epochs} | Step {step:3d} | Loss: {loss.item():.4f} | PPL: {perplexity:.2f}")

        avg_epoch_loss = epoch_loss / total_batches
        print(f"=== Finished Epoch {epoch+1} | Average Loss: {avg_epoch_loss:.4f} ===")


def run_main():

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--debug', action='store_true', default=False)
    args_parser.add_argument("--device", type=str, default="gpu")

    args_parser.add_argument('--vocab',      action='store_true', default=10000)
    args_parser.add_argument('--batch_size', action='store_true', default=16)
    args_parser.add_argument('--contextlen', action='store_true', default=256)
    args_parser.add_argument('--d_model',    action='store_true', default=512)
    args_parser.add_argument('--d_ff',       action='store_true', default=1408)
    args_parser.add_argument('--theta',      action='store_true', default=10000)
    args_parser.add_argument('--num_layers', action='store_true', default=4)
    args_parser.add_argument('--num_heads',  action='store_true', default=16)
    args_parser.add_argument('--epochs',     action='store_true', default=3)
    args_parser.add_argument('--resume',     action='store_true', default=False)
    args_parser.add_argument('--tokenfile',  type=str, default=None)

    args = args_parser.parse_args()
    if args.tokenfile == None:
       print("Token file is a mandatory input, please provide the path to the token file.") 
       exit(1)
    
    if not os.path.exists(args.tokenfile):
        print("Token file does not exist, please provide the path to the token file.")
        exit(1)


    hyper_params = ConfigParams(args.batch_size, args.contextlen,
                                args.vocab, args.d_model,
                                args.d_ff, args.num_layers,
                                args.num_heads, args.theta, args.epochs)
    hyper_params.show()
    print("Training file: ", args.tokenfile)

    if args.device == "gpu"\
            and torch.backends.mps.is_available()\
            and torch.backends.mps.is_built():
        args.device = torch.device("mps")
    elif args.device == "gpu" and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    print("Using device: ", args.device)

    assert (torch.__version__ >= "1.0.0"), "Please install torch version >=1.0.0 "

    print(80 * "=")
    print(f"Starting Training Loop for {args.epochs}...")
    print(80 * "=")
    start = time.time()

    dataset = np.load(args.tokenfile, mmap_mode="r")

    print("Corpus size: ", len(dataset))
    training_together(dataset, hyper_params, args.device, args.resume)

    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    """
    output_dir = "run_results_(soln)/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005, device=args.device)

    if not args.debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        print("Done!")
    """

if __name__ == "__main__":
    run_main()
