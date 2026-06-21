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

from cs336_basics.nn_utils import get_batch, cross_entropy_loss
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel
from cs336_basics.checkpoints import ConfigParams, CheckPtConfig,
                                     checkpoint_hyperparams,
                                     checkpoint_sync, checkpoint_resume


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
                print(f"Epoch {epoch+1}/{total_epochs} | Step {step:3d} |",
                      f"Loss: {loss.item():.4f} | PPL: {perplexity:.2f}")

        avg_epoch_loss = epoch_loss / total_batches
        print(f"=== Final Checkpoint ===")
        checkpoint_sync(model, optimizer, global_step, chkpt)
        print(f"=== Finished Epoch {epoch+1} | Average Loss: {avg_epoch_loss:.4f} ===")


def run_main():

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--debug', action='store_true', default=False)
    args_parser.add_argument("--device", type=str, default="gpu")

    args_parser.add_argument('--vocab',      type=int, default=10000)
    args_parser.add_argument('--batch_size', type=int, default=16)
    args_parser.add_argument('--contextlen', type=int, default=256)
    args_parser.add_argument('--d_model',    type=int, default=512)
    args_parser.add_argument('--d_ff',       type=int, default=1408)
    args_parser.add_argument('--theta',      type=int, default=10000)
    args_parser.add_argument('--num_layers', type=int, default=4)
    args_parser.add_argument('--num_heads',  type=int, default=16)
    args_parser.add_argument('--epochs',     type=int, default=1)
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


    #dataset = np.load(args.tokenfile, mmap_mode="r")
    #print("Corpus size: ", len(dataset))
    #training_together(dataset, hyper_params, args.device, args.resume)

    mock_corpus = torch.randint(0, args.vocab, (500,)).to(args.device)
    training_together(mock_corpus, hyper_params, args.device, args.resume)

    checkpoint_hyperparams(hyper_params, args.tokenfile)

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
