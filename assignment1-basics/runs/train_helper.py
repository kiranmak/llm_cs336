import argparse
import sys
import os
import torch
import numpy as np
from cs336_basics.configs import TrainingConfig
from cs336_basics.paths import set_device,  get_amptype, DATA_PATH, OUT_PATH
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel

from cs336_basics.nn_utils import (
        get_batch,
        cross_entropy_loss,
 )

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
    if not token_npy_path:
        return None
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
                  device_type: str,
                  amp_dtype: torch.dtype,
                  device) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        X, Y = get_batch(
                   data,
                   batch_size,
                   context_length,
                   device)

        with torch.amp.autocast(device_type, dtype=amp_dtype):
            logits = model(X)  # (B, S, V)
            B, S, V = logits.shape
            loss = cross_entropy_loss(
                    logits.reshape(B * S, V),
                    Y.reshape(B * S))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))

def conditional_compile(model, device):
    """Compiles the model ONLY if running on CUDA to prevent
       Mac crashes."""
    if device.type == "cuda":
        print("CUDA detected: Compiling model graph...")
        return torch.compile(model)
    if device.type == "mps":
        print("MPS detected: using AOT eager mode for graph compilation.")
        model = model.to(device)
        return torch.compile(model, backend="aot_eager")
    else:
        print("CPU detected: using eager mode.")
    return model

def training_initializer(cfg: TrainingConfig):

    torch.manual_seed(68)
    np.random.seed(68)

    # harware initial setup
    device = set_device(None)
    amp_dtype, device_type = get_amptype()
    model_dtype = torch_dtype_from_string(cfg.model_dtype)

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

    criterion =  cross_entropy_loss
    return model, optimizer, criterion, device


def parse_user_params():

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--name', type=str, default="train_00")
    args_parser.add_argument('--vocab_size', type=int, default=10000)
    args_parser.add_argument('--batch_size', type=int, default=16)
    args_parser.add_argument('--contextlen', type=int, default=256)
    args_parser.add_argument('--d_model',    type=int, default=512)
    args_parser.add_argument('--d_ff',       type=int, default=1344)
    args_parser.add_argument('--theta',      type=int, default=10000)
    args_parser.add_argument('--num_layers', type=int, default=4)
    args_parser.add_argument('--num_heads',  type=int, default=16)
    args_parser.add_argument('--resume', action='store_true', default=False)
    args_parser.add_argument('--loginterval', type=int, default=200)
    args_parser.add_argument('--evalinterval', type=int, default=200)
    args_parser.add_argument('--tokenfile',  type=str, default=None)
    args_parser.add_argument('--maxsteps',  type=int, default=5000)
    args_parser.add_argument('--warmup',  type=float, default=0.05)
    args_parser.add_argument('--cosine',  type=float, default=1.0)
    args_parser.add_argument('--lr',  type=float, default=3e-4)

    if len(sys.argv)==1:
        args_parser.print_help(sys.stderr)
        sys.exit(1)
    args = args_parser.parse_args()
    if args.tokenfile == None:
        print("Token file is needed, please provide the token file.")
        sys.exit(1)
    """
    print(f"Using Training file: {args.tokenfile}")
    print(f"      vocab file: {OUT_PATH /args.tokenfile}-train_vocab.json")
    print(f"      merge file: {OUT_PATH /args.tokenfile}-train_merges.txt")
    print(f"      token file: {OUT_PATH /args.tokenfile}-train.bin    ")
    print(f"      validation file: {OUT_PATH /args.tokenfile}-valid.bin    ")
    """

    print("------args.name", args.name)
    # Initialize your config (overriding only what you need)
    config = TrainingConfig(
        input_src_file = str(DATA_PATH /(args.tokenfile+"-train.bin")),
        vocab_file = str(OUT_PATH /(args.tokenfile+"-train_vocab.json")),
        merge_file = str(OUT_PATH /(args.tokenfile+"-train_merges.txt")),
        dataset = str(OUT_PATH /(args.tokenfile+"-train.bin")),
        valid_set = str(OUT_PATH /(args.tokenfile+"-valid.bin")),
        exp_name = args.name,
        batch_size=args.batch_size,
        max_steps = args.maxsteps,
        context_length= args.contextlen,
        resume = args.resume,
        vocab_size = args.vocab_size,
        d_model = args.d_model,
        d_ff = args.d_ff,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        rope_theta = args.theta,
        log_interval = args.loginterval,
        eval_interval = args.evalinterval,
        warmup_iters = int(args.maxsteps * args.warmup),
        cosine_cycle_iters = int(args.maxsteps * args.cosine),
        lr_max = args.lr,
        lr_min = args.lr * 0.1,
    )

    print("------config.exp_name", config.exp_name)
    return config

