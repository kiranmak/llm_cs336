import os, time, torch, numpy as np
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel
from cs336_basics.paths import OUT_PATH, set_device, get_amptype

from cs336_basics.nn_utils import (
        get_batch,
        cross_entropy_loss,
        gradient_clipping,
        learning_rate_schedule,
 )

torch.set_num_threads(os.cpu_count() - 2)
device = set_device(None)
batch_size, context_length, vocab_size = 256, 256, 10000
d_model, d_ff, num_layers, num_heads, rope_theta = 512, 1344, 4, 16, 10000

def benchmark_cpu_training_step(dataset, model, optimizer):
    # warmup
    for _ in range(3):
        X, Y = get_batch(dataset, batch_size, context_length, device)
        logits = model(X)
        loss = cross_entropy_loss(logits.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        X, Y = get_batch(dataset, batch_size, context_length, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = cross_entropy_loss(logits.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - t0
    print(f"CPU threads:         {torch.get_num_threads()}")
    print(f"Avg step time:       {elapsed/n:.2f}s")
    print(f"Sample dataset size: {len(dataset):,} tokens")
    print(f"100 steps ETA:       {elapsed/n*100/60:.1f} min")
    print(f"total steps ETA:     {(len(dataset)//batch_size):,} ")
    print(f"Total batches/epoch (bs=4): {len(dataset)//batch_size}")
    print(f"Full epoch ETA:      {elapsed/n * (len(dataset)//batch_size) / 3600:.1f} hours")

    # param count
    params = sum(p.numel() for p in model.parameters())
    print(f"Params:              {params/1e6:.2f}M")

def estimate_gpu_step_time(dataset, vocab_size, batch_size, context_length,
                        gpu_peak_tflops, mfu, total_steps):

    model = TransformerModel(vocab_size, d_model, context_length,
                             rope_theta, num_heads, d_ff,
                             num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3,
                      weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    params = sum(p.numel() for p in model.parameters())
    tokens = batch_size * context_length
    flops_per_step = 6 * params * tokens
    effective = gpu_peak_tflops * mfu * 1e12
    step_sec = flops_per_step / effective
    print(f"context length: {context_length}")
    print(f"batch size: {batch_size}")
    print(f"vocab size: {vocab_size}")
    print(f"d_model: {d_model}")
    print(f"d_ff: {d_ff}")
    print(f"num_layers: {num_layers}")
    print(f"num_heads: {num_heads}")
    print(f"rope_theta: {rope_theta}")
    print(f"Sample dataset size: {len(dataset):,} tokens")
    print(f"Params preclipping:  {params/1e6:.2f}M")

    elapsed = one_step_run(dataset, model, optimizer, vocab_size,
                           batch_size, context_length, device,
                           step=1, running_loss = 0.0)

    print(f"\n--- GPU estimate (peak={gpu_peak_tflops} TFLOPS, MFU={mfu:.0%}) ---")
    print(f"Avg step time:       {elapsed:.2f}s")
    print(f"FLOPs/step:          {flops_per_step/1e9:.1f} GFLOP")
    print(f"Est. GPU step time:  {step_sec:.3f}s")
    print(f"100 steps ETA:       {step_sec*100/60:.1f} min")
    print(f"Full epoch ETA:      {step_sec*total_steps/3600:.1f} hours")
    print(f"total steps ETA:     {(total_steps):,} ")
    print(f"Total batches/epoch (bs=4): {total_steps}")
    print(f"Full epoch ETA:      {step_sec*total_steps/3600:.1f} hours")
    return step_sec

def one_step_run(dataset, model, optimizer,
                 vocab_size,
                 batch_size, context_length,
                 device, step, running_loss):

    step_start = time.time()
    model = torch.compile(model)
    model.train()
    lr = learning_rate_schedule(t=step, lr_max= 1e-3, lr_min= 1e-4,
        tw = 2000, tc = 106_667)
    for group in optimizer.param_groups:
       group["lr"] = lr

    # 1. Grab the full batch (256 sequences)
    X_full, Y_full = get_batch(dataset, batch_size, context_length, device)

    # Clear out previously accumulated gradients
    optimizer.zero_grad(set_to_none=True)

    # 2. Define a smaller micro-batch size that easily fits into your Mac's RAM
    micro_batch_size = 32
    total_loss_scalar = 0.0
    amp_dtype, device_type = get_amptype()
    # 3. Process the batch in smaller slices
    for i in range(0, batch_size, micro_batch_size):
        X_micro = X_full[i : i + micro_batch_size]
        Y_micro = Y_full[i : i + micro_batch_size]

        # Forward pass on a tiny slice (32 x 256 x 10000) instead of the massive one
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
            logits_micro = model(X_micro)
            # Calculate loss for this micro-step
            loss_micro = cross_entropy_loss(
                       logits_micro.view(-1,vocab_size),
                       Y_micro.view(-1))

            # Scale the loss relative to the chunk size so gradients
            # average out perfectly
            scale_factor = micro_batch_size / batch_size
            loss_scaled = loss_micro * scale_factor

            # Backward pass accumulates gradients directly into
            # model.parameters().grad
            loss_scaled.backward()

        # Track the unscaled loss value for printing
        total_loss_scalar += loss_micro.item() * scale_factor

    grad_norm = gradient_clipping(model.parameters(),1.0)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params with clipping:  {params/1e6:.2f}M")
    optimizer.step()
    running_loss += total_loss_scalar

    step_end = time.time()
    tokens_processed = batch_size * context_length
    tok_s = tokens_processed/(step_end - step_start)
    msg = f"step={step+1} loss={total_loss_scalar:.4f} lr={lr:.3e}"
    msg += f" tok/s={tok_s:.1f}"
    print(msg)
    return (step_end - step_start)


if __name__ == "__main__":
    from runs.train import open_memmap_1d
    tokenfile = "TinyStoriesV2-GPT4"
    token_train_path = OUT_PATH / f"{tokenfile}-train.bin"
    dataset = open_memmap_1d(token_train_path, np_dtype = "uint16")
    estimate_gpu_step_time(dataset, 10000, batch_size, context_length,
                       gpu_peak_tflops=8.9,
                       mfu=0.35,
                       total_steps=len(dataset) // batch_size)

    tokenfile = "OpenWebText"
    token_train_path = OUT_PATH / f"{tokenfile}-train.bin"
    dataset = open_memmap_1d(token_train_path, np_dtype = "uint16")
    estimate_gpu_step_time(dataset, 32000, batch_size, context_length,
                       gpu_peak_tflops=8.9,
                       mfu=0.35,
                       total_steps=len(dataset) // batch_size)

