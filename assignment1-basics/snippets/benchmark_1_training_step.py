import os, time, torch, numpy as np
from cs336_basics.nn_utils import get_batch, cross_entropy_loss
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel
from cs336_basics.paths import OUT_PATH

torch.set_num_threads(os.cpu_count() - 2)
device = torch.device('cpu')
batch_size, context_length, vocab_size = 256, 256, 32000
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

def estimate_gpu_step_time(params, batch_size, context_length,
                        gpu_peak_tflops, mfu, total_steps):
    tokens = batch_size * context_length
    flops_per_step = 6 * params * tokens
    effective = gpu_peak_tflops * mfu * 1e12
    step_sec = flops_per_step / effective

    print(f"\n--- GPU estimate (peak={gpu_peak_tflops} TFLOPS, MFU={mfu:.0%}) ---")
    print(f"FLOPs/step:          {flops_per_step/1e9:.1f} GFLOP")
    print(f"Est. GPU step time:  {step_sec:.3f}s")
    print(f"100 steps ETA:       {step_sec*100/60:.1f} min")
    print(f"Full epoch ETA:      {step_sec*total_steps/3600:.1f} hours")
    print(f"total steps ETA:     {(total_steps):,} ")
    print(f"Total batches/epoch (bs=4): {total_steps}")
    print(f"Full epoch ETA:      {step_sec*total_steps/3600:.1f} hours")
    return step_sec


if __name__ == "__main__":
    #dataset = np.load(OUT_PATH / "TinyStoriesV2-GPT4-samples.npy")
    dataset = np.load(OUT_PATH / "OpenWebText-train.npy")
    model = TransformerModel(vocab_size, d_model, context_length, rope_theta, num_heads, d_ff, num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    params = sum(p.numel() for p in model.parameters())
    print(f"context length: {context_length}")
    print(f"batch size: {batch_size}")
    print(f"vocab size: {vocab_size}")
    print(f"d_model: {d_model}")
    print(f"d_ff: {d_ff}")
    print(f"num_layers: {num_layers}")
    print(f"num_heads: {num_heads}")
    print(f"rope_theta: {rope_theta}")
    print(f"Sample dataset size: {len(dataset):,} tokens")
    print(f"Params:  {params/1e6:.2f}M")
    #benchmark_cpu_training_step(dataset, model, optimizer)
    estimate_gpu_step_time(params, batch_size, context_length,
                       gpu_peak_tflops=8.9, mfu=0.35, total_steps=len(dataset) // batch_size)
