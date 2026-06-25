import os, time, torch, numpy as np
from cs336_basics.nn_utils import get_batch, cross_entropy_loss
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel
from cs336_basics.paths import OUT_PATH

torch.set_num_threads(os.cpu_count() - 2)
device = torch.device('cpu')
batch_size, context_length, vocab_size = 64, 256, 10000
d_model, d_ff, num_layers, num_heads, rope_theta = 512, 1408, 4, 16, 10000

dataset = np.load(OUT_PATH / "TinyStoriesV2-GPT4-samples.npy")
model = TransformerModel(vocab_size, d_model, context_length, rope_theta, num_heads, d_ff, num_layers).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

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
print(f"CPU threads: {torch.get_num_threads()}")
print(f"Avg step time: {elapsed/n:.2f}s")
print(f"100 steps ETA: {elapsed/n*100/60:.1f} min")
print(f"Total batches/epoch (bs=64): {len(dataset)//batch_size}")
print(f"Full epoch ETA: {elapsed/n * (len(dataset)//batch_size) / 3600:.1f} hours")

# param count
params = sum(p.numel() for p in model.parameters())
print(f"Params: {params/1e6:.2f}M")