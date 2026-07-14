#!/bin/bash
for lr in 1e-4 2e-4 3e-4 4e-4 5e-4; do
    echo "Training with LR: $lr"
    uv run runs/traincmd.py --tokenfile TinyStoriesV2-GPT4 --maxsteps 1000 --batch_size 32 --contextlen 256 --evalinterval 50 --loginterval 50 --warmup 0.1 --cosine 1.0 --lr $lr
done