#!/bin/bash
batch_size_experiment() {
    tokens_to_process=40000000
    context_len=256
    for bs in 32 64 128 256; do
        msteps=$((tokens_to_process / (bs * context_len)))
        echo "Training with Batch Size: $bs and maxsteps $msteps"
        uv run runs/traincmd.py --tokenfile TinyStoriesV2-GPT4 --maxsteps $msteps\
            --batch_size $bs --contextlen 256 --evalinterval 50 --loginterval 50 --warmup 0.1 --cosine 1.0 --lr 1e-4
    done
}

learning_rate() {
    for lr in 1e-4 2e-4 3e-4 4e-4 5e-4; do
        echo "Training with Batch Size: $bs"
        uv run runs/traincmd.py --tokenfile TinyStoriesV2-GPT4 --maxsteps 1000 --batch_size $bs --contextlen 256 --evalinterval 50 --loginterval 50 --warmup 0.1 --cosine 1.0 --lr $lr
    done
}
batch_size_experiment
#learning_rate
