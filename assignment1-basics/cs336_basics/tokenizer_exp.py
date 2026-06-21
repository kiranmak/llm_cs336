from torch import _weights_only_unpickler
from pathlib import Path
import json
import sys
import pickle
import os
import multiprocessing
from array import array
import numpy as np

# Add the project root directory to python's import path before importing package modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from cs336_basics.bpe_train import DATA_PATH, OUT_PATH
from cs336_basics.bpe_tokenizer import BPETokenizer, get_compression_ratio
from cs336_basics.bpe_tokenizer import TinyStoriesTokenizer, OpenWebTextTokenizer

TOKENS_PATH = OUT_PATH


vocab_files  = ["TinyStories_vocab.json", "owt_vocab.json"]
merges_files = ["TinyStories_merges.txt", "owt_merges.txt"]
sample_files = ["TinyStoriesV2-GPT4-samples.txt","owt_samples.txt"]
dev_files = ["TinyStoriesV2-GPT4-valid.txt","owt_valid.txt"]
train_files = ["TinyStoriesV2-GPT4-train.txt","owt_train.txt"]

def encode_file(corpus: str, tokenizer: BPETokenizer, out_path: Path):
    import time
    t0 = time.time()
    global _worker_tokenizer
    _worker_tokenizer = tokenizer
    encoded_ids = encode_chunk(corpus)
    encoded_arr = np.array(encoded_ids, dtype=np.uint16)
    print("first 20 ids ", encoded_arr[:20]) 
    np.save(out_path, encoded_arr)
 
    total_tokens = len(encoded_ids)
    print(f"Encoding finished in {time.time() - t0:.1f}s. Total tokens: {total_tokens}")
    print(f"Saved to {out_path}")

def tokenizer_experiment_web():
    vocab_path = TOKENS_PATH / vocab_files[1]
    merge_path = TOKENS_PATH / merges_files[1]

    samples_path = DATA_PATH / sample_files[1]
    spltok = ["<|endoftext|>"]

    tokenizer =  OpenWebTextTokenizer.from_files(vocab_path, merge_path, spltok)

    with open(samples_path, "r") as f:
        corpus_contents = f.read()
    encode_file(corpus_contents, tokenizer, TOKENS_PATH / "owt_samples_ids.npy")

def tokenizer_experiment_tiny():
    vocab_path = TOKENS_PATH / vocab_files[0]
    merge_path = TOKENS_PATH / merges_files[0]
    spltok = ["<|endoftext|>"]

    tokenizer =  TinyStoriesTokenizer.from_files(vocab_path, merge_path, spltok)
    sample_path = DATA_PATH / sample_files[0]
    with open(sample_path, "r") as f:
        corpus_contents = f.read()
    encode_file(corpus_contents, tokenizer, TOKENS_PATH / "tinystories_samples_ids.npy")


_worker_tokenizer = None

def init_worker(vocab_path, merge_path, special_tokens, tokenizer_cls_name):
    global _worker_tokenizer
    from cs336_basics.bpe_tokenizer import TinyStoriesTokenizer, OpenWebTextTokenizer
    if tokenizer_cls_name == "TinyStoriesTokenizer":
        _worker_tokenizer = TinyStoriesTokenizer.from_files(vocab_path,
                                                            merge_path,
                                                            special_tokens)
    else:
        _worker_tokenizer = OpenWebTextTokenizer.from_files(vocab_path,
                                                            merge_path,
                                                            special_tokens)

def encode_chunk(lines):
    ids = []
    for line in lines:
        line = line.strip()
        if line:
            ids.extend(_worker_tokenizer.encode(line))
    return ids

def chunk_generator(filepath, chunk_size=50000):
    chunk = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def encode_file_parallel(txt_path: Path, vocab_path: Path,
                         merge_path: Path,
                         special_tokens: list[str],
                         tokenizer_cls_name: str,
                         out_path: Path, num_workers: int = None):
    import time
    print(f"Parallel encode {txt_path} -> {out_path} using {tokenizer_cls_name}.")
    t0 = time.time()

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} processes.")

    temp_bin_path = out_path.with_suffix(".bin")

    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(vocab_path, merge_path, special_tokens, tokenizer_cls_name)
    )

    total_tokens = 0
    chunks = chunk_generator(txt_path, chunk_size=100000)

    with open(temp_bin_path, "wb") as bin_f:
        for encoded_chunk in pool.imap(encode_chunk, chunks, chunksize=1):
            arr = array('H', encoded_chunk)
            arr.tofile(bin_f)
            total_tokens += len(encoded_chunk)

    pool.close()
    pool.join()

    print(f"Encoding finished in {time.time() - t0:.1f}s. Total tokens: {total_tokens}")
    print(f"Converting raw binary to .npy format...")

    t_conv = time.time()
    data = np.fromfile(temp_bin_path, dtype=np.uint16)
    np.save(out_path, data)

    os.remove(temp_bin_path)
    print(f"Conversion and save to {out_path} finished in {time.time() - t_conv:.1f}s.")

def main_file_encoder():
    # TinyStories
    encode_file_parallel(
        txt_path=DATA_PATH / train_files[0],
        vocab_path=TOKENS_PATH / vocab_files[0],
        merge_path=TOKENS_PATH / merges_files[0],
        special_tokens=["<|endoftext|>"],
        tokenizer_cls_name="TinyStoriesTokenizer",
        out_path=TOKENS_PATH / "tinystories_train_ids.npy"
    )
    encode_file_parallel(
        txt_path=DATA_PATH / dev_files[0],
        vocab_path=TOKENS_PATH / vocab_files[0],
        merge_path=TOKENS_PATH / merges_files[0],
        special_tokens=["<|endoftext|>"],
        tokenizer_cls_name="TinyStoriesTokenizer",
        out_path=TOKENS_PATH / "tinystories_dev_ids.npy"
    )

    # OpenWebText
    encode_file_parallel(
        txt_path=DATA_PATH / train_files[1],
        vocab_path=TOKENS_PATH / vocab_files[1],
        merge_path=TOKENS_PATH / merges_files[1],
        special_tokens=["<|endoftext|>"],
        tokenizer_cls_name="OpenWebTextTokenizer",
        out_path=TOKENS_PATH / "openwebtext_train_ids.npy"
    )
    encode_file_parallel(
        txt_path=DATA_PATH / dev_files[1],
        vocab_path=TOKENS_PATH / vocab_files[1],
        merge_path=TOKENS_PATH / merges_files[1],
        special_tokens=["<|endoftext|>"],
        tokenizer_cls_name="OpenWebTextTokenizer",
        out_path=TOKENS_PATH / "openwebtext_dev_ids.npy"
    )


if __name__ == "__main__":
    main_file_encoder()
    #tokenizer_experiment_tiny()
    #tokenizer_experiment_web()
