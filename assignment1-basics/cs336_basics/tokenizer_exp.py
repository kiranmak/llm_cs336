from pathlib import Path
import json
import sys
import time
import os
import multiprocessing
from array import array
import numpy as np

import re
from cs336_basics.pre_tokenizer import print_time
from cs336_basics.paths import PROJECT_ROOT, DATA_PATH, OUT_PATH
from cs336_basics.bpe_tokenizer import BPETokenizer, get_compression_ratio
DATASET_TYPES = ["valid", "train", "samples"]

def get_vocab_merge_fname(prefix, flagtype: str):
    if flagtype not in DATASET_TYPES:
        print(f"{flagtype} is Not ok input type of file")
        exit(1)

    f_roster = {}
    f_roster["text"]  = DATA_PATH / f"{prefix}-{flagtype}.txt"
    f_roster["vocab"] = OUT_PATH / f"{prefix}-train_vocab.json"
    f_roster["merge"] = OUT_PATH / f"{prefix}-train_merges.txt"
    f_roster["npy"]   = OUT_PATH / f"{prefix}-{flagtype}.npy"
    #print("------ FILES to work with-----")
    for k,v in f_roster.items():
        if k != "npy":
            try:
                if not v.is_file():
                    raise FileNotFoundError
            except FileNotFoundError:
                print(f"Error: The specified file {v} not found.")
                exit(1)
    return f_roster


_worker_tokenizer = None
def init_worker(vocab_path, merge_path, special_tokens):
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer.from_files(vocab_path,
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

def encode_file_parallel(txt_path: Path,
                         vocab_path: Path,
                         merge_path: Path,
                         special_tokens: list[str],
                         out_path: Path,
                         skip_npy=True,
                         num_workers: int = None):
    current_dir = Path.cwd()
    src = txt_path.relative_to(current_dir)
    dst = out_path.relative_to(current_dir)
    print(f"Parallel Encode {src} -> {dst}.")
    t0 = time.time()

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} processes.")

    temp_bin_path = out_path.with_suffix(".bin")

    pool = multiprocessing.Pool(
        processes   = num_workers,
        initializer = init_worker,
        initargs  =(vocab_path, merge_path, special_tokens)
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

    t_conv = time.time()
    if not skip_npy:
        data = np.fromfile(temp_bin_path, dtype=np.uint16)
        np.save(out_path, data)
        os.remove(temp_bin_path)

    print(f"Total Tokens: {total_tokens}")
    print_time("Encoding", t_conv - t0)

def main_file_encoder():
    print(f"====== Encoding text to NPY ======")
    #file_prefix = ["OpenWebText"]
    file_prefix = ["TinyStoriesV2-GPT4"]
    #file_prefix = ["TinyStoriesV2-GPT4", "OpenWebText"]
    for in_file in file_prefix:
        for dataset in DATASET_TYPES:
            roster = get_vocab_merge_fname(in_file, dataset)
            encode_file_parallel(
                txt_path   = roster["text"],
                vocab_path = roster["vocab"],
                merge_path = roster["merge"],
                special_tokens = ["<|endoftext|>"],
                out_path   = roster["npy"]
            )
    print(f"====== Encoding to NPY Finished ======")


def file_encode_bin_from_vocab_merges(dataset_prefix: str,
                                  dataset_type: str = "samples"):
    roster = get_vocab_merge_fname(dataset_prefix, dataset_type)
    encode_file_parallel(
        txt_path   = roster["text"],
        vocab_path = roster["vocab"],
        merge_path = roster["merge"],
        special_tokens = ["<|endoftext|>"],
        out_path   = roster["npy"],
        skip_npy=True,
    )


if __name__ == "__main__":
    #main_file_encoder()
    tokenfile = "TinyStoriesV2-GPT4"
    #tokenfile = "OpenWebText"
    print("Generating BIN file for the dataset...", tokenfile, "train")
    file_encode_bin_from_vocab_merges(tokenfile, "train")
