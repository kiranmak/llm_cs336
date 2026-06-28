import time
import os
from cs336_basics.paths import PROJECT_ROOT
from cs336_basics.bpe_tokenizer import train_bpe_on_corpus
from cs336_basics.pre_tokenizer import BPEPreTokenizer

def small_corpus_main():
    input_path = os.path.join(PROJECT_ROOT, "tests/fixtures/tinystories_sample_5M.txt")
    print(f"Pre-tokenizing {input_path}...")
    pretokenizer = BPEPreTokenizer(["<|endoftext|>"])
    corpus = pretokenizer.pre_tokenize_file(input_path, encode_it=True)

    print(f"Corpus unique words: {len(corpus)}")
    # Measure time for 1000 merges
    num_merges = 1000
    print(f"Running BPE training for {num_merges} merges...")
    start_time = time.perf_counter()
    train_bpe_on_corpus(corpus, num_merges)
    end_time = time.perf_counter()
    print(f"BPE training took {end_time - start_time:.4f} seconds.")


def large_corpus_main():
    input_path = os.path.join(PROJECT_ROOT, "data/TinyStoriesV2-GPT4-valid.txt")
    print(f"Reading {input_path}...")

    # Read a portion of the file to get around 30k-50k unique words
    temp_file = os.path.join(PROJECT_ROOT, "data/temp_benchmark.txt")
    with open(input_path, "r", encoding="utf-8") as f_in:
        lines = [f_in.readline() for _ in range(50000)]

    with open(temp_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(lines)

    print(f"Pre-tokenizing temp benchmark file...")
    pretokenizer = BPEPreTokenizer(["<|endoftext|>"])
    corpus = pretokenizer.pre_tokenize_file(temp_file, encode_it=True)

    print(f"Corpus unique words: {len(corpus)}")

    # Run BPE training for 5000 merges
    num_merges = 5000
    print(f"Running BPE training for {num_merges} merges...")
    start_time = time.perf_counter()
    train_bpe_on_corpus(corpus, num_merges)
    end_time = time.perf_counter()
    print(f"BPE training took {end_time - start_time:.4f} seconds.")

    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)


small_corpus_main()
#large_corpus_main()
