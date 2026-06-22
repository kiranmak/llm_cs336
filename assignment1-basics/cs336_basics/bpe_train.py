import sys
import json
import pathlib
import time

# Add the project root directory to python's import path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.pre_tokenizer import print_time
from cs336_basics.bpe_tokenizer import bpe_tokenizer_fn

DATA_PATH = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "out"
OUT_PATH.mkdir(parents=True, exist_ok=True)

def train_bpe_common(in_file, out_vocab_file, out_merges_file, vocab_size=1000, special_tokens=None):
    from tests.common import gpt2_bytes_to_unicode

    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    # Train BPE
    start_time = time.perf_counter()
    vocab, merges = bpe_tokenizer_fn(
        input_path=in_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    train_time = time.perf_counter() - start_time
    print_time("Training", train_time)
    print("  Vocabulary size:", len(vocab))
    print("  Merges size:", len(merges))

    # Convert bytes to GPT-2 unicode strings
    start_time = time.perf_counter()
    gpt2_unicode_map = gpt2_bytes_to_unicode()

    def bytes_to_unicode(b: bytes) -> str:
        return "".join(gpt2_unicode_map[x] for x in b)

    vocab_json = {bytes_to_unicode(token_bytes): token_index for token_index, token_bytes in vocab.items()}

    # Pre-convert merges to avoid repeated conversion during file write
    merges_unicode = [(bytes_to_unicode(a), bytes_to_unicode(b)) for a, b in merges]

    conversion_time = time.perf_counter() - start_time

    # Save vocab and merges to files
    start_time = time.perf_counter()
    with open(out_vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    with open(out_merges_file, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{a} {b}" for a, b in merges_unicode))

    save_time = time.perf_counter() - start_time

def train_bpe_expts_sample():
    train_bpe_common(
        in_file=       DATA_PATH / "owt_samples.txt",
        out_vocab_file = OUT_PATH / "owt_samples_vocab.json",
        out_merges_file= OUT_PATH / "owt_samples_merges.txt",
        vocab_size=300,
        special_tokens=["<|endoftext|>"]
    )


def train_bpe_tinystories():
    prefix="TinyStoriesV2-GPT4"
    train_bpe_common(
        in_file       =  DATA_PATH / f"{prefix}-train.txt",
        out_vocab_file=  OUT_PATH  / f"{prefix}_vocab.json",
        out_merges_file= OUT_PATH  / f"{prefix}_merges.txt",
        vocab_size     = 10000,
        special_tokens=["<|endoftext|>"]
    )

def train_bpe_expts_owt():
    prefix="OpenWebText"
    train_bpe_common(
        in_file       =  DATA_PATH / f"{prefix}-train.txt",
        out_vocab_file=  OUT_PATH  / f"{prefix}_vocab.json",
        out_merges_file= OUT_PATH  / f"{prefix}_merges.txt",
        vocab_size     = 32000,
        special_tokens=["<|endoftext|>"]
    )


if __name__ == "__main__":
    print("---Training BPE on TinyStories...---")
    train_bpe_tinystories()

    print("\n---Training BPE on OpenWebText---")
    train_bpe_expts_owt()
