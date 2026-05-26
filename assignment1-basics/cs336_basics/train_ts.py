import sys
import json
import pathlib
import time
import json

# Add the project root directory to python's import path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.adapters import run_train_bpe
from cs336_basics.pre_tokenizer import print_time

DATA_PATH = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "out"

def train_bpe_common(in_file, out_vocab_file, out_merges_file, vocab_size=50257, special_tokens=["<|endoftext|>"]):   
    from tests.common import gpt2_bytes_to_unicode
    
    start_time = time.perf_counter()  # Before
    vocab, merges = run_train_bpe(
        input_path=in_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    print("Vocabulary size:", len(vocab))
    print("Merges size:", len(merges))

    # Convert bytes to GPT-2 unicode strings
    gpt2_unicode_map = gpt2_bytes_to_unicode()
    def bytes_to_unicode(b: bytes) -> str:
        return "".join(gpt2_unicode_map[x] for x in b)

    vocab_json = {bytes_to_unicode(token_bytes): token_index for token_index, token_bytes in vocab.items()}

    end_time = time.perf_counter()  # After
    print("Elapsed Time:", print_time("Proc Done", end_time - start_time))
    start_time = time.perf_counter()  # Before
    # save vocab and merges to files

    with open(out_vocab_file, "w", encoding="utf-8") as f:
        # Standard library JSON is fully compatible everywhere!
        json.dump(vocab_json, f, ensure_ascii=False)

    with open(out_merges_file, "w", encoding="utf-8") as f:
        f.write("\n".join([f"{bytes_to_unicode(a)} {bytes_to_unicode(b)}" for a, b in merges]))

    end_time = time.perf_counter()  # After
    print("Elapsed Time:", print_time("Sav Done", end_time - start_time))

    
def train_bpe_tinystories():
    train_bpe_common(
        in_file=DATA_PATH / "TinyStoriesV2-GPT4-train.txt",
        out_vocab_file=OUT_PATH / "TinyStories_vocab.json",
        out_merges_file=OUT_PATH / "TinyStories_merges.txt",
    )

def train_bpe_expts_owt():
    train_bpe_common(
        in_file=DATA_PATH / "owt_train.txt",
        out_vocab_file=OUT_PATH / "owt_vocab.json",
        out_merges_file=OUT_PATH / "owt_merges.txt",
    )


if __name__ == "__main__":
    print("---Training BPE on TinyStories...---")
    train_bpe_tinystories()

    print("\n---Training BPE on OpenWebText---")
    train_bpe_expts_owt()
