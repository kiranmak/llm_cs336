import json
import sys
import pathlib

# Add the project root directory to python's import path before importing package modules
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from cs336_basics.bpe_train import DATA_PATH, OUT_PATH
from cs336_basics.bpe_tokenizer import BPETokenizer

# Add the project root directory to python's import path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOKENS_PATH = OUT_PATH

vocab_files  = ["TinyStories_vocab.json", "owt_vocab.json"]
merges_files = ["TinyStories_merges.txt", "owt_merges.txt"]
sample_files = ["TinyStoriesV2-GPT4-samples.txt","owt_samples.txt"]

def read_doc_chunk(reservoir, keyword):
    # Use a positive lookahead (?=...) for the second keyword
    pattern = f"{re.escape(keyword)}(.*?)(?={re.escape(keyword)})"
    matches = re.findall(pattern, reservoir, flags=re.DOTALL)
    print(matches)  # Output: ['target1', 'target2']
    return matches

def tokenizer_experiment_common(
        vocab_path: str, merge_path: str, doc_to_decode: str):
    spltok = ["<|endoftext|>"]
    tokenizer =  BPETokenizer.from_files(vocab_path, merge_path, spltok)

    with open(doc_to_decode, "r") as f:
        corpus_contents = f.read()

    read_doc_chunk(corpus_contents, spltok[0])
    #ids = tokenizer.encode(corpus_contents)
    #assert tokenizer.decode(ids) == corpus_contents

def tokenizer_experiments():
    #for i in range(len(vocab_files)):
    for i in range(1):
        tokenizer_experiment_common(
            vocab_path=   TOKENS_PATH / vocab_files[i],
            merge_path=   TOKENS_PATH /merges_files[i],
            doc_to_decode= DATA_PATH /   sample_files[i]
            )

if __name__ == "__main__":
    tokenizer_experiments()

