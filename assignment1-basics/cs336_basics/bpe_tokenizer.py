import os
from typing import Counter
import regex
import time
import datetime
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict, Counter
from itertools import pairwise
from pre_tokenizer import BPEPreTokenizer, load_pkl, print_time

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

def merge(word_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    new_word_ids = []
    i = 0
    while i < len(word_ids):
        # Check if the current element and the next element
        # match our winning pair
        if i < len(word_ids) -1 and\
           word_ids[i] == pair[0] and word_ids[i+1] == pair[1]:
            new_word_ids.append(new_id)
            i += 2  # Skip both elements because they are now merged!
        else:
            new_word_ids.append(word_ids[i])
            i += 1
    return new_word_ids

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index
    next: int # next index

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(string.encode("utf-8"))
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        byte_chunks = [self.params.vocab[i] for i in indices]
        return b"".join(byte_chunks).decode("utf-8")

def get_compression_ratio(string: str, indices: list[int]) -> float:
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens



def train_bpe_on_corpus(pre_tokens: Counter,
                        num_merges: int) -> BPETokenizerParams:

    # pre_tokens are already utf-encoded. Just get the index
    # index1, index2 => merged index
    merges: dict[tuple[int, int], int] = {}
    # index -> bytes
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    next_token_id = 256
    #print(f"Original Corpus: {pre_tokens}\n")
    for m in range(num_merges):
        # 1. Count pairs within word boundaries
        counts = defaultdict(int)
        for word_ids, freq in pre_tokens.items():
            for pair in zip(word_ids, word_ids[1:]):
                counts[pair] += freq

        if not counts:
            break # No more pairs left to merge

        # 2. Pick the winner
        best_pair = max(counts, key=counts.get)

        # 3. Register the new token
        merges[best_pair] = next_token_id
        vocab[next_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # 4. Update the corpus in place
        updated: Counter = Counter()
        for word_ids, freq in pre_tokens.items():
            merged = tuple(merge(list(word_ids), best_pair, next_token_id))
            updated[merged] += freq
        pre_tokens = updated

        #print(f"Round {m + 1}: Merged {best_pair} into {next_token_id}")
        #print(f"Updated Corpus: {pre_tokens}\n")

        next_token_id += 1

    return BPETokenizerParams( vocab=vocab, merges=merges,)

# Char-atomic vocab: single-char str -> UTF-8 bytes; merged tokens use int ids (256+).
def token_to_bytes(vocab: dict, token: str | int | bytes) -> bytes:
    if isinstance(token, bytes):
        return token
    return vocab[token]


def merge_bytes(
    word_ids: list[str | bytes],
    pair: tuple[str | bytes, str | bytes],
    new_token: bytes,
) -> list[str | bytes]:
    """Like merge(), but appends the merged token's bytes instead of an int id."""
    new_word_ids: list[str | bytes] = []
    i = 0
    while i < len(word_ids):
        if (
            i < len(word_ids) - 1
            and word_ids[i] == pair[0]
            and word_ids[i + 1] == pair[1]
        ):
            new_word_ids.append(new_token)
            i += 2
        else:
            new_word_ids.append(word_ids[i])
            i += 1
    return new_word_ids


def train_bpe_on_byte_corpus(pre_tokens: Counter, params) -> BPETokenizerParams:

    merges = params.merges
    vocab = params.vocab
    next_token_id = params.next
    #print(f"Original Corpus: {pre_tokens}\n")
    # 1. Count pairs within word boundaries
    counts = defaultdict(int)
    for word_ids, freq in pre_tokens.items():
        for pair in zip(word_ids, word_ids[1:]):
            counts[pair] += freq

    if counts:
        # 2. Pick the winner
        best_pair = max(counts, key=counts.get)

        # 3. Register the new token
        merges[best_pair] = next_token_id
        vocab[next_token_id] = (
            token_to_bytes(vocab, best_pair[0]) + token_to_bytes(vocab, best_pair[1])
        )

        # 4. Update the corpus in place
        updated: Counter = Counter()
        new_token = vocab[next_token_id]
        for word_ids, freq in pre_tokens.items():
            merged = tuple(merge_bytes(list(word_ids), best_pair, new_token))
            updated[merged] += freq
        pre_tokens = updated

        #print(f"Round {m + 1}: Merged {best_pair} into {next_token_id}")
        #print(f"Updated Corpus: {pre_tokens}\n")

        next_token_id += 1

    return pre_tokens, BPETokenizerParams( vocab=vocab, merges=merges, next=next_token_id)

def bpe_tokenizer_fn(input_path, vocab_size, special_tokens):

    pretokenizer = BPEPreTokenizer(special_tokens)
    corpus = pretokenizer.pre_tokenize_file(input_path)
    params = BPETokenizerParams(
        merges={},
        vocab={chr(x): bytes([x]) for x in range(vocab_size)},
        next=256,
    )

    start_time = time.perf_counter()  # Before
    corpus, params = train_bpe_on_byte_corpus(corpus, params)
    end_time = time.perf_counter()    # After

    print_time("      TOKENS",  end_time - start_time)
    print(f"Merged {len(params.merges)} Vocab {len(params.vocab)}")
    return params.vocab, params.merges

if __name__ == "__main__":

    owt_file = """./data/owt_train.txt"""
    samples_file = "./data/test_samples1.txt"
    tiny_stories_file = """../data/TinyStoriesV2-GPT4-train.txt"""
    bpe_tokenizer_fn(samples_file, num_merges=3)
