import os, time
from collections.abc import Iterable
from typing import Counter
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict, Counter
try:
    from .pre_tokenizer import BPEPreTokenizer, print_time
except ImportError:
    from pre_tokenizer import BPEPreTokenizer, print_time

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

def merge2(word_ids: list[int],
            pair: tuple[int, int], new_id: int) -> list[int]:
    new_word_ids = []
    i = 0

    while i < len(word_ids):
        # Check if the current element and the next element
        # match our winning pair
        if i < len(word_ids) - 1\
            and word_ids[i] == pair[0]\
            and word_ids[i+1] == pair[1]:
            new_word_ids.append(new_id)
            i += 2  # Skip both elements as they are now merged!
        else:
            new_word_ids.append(word_ids[i])
            i += 1
    return new_word_ids

def merge(word_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace every non-overlapping occurrence of `pair` with `new_id`
    in one left-to-right pass. Shrinks `indices` in place; no extra list.
    """
    n = len(word_ids)
    if n < 2:
        return word_ids
    p0, p1 = pair
    read = write = 0
    idx = word_ids
    while read < n:
        if read + 1 < n and idx[read] == p0 and idx[read + 1] == p1:
            idx[write] = new_id
            write += 1
            read += 2
        else:
            idx[write] = idx[read]
            write += 1
            read += 1
    del idx[write:]
    return word_ids

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams, special_tokens: list[str] | None = None):
        self.params = params
        self.special_tokens = special_tokens or []
        self.special_token_to_id = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            for k, v in self.params.vocab.items():
                if v == token_bytes:
                    self.special_token_to_id[token] = k
                    break

        # Map each byte value (0..255) to its base token ID in vocab.
        # If there are duplicates, we ignore special tokens and prefer smaller IDs.
        self.byte_to_token_id = {}
        for k, v in self.params.vocab.items():
            if len(v) == 1:
                try:
                    decoded = v.decode("utf-8")
                    if decoded in self.special_tokens:
                        continue
                except UnicodeDecodeError:
                    pass

                b = v[0]
                if b in self.byte_to_token_id:
                    self.byte_to_token_id[b] = min(self.byte_to_token_id[b], k)
                else:
                    self.byte_to_token_id[b] = k

        import regex
        self.bpe_regex = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        if self.special_tokens:
            # Sort special tokens by length in descending order to match longer tokens first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [regex.escape(t) for t in sorted_special_tokens]
            self.special_split_regex = regex.compile(f"({'|'.join(escaped_tokens)})")

    def encode_piece2(self, piece: str) -> list[int]:
        indices = []
        for match in self.bpe_regex.finditer(piece):
            token_str = match.group()
            token_indices = [self.byte_to_token_id[b] for b in token_str.encode("utf-8")]
            for pair, new_index in self.params.merges.items():
                token_indices = merge(token_indices, pair, new_index)
            indices.extend(token_indices)
        return indices

    def encode_piece(self, piece: str) -> list[int]:
        """
        This function implements the logic for encoding a single token in a BPE tokenizer.
        it is rank based BPE. Instead of looping all merges,
        we only consider the best pair at each step.
        TODO: understand better - rank is freq of word. This algo improves from O(merges)
        to O(vocab)
        """
        indices = []
        for match in self.bpe_regex.finditer(piece):
            token_str = match.group()
            token_indices = [self.byte_to_token_id[b] for b in token_str.encode("utf-8")]

            while len(token_indices) >= 2:
            # Find the adjacent pair with the lowest merge rank
                best_pair = None
                best_rank = float('inf')
                for i in range(len(token_indices) - 1):
                    pair = (token_indices[i], token_indices[i+1])
                    rank = self.params.merges.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair = pair

                if best_pair is None:
                    break

                # Merge the best pair
                token_indices = merge(token_indices, best_pair, best_rank)

            indices.extend(token_indices)
        return indices

    def encode(self, string: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_piece(string)

        pieces = self.special_split_regex.split(string)
        indices = []
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_token_to_id:
                indices.append(self.special_token_to_id[piece])
            else:
                indices.extend(self.encode_piece(piece))
        return indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, indices: list[int]) -> str:
        byte_chunks = [self.params.vocab[i] for i in indices]
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

def get_compression_ratio(string: str, indices: list[int]) -> float:
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens

def train_bpe_on_byte_corpus(pre_tokens: Counter, params:BPETokenizerParams, next_id) -> tuple[int, Counter]:

    # pre_tokens are already utf-encoded. Just get the index
    # index1, index2 => merged index
    vocab = params.vocab
    merges = params.merges

    next_token_id = next_id
    # 1. Count pairs within word boundaries
    counts = defaultdict(int)
    for word_ids, freq in pre_tokens.items():
        for pair in zip(word_ids, word_ids[1:]):
            counts[pair] += freq

    if counts:
        # 2. Pick the winner, breaking ties lexicographically by byte values of the tokens
        best_pair = max(
            counts,
            key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]])
        )

        # 3. Register the new token
        merges[best_pair] = next_token_id
        vocab[next_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # 4. Update the corpus
        next_pre_tokens = Counter()
        for word_ids, freq in pre_tokens.items():
            if best_pair[0] in word_ids and best_pair[1] in word_ids:
                merged_ids = tuple(merge(list(word_ids), best_pair, next_token_id))
                next_pre_tokens[merged_ids] += freq
            else:
                next_pre_tokens[word_ids] += freq

    pre_tokens = next_pre_tokens
    next_token_id += 1

    return next_token_id, pre_tokens

def train_bpe_on_corpus(pre_tokens: Counter,
                        num_merges: int,
                        min_pair_count: int = 1,
                        verbose: bool = False) -> BPETokenizerParams:
    """
    Train BPE tokenizer on a corpus.

    Args:
        pre_tokens: Counter of pre-tokenized words
        num_merges: Number of merge operations to perform
        min_pair_count: Skip merging pairs with count below this threshold (default: 1)
        verbose: Print progress information (default: False)

    Returns:
        BPETokenizerParams with trained vocab and merges
    """
    # pre_tokens are already utf-encoded. Just get the index
    # index1, index2 => merged index
    merges: dict[tuple[int, int], int] = {}
    # index -> bytes
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    next_token_id = 256
    for m in range(num_merges):
        # 1. Count pairs within word boundaries
        counts = defaultdict(int)
        for word_ids, freq in pre_tokens.items():
            for pair in zip(word_ids, word_ids[1:]):
                counts[pair] += freq

        if not counts:
            break  # No more pairs left to merge

        # 2. Pick the winner, breaking ties lexicographically by byte values of the tokens
        best_pair = max(
            counts,
            key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]])
        )

        # 3. Register the new token
        merges[best_pair] = next_token_id
        vocab[next_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # 4. Update the corpus
        next_pre_tokens = Counter()
        for word_ids, freq in pre_tokens.items():
            if best_pair[0] in word_ids and best_pair[1] in word_ids:
                merged_ids = tuple(merge(list(word_ids), best_pair, next_token_id))
                next_pre_tokens[merged_ids] += freq
            else:
                next_pre_tokens[word_ids] += freq
        pre_tokens = next_pre_tokens

        next_token_id += 1

    return BPETokenizerParams(vocab=vocab, merges=merges)

def bpe_tokenizer_fn(input_path, vocab_size, special_tokens):
    print()
    x_time = time.perf_counter()
    pretokenizer = BPEPreTokenizer(special_tokens)
    corpus = pretokenizer.pre_tokenize_file(input_path,
                                            encode_it=True)

    num_merges = vocab_size - 256 - len(special_tokens)

    start_time = time.perf_counter()
    params = train_bpe_on_corpus(corpus, num_merges, min_pair_count=1)
    end_time = time.perf_counter()    # After
    print_time("   TRAIN", end_time - start_time)
    start_time = time.perf_counter()

    # Format merges as list of bytes tuples ordered by creation
    # need this for testing. eg. change  {(32, 116): 256} to [(' ', 'b'), ...]
    merges_list = []
    for (left, right) in params.merges.keys():
        merges_list.append((params.vocab[left], params.vocab[right]))

    # Add special tokens to vocab
    vocab = params.vocab.copy()
    next_id = 256 + len(params.merges)
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    end_time = time.perf_counter()    # After
    print_time("   Format", end_time - start_time)
    print_time("   Total", end_time - x_time)
    return vocab, merges_list


if __name__ == "__main__":

    owt_file = """./data/owt_train.txt"""
    samples_file = "./data/test_samples1.txt"
    tiny_stories_file = """../data/TinyStoriesV2-GPT4-train.txt"""
    vocab, merges = bpe_tokenizer_fn(samples_file, vocab_size=259)
