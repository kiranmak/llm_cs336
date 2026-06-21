from cs336_basics import pre_tokenizer
import os, time
import pathlib
import sys
import json
from collections.abc import Iterable
from typing import Counter
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict, Counter

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

        #  Ensure we have a valid <unk> token id.
        unk_token = "<unk>"
        self.unk_id = None
        for tid, tbytes in self.params.vocab.items():
            try:
                if tbytes.decode("utf-8") == unk_token:
                    self.unk_id = tid
                    break
            except UnicodeDecodeError:
                continue
        if self.unk_id is None:
            if self.params.vocab:
                self.unk_id = max(self.params.vocab) + 1
            else:
                self.unk_id = 0
            self.params.vocab[self.unk_id] = unk_token.encode("utf-8")
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
            token_indices = [
                self.byte_to_token_id.get(b, self.unk_id)
                for b in token_str.encode("utf-8")
                ]

            while len(token_indices) >= 2:
            # Find the adjacent pair with the lowest merge rank
                best_pair = None
                best_rank = float("inf")
                for i in range(len(token_indices) - 1):
                    pair = (token_indices[i], token_indices[i+1])
                    if pair in self.params.merges:
                        rank = self.params.merges[pair]
                        if rank < best_rank:
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

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges
        #  (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        from tests.common import gpt2_bytes_to_unicode

        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)

        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

        vocab:dict[int, bytes] = {}
        for token_str, token_id in gpt2_vocab.items():
            token_bytes = bytes(gpt2_byte_decoder[ch] for ch in token_str)
            vocab[token_id] = token_bytes

        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = {}
        for merge_token_1, merge_token_2 in gpt2_bpe_merges:
            id1 = gpt2_vocab.get(merge_token_1)
            id2 = gpt2_vocab.get(merge_token_2)
            id_new = gpt2_vocab.get(merge_token_1 + merge_token_2)
            if id1 is None or id2 is None or id_new is None:
                print(f"Warning: Could not find merge IDs for {merge_token_1} {merge_token_2}")
                continue
            merges[(id1, id2)] = id_new

        return cls(BPETokenizerParams(vocab, merges))

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

def train_bpe_on_byte_corpus(pre_tokens: Counter,
                             params:BPETokenizerParams,
                             next_id) -> tuple[int, Counter]:

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
        # 2. Pick the winner, breaking ties lexicographically by byte values
        #    of the tokens
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
                merged_ids = tuple(merge(list(word_ids),
                                         best_pair, next_token_id)
                                  )
                next_pre_tokens[merged_ids] += freq
            else:
                next_pre_tokens[word_ids] += freq

    pre_tokens = next_pre_tokens
    next_token_id += 1

    return next_token_id, pre_tokens

def train_bpe_on_corpus(pre_tokens: Counter,
                        num_merges: int) -> BPETokenizerParams:
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
    from collections import Counter, defaultdict
    from tqdm import tqdm  # Import the progress bar module

    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    #------------------------------------------------------
    # 1. Build the inverse lookup index of pairs to words
    # from pre_tokens, create pair based dictionary and 
    # track which words a pair is associated to.
    # This may take heck of my RAM
    #------------------------------------------------------
    pair_counts = defaultdict(int)
    # to keep track of words that should be changed when pair count changes
    pair_to_words = defaultdict(lambda: defaultdict(int))
    corpus_words = dict(pre_tokens)

    for word_tuple, freq in corpus_words.items():
        for i in range(len(word_tuple) -1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair][word_tuple] +=1

    next_token_id = 256
    # desc adds a label, and leave=True keeps the finished bar on the screen.
    progress_bar = tqdm(range(num_merges), desc="Training BPE", leave=True)

    #------------------------------------------------------
    # 1. This is where speed optimization is done
    #------------------------------------------------------
    for m in progress_bar:

        # we will keep trimming pair_counts
        if not pair_counts:
            progress_bar.set_postfix_str("No more pairs left to merge.")
            break

        # 2. Pick the winner - pair_counts contain unique pair.
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]])
        )

        if pair_counts[best_pair] <= 0:
            break

        # 3. Register the new token
        merges[best_pair] = next_token_id
        vocab[next_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        if m % 100 == 0:
            try:
               token_str = vocab[next_token_id].decode("utf-8")
            except UnicodeDecodeError:
               token_str = f"bytes({list(vocab[next_token_id])})"

            progress_bar.set_postfix({
                "freq": pair_counts[best_pair],
                "unique_words": len(pre_tokens)
            })

        # Dynamic stats printed right next to the progress bar
        """
        progress_bar.set_postfix({
            "token_id": next_token_id,
            "merged": f"'{token_str}'",
            "freq": counts[best_pair],
            "unique_words": len(pre_tokens)
        })
        """
        # track all the words with best pair
        affected_words = pair_to_words[best_pair]

        del pair_counts[best_pair] # remove winning pair and trim
        del pair_to_words[best_pair] # also dict of dict

        # ---------------------------------------------------
        # 3. Mutate the dictionary in-place by localized
        # increment and decrements
        # ---------------------------------------------------

        for old_words, pair_occurences_in_word in list(affected_words.items()):
            if old_words not in corpus_words:
                continue

            freq = corpus_words.pop(old_words)
            # A. Step through the word and DECREMENT global counts for all pairs
            # that are about to disappear because of the merge
            for i in range(len(old_words)-1):
                pr = (old_words[i], old_words[i+1])
                if pr in pair_counts:
                    count_to_remove = pair_occurences_in_word * freq
                    pair_counts[pr] -= freq
                    pair_to_words[pr][old_words] -= 1
                    if pair_to_words[pr][old_words] <= 0:
                        del pair_to_words[pr][old_words]
                    if pair_counts[pr] <= 0:
                        del pair_counts[pr]

            # B. Construct the newly merged word structure
            new_word_list = []
            i = 0
            while i < len(old_words):
                if i < len(old_words) - 1\
                        and old_words[i] == best_pair[0]\
                        and old_words[i+1] == best_pair[1]:
                    new_word_list.append(next_token_id)
                    i += 2
                else:
                    new_word_list.append(old_words[i])
                    i += 1
            new_word_tuple = tuple(new_word_list)

            # add new word to local tracker
            corpus_words[new_word_tuple] = freq

            # increment global counts for new pairs created by merge
            local_new_pairs = defaultdict(int)
            for j in range(len(new_word_tuple)-1):
                local_new_pairs[(new_word_tuple[j], new_word_tuple[j+1])] +=1

            for pr, local_count in local_new_pairs.items():
                 added_freq = local_count * freq
                 pair_counts[pr] += added_freq
                 pair_to_words[pr][new_word_tuple] += local_count

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
    params = train_bpe_on_corpus(corpus, num_merges)
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

class TinyStoriesTokenizer (BPETokenizer):
    pass

class OpenWebTextTokenizer (BPETokenizer):
    pass


if __name__ == "__main__":

    owt_file = """./data/owt_train.txt"""
    samples_file = "./data/owt_samples.txt"
    tiny_stories_file = """../data/TinyStoriesV2-GPT4-train.txt"""
    vocab, merges = bpe_tokenizer_fn(samples_file, vocab_size=259,
                                     special_tokens=["<|endoftext|>"])
