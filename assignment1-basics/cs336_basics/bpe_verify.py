"""Verification helpers for BPE training output."""

import time
from collections import Counter
try:
    from .bpe_tokenizer import BPETokenizer, BPETokenizerParams, merge
    from .bpe_tokenizer import train_bpe_on_corpus, bpe_tokenizer_fn, print_time
    from .pre_tokenizer import load_pkl
except ImportError:
    from bpe_tokenizer import BPETokenizer, BPETokenizerParams, merge
    from bpe_tokenizer import train_bpe_on_corpus, bpe_tokenizer_fn, print_time
    # pyrefly: ignore [missing-import]
    from pre_tokenizer import load_pkl

def _token_ids_to_bytes(word_ids: tuple[int, ...], vocab: dict[int, bytes]) -> bytes:
    return b"".join(vocab[i] for i in word_ids)


def _replay_merges_on_corpus(
    corpus: Counter,
    merges: dict[tuple[int, int], int],
    num_steps: int,
) -> Counter:
    """Apply the first num_steps merges (in insertion order) to a pre-token Counter."""
    updated = Counter(corpus)
    for step, (pair, new_id) in enumerate(merges.items()):
        if step >= num_steps:
            break
        next_corpus: Counter = Counter()
        for word_ids, freq in updated.items():
            merged = tuple(merge(list(word_ids), pair, new_id))
            next_corpus[merged] += freq
        updated = next_corpus
    return updated


def verify_params(
    params: BPETokenizerParams,
    original_corpus: Counter,
    num_merges: int,
    *,
    tokenizer: BPETokenizer | None = None,
    roundtrip_strings: list[str] | None = None,
    expected_merges: dict[tuple[int, int], int] | None = None,
) -> None:
    """
    Structural checks for BPE training output. Raises AssertionError on failure.
    """
    vocab = params.vocab
    merges = params.merges

    # --- 1. Byte vocabulary ---
    for i in range(256):
        assert vocab[i] == bytes([i]), f"vocab[{i}] should be single byte, got {vocab[i]!r}"

    # --- 2. Merge IDs and merge bytes ---
    expected_ids = list(range(256, 256 + len(merges)))
    actual_ids = list(merges.values())
    assert actual_ids == expected_ids, (
        f"merge token ids should be {expected_ids}, got {actual_ids}"
    )
    for (left, right), new_id in merges.items():
        assert left in vocab and right in vocab, (
            f"merge ({left}, {right}) references missing vocab ids"
        )
        assert vocab[new_id] == vocab[left] + vocab[right], (
            f"vocab[{new_id}] = {vocab[new_id]!r} != "
            f"vocab[{left}] + vocab[{right}] = {(vocab[left] + vocab[right])!r}"
        )

    if expected_merges is not None:
        assert merges == expected_merges, (
            f"merges mismatch:\n  got {merges}\n  expected {expected_merges}"
        )

    # --- 3. Corpus replay matches training (merge-order consistency) ---
    replayed = _replay_merges_on_corpus(original_corpus, merges, num_merges)
    for word_ids, _freq in replayed.items():
        assert all(tid in vocab for tid in word_ids), f"unknown token id in {word_ids}"
    assert len(merges) <= num_merges

    def weighted_byte_mass(corpus: Counter) -> int:
        return sum(
            len(_token_ids_to_bytes(word_ids, vocab)) * freq
            for word_ids, freq in corpus.items()
        )

    assert weighted_byte_mass(original_corpus) == weighted_byte_mass(replayed), (
        "total bytes in corpus (weighted by freq) changed after replaying merges"
    )
    for word_ids, _freq in replayed.items():
        original_bytes = _token_ids_to_bytes(word_ids, vocab)
        try:
            original_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise AssertionError(
                f"replayed word {word_ids} does not form valid UTF-8: {original_bytes!r}"
            ) from e

    # --- 4. Optional encode/decode roundtrip ---
    if tokenizer is not None and roundtrip_strings:
        for s in roundtrip_strings:
            encoded = tokenizer.encode(s)
            decoded = tokenizer.decode(encoded)
            assert decoded == s, f"roundtrip failed for {s!r}: got {decoded!r}"

    print(
        f"verify_params: OK ({len(merges)} merges, "
        f"{len(vocab)} vocab entries, "
        f"{len(replayed)} distinct words after replay)"
    )

def verify_toy_corpus(num_merges):
    # Toy corpus with known gold merges
    toy_corpus = Counter({
        (108, 111): 3,         # "lo"
        (108, 111, 119): 6,    # "low"
        (32, 108, 111): 2,     # " lo"
    })
    toy_original = Counter(toy_corpus)
    toy_params = train_bpe_on_corpus(toy_corpus, num_merges=2)
    verify_params(
        toy_params,
        toy_original,
        num_merges=2,
        tokenizer=BPETokenizer(toy_params),
        roundtrip_strings=["lo", "low", " lo"],
        expected_merges={(108, 111): 256, (256, 119): 257},
    )

def verify_pkl_corpus(pkl_file, num_merges):
    #corpus = load_pkl("cs336_basics/out_test_samples.txt.pkl")
    """
    pre_tokenize takes a long time, so if you are happy with
    it, save pre_tokens as pkl and then run tokenizer.
    owt_file = "../data/owt_train.txt"
    """
    print(f"Start training corpus from pkl:", pkl_file)
    start_time = time.perf_counter()  # Before
    corpus = load_pkl(pkl_file)
    original_corpus = Counter(corpus)
    end_time = time.perf_counter()    # After
    print_time(" Finished reading ",  end_time - start_time)
    start_time = time.perf_counter()  # Before
    params = train_bpe_on_corpus(corpus, num_merges=num_merges)
    end_time = time.perf_counter()    # After
    print_time(" Finished training ",  end_time - start_time)
    start_time = time.perf_counter()  # Before
    verify_params(
        params,
        original_corpus,
        num_merges=num_merges,
        tokenizer=BPETokenizer(params),
        roundtrip_strings=["hello", "world", "tokenizer"],
    )
    end_time = time.perf_counter()    # After
    print_time(" Finished verifying ", end_time - start_time)


def verify_and_create_corpus(train_file, num_merges):
    """
    pre_tokenize and then merge. one shot.
    owt_file = "../data/owt_train.txt"
    """
    original_corpus, params = bpe_tokenizer_fn(train_file, num_merges)
    verify_params(
        params,
        original_corpus,
        num_merges,
        tokenizer=BPETokenizer(params),
        roundtrip_strings=["hello", "world", "tokenizer"],
    )
if __name__ == "__main__":
    num_merges = 3
    #owt_file = """../data/owt_train.txt"""
    #test_file = "test_samples.txt"
    # Option 1: Use one shot
    """
    tiny_stories_file = "../data/TinyStoriesV2-GPT4-train.txt"
    verify_and_create_corpus(tiny_stories_file, num_merges)
    """
    # Option 2: read pkl if it was already created
    pkl_file = "../out/owt_train.pkl"
    verify_pkl_corpus(pkl_file, num_merges)
