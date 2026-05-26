from collections import defaultdict, Counter
# Initial Setup
"""
corpus_keys = [
    (108, 111),       # "lo"
    (108, 111, 119),  # "low"
    (108, 111)        # "lo"
]
corpus_count = [ 2, 5, 1]
"""
corpus_keys = [
    (108, 111),       # "lo"
    (108, 111, 119, 108),# "lowl"
    (108, 111, 119),  # "low"
    (108, 111),       # "lo"
    (108, 111, 119, 113, 99), # "lower"
    (119, 111, 119),   # "wow"
]
corpus_count = [ 3, 2, 2, 2, 15, 1]
next_token_id = 256

def merge_word(word_ids: list[int], freq:int, pair: tuple[int, int], new_id: int) -> Counter:
    new_word_ids = Counter()
    i = 0
    while i < len(word_ids):
        # Check if the current element and the next element match our winning pair
        if i < len(word_ids) - 1 and word_ids[i] == pair[0] and word_ids[i+1] == pair[1]:
            new_word_ids[new_id] = freq
            i += 2  # Skip both elements because they are now merged!
        else:
            new_word_ids[word_ids[i]] = freq
            i += 1
    return new_word_ids

def show(cps: Counter):
    for word_ids, freq in cps.items():
        print(list(word_ids), ":", freq, end=", ")
    print("")

def bpe_tokenizer_simple(corpus: Counter, num_merges:int):
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)} # Base vocab (0-255)
    next_token_id = 256
    print(f"Original Corpus: {corpus}\n")
    for merge_round in range(num_merges):
        # 1. Count pairs within word boundaries
        #counts = defaultdict(int)
        counts = Counter()
        for word_ids, freq in corpus.items():
            for pair in zip(word_ids, word_ids[1:]):
                counts[pair] += freq

        print("counts are")
        show(counts)
        if not counts:
            break # No more pairs left to merge

        # 2. Pick the winner
        best_pair = max(counts, key=counts.get)

        # 3. Register the new token
        merges[best_pair] = next_token_id
        vocab[next_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # 4. Update the corpus in place
        new_corpus = [merge_word(wlist, freq, best_pair, next_token_id) for wlist, freq in corpus.items()]
        corpus = sum(new_corpus, Counter())


        print(f"Round {merge_round + 1}:\n Merged {best_pair} into {next_token_id}")
        print(f"Updated Corpus:\n {corpus}\n")

        next_token_id += 1
    return merges, corpus

num_merges = 3
corpus = Counter()
for key, count in zip(corpus_keys, corpus_count):
    corpus[key] += count

merges, corpus = bpe_tokenizer_simple(corpus, num_merges)
