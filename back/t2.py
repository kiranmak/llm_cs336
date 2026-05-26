# Initial Setup
from collections import defaultdict

def merge_word(word_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    new_word_ids = []
    i = 0
    while i < len(word_ids):
        # Check if the current element and the next element match our winning pair
        if i < len(word_ids) - 1 and word_ids[i] == pair[0] and word_ids[i+1] == pair[1]:
            new_word_ids.append(new_id)
            i += 2  # Skip both elements because they are now merged!
        else:
            new_word_ids.append(word_ids[i])
            i += 1
    return new_word_ids

corpus = [
    [108, 111],       # "lo"
    [108, 111, 119, 108],# "lowl"
    [108, 111, 119],  # "low"
    [108, 111],       # "lo"
    [108, 111, 119, 113, 99], # "lower"
    [119, 111, 119]   # "wow"
]

vocab = {i: bytes([i]) for i in range(256)} # Base vocab (0-255)
merges = {}
num_merges = 4
next_token_id = 256

for merge_round in range(num_merges):
    # 1. Count pairs within word boundaries
    counts = defaultdict(int)
    for word_ids in corpus:
        for pair in zip(word_ids, word_ids[1:]):
            counts[pair] += 1

    if not counts:
        print(f"Round {merge_round + 1}: Empty count for next token {next_token_id}")
        break # No more pairs left to merge

    # 2. Pick the winner
    best_pair = max(counts, key=counts.get)

    # 3. Register the new token
    merges[best_pair] = next_token_id
    vocab[next_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

    # 4. Update the corpus in place
    corpus = [merge_word(word_ids, best_pair, next_token_id) for word_ids in corpus]

    print(f"Round {merge_round + 1}: Merged {best_pair} into {next_token_id}")
    print(f"Updated Corpus: {corpus}\n")

    next_token_id += 1

