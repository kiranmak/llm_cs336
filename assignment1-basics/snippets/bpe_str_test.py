from collections import Counter
from cs336_basics.bpe_tokenizer import BPETokenizer, BPETokenizerParams
from cs336_basics.bpe_tokenizer import train_bpe_on_corpus, print_time, train_bpe_on_byte_corpus
from cs336_basics.pre_tokenizer import BPEPreTokenizer



def bpe_tokenizer_str_fn(input_text, vocab_size, special_tokens):
    next_id = 256
    pretokenizer = BPEPreTokenizer(special_tokens)
    corpus = pretokenizer.pre_tokenize_str(input_text)

    num_merges = vocab_size - 256 - len(special_tokens)

    params = BPETokenizerParams(
        merges={},
        vocab={x: bytes([x]) for x in range(256)},
    )
    for _ in range(num_merges):
        next_id, corpus = train_bpe_on_byte_corpus(corpus, params, next_id)

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

    return BPETokenizerParams(vocab, merges_list)


if __name__ == "__main__":
    test_text = """low low low low low\
        lower lower widest widest widest\
        newest newest newest newest newest newest"""

    special_tokens_test = ["<|endoftext|>", " ", "[PAD]"]
    params = bpe_tokenizer_str_fn(test_text, vocab_size=273,
                                  special_tokens = special_tokens_test)

    tokenizer = BPETokenizer(params)
    print("V", len(params.vocab), "M", len(params.merges))
    for m in params.merges:
        print(m)


