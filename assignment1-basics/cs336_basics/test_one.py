from collections import Counter
try:
    from .bpe_tokenizer import BPETokenizer, BPETokenizerParams
    from .bpe_tokenizer import train_bpe_on_corpus, print_time, train_bpe_on_byte_corpus
    from .pre_tokenizer import BPEPreTokenizer
except ImportError:
    from bpe_tokenizer import BPETokenizer, BPETokenizerParams
    from bpe_tokenizer import train_bpe_on_corpus, print_time, train_bpe_on_byte_corpus
    from pre_tokenizer import BPEPreTokenizer

if __name__ == "__main__":
    test_text = """low low low low low\
        lower lower widest widest widest\
        newest newest newest newest newest newest"""


    params = BPETokenizerParams(
        merges={},
        vocab={chr(x): bytes([x]) for x in range(256)},
    )
    next_id = 256
    pretokenizer = BPEPreTokenizer(special_tokens=["<|endoftext|>", " ", "[PAD]"])
    token_bytes =  pretokenizer.pre_tokenize_str(test_text)
    print(f"Token Bytes: {token_bytes}")
    pretok = Counter()
    for raw, freq in token_bytes.items():
        decoded_string = "".join(raw)  # raw is a tuple of strings, so join them to decode/reconstruct
        if decoded_string != " ":
            word = tuple(decoded_string)
            pretok[word] = freq
            for ch in word:
                if ch not in params.vocab:
                    params.vocab[ch] = ch.encode("utf-8")
    print(f"Pretok: {pretok}")
    next_id, pretok = train_bpe_on_byte_corpus(pretok, params, next_id)
    print("Second Time")
    next_id , pretok= train_bpe_on_byte_corpus(pretok, params, next_id)
    tokenizer = BPETokenizer(params)
    print("V", len(params.vocab), "M", len(params.merges))
    print(pretok)
