from collections import Counter
from bpe_tokenizer import BPETokenizer, BPETokenizerParams
from bpe_tokenizer import train_bpe_on_corpus, train_bpe_on_byte_corpus, print_time
from pre_tokenizer import BPEPreTokenizer

test_text = """low low low low low\
    lower lower widest widest widest\
    newest newest newest newest newest newest"""


params = BPETokenizerParams(
    merges={},
    vocab={chr(x): bytes([x]) for x in range(256)},
    next=256,
)
pretokenizer = BPEPreTokenizer(special_tokens=["<|endoftext|>", " ", "[PAD]"])
token_bytes =  pretokenizer.pre_tokenize_str(test_text)
print(f"Token Bytes: {token_bytes}")
pretok = Counter()
for raw, freq in token_bytes.items():
    decoded_string = bytes(raw).decode("utf-8")
    if decoded_string != " ":
        word = tuple(decoded_string)
        pretok[word] = freq
        for ch in word:
            if ch not in params.vocab:
                params.vocab[ch] = ch.encode("utf-8")
print(f"Pretok: {pretok}")
pretok, params = train_bpe_on_byte_corpus(pretok, params)
print("Second Time")
pretok, params = train_bpe_on_byte_corpus(pretok, params)
tokenizer = BPETokenizer(params)
print("V", len(params.vocab), "M", len(params.merges))
print(pretok)
