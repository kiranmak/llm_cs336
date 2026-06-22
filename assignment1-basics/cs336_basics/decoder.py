"""
Problem (decoding): Decoding (3 points)
Deliverable: Implement a function to decode from your language model.
We recommend that you support the following features:

    1. Generate completions for a user-provided prompt (i.e., take in
     some 𝑥1…𝑡 and sample a completion until you hit an <|endoftext|> token).
    2. Allow the user to control the maximum number of generated tokens.
    3. Given a desired temperature value, apply softmax temperature scaling to the
       predicted next-token distributions before sampling.
    4. Top-𝑝 sampling ([A. Holtzman et al., 2020] also referred to as nucleus
       sampling), given a user-specified threshold value.
"""

"""
To provide the model with a sequence of prefix tokens ("prompt") and
get the next-token probability distribution, you need to perform the 
following steps:

    1. Tokenize the prompt: Convert the string prompt into a list/tensor of token IDs
       using your BPE tokenizer.
    2. Batch & device placement: Add a batch dimension using .unsqueeze(0) and
       place the tensor on the correct device (CPU/GPU/MPS).
    3. Truncate to context length: If the prompt (or generated history) is
       longer than the model's supported context_length, you must slice it to
       the most recent context_length tokens.
    4. Run forward pass: Pass the tensor to the model to get the logits of
       shape (batch_size, sequence_length, vocab_size).
    5. Extract next-token distribution: Take the logits of the last token in
       the sequence (logits[0, -1, :]).
    6. Apply temperature and nucleus (top-p) sampling: Adjust the logits
       and convert them to a probability distribution using softmax.
"""
from tests.test_tokenizers import get_tokenizer_from_vocab_merges_path
from cs336_basics.run import load_hyperparams, CHECKPOINT_PATH
from cs336_basics.bpe_train import DATA_PATH, OUT_PATH

def decoding(prompt_string, vocab_path, device):

    checkpt_params_path = os.path.join(CHECKPOINT_PATH,
                                       f"hyperparams.json")
    hp = load_hyperparams(CHECKPOINT_PATH)
    vocab_size = hp.vocab_size
    num_layers = hp.num_layers
    batch_size = hp.batch_size
    num_heads  = hp.num_heads
    d_ff       = hp.d_ff
    d_model    = hp.d_model
    rope_theta = hp.rope_theta

    model = TransformerModel(vocab_size, d_model,
                             context_length, rope_theta,
                             num_heads, d_ff, num_layers).to(device)
    model.to(device)
    vocab_path = OUT

    tokenizer = get_tokenizer_from_vocab_merges_path(
            vocab_path,
            merges_path)
    encoded_tokens = tokenizer.encode(prompt_string)
    token_ids = torch.tensor(encoded_tokens).unsqueeze(0).to(device)

def  prompt():
    # The input is automatically treated as a string type
    user_text = input("Ask something: ")
    return user_text + "|<endoftext>|"

if __name__ == "__main__":
    prompt_text = prompt()
    src_file = "TinyStories"
    VOCAB_FILE =  OUT_PATH / f"{src_file}_vocab.json"
    MERGES_FILE = OUT_PATH / f"{src_file}_merges.txt"
    decoding(prompt_text, vocab_file)

