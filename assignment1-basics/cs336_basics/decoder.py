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
from cs336_basics.optim import AdamW
from cs336_basics.transformer import TransformerModel
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.checkpoints import checkpoint_resume,load_hyperparams
from cs336_basics.checkpoints import CHECKPOINT_PATH
from cs336_basics.paths import PROJECT_ROOT, DATA_PATH, OUT_PATH
import os
import math
from pathlib import Path
import torch


def get_bpe_params_for_dataset(tokenfile):
    # Get only the name
    tokens_topic = Path(tokenfile).stem
    #print("tokens_topic - ", tokens_topic)
    vocab_file = OUT_PATH / f"{tokens_topic}_vocab.json"
    merge_file = OUT_PATH / f"{tokens_topic}_merges.txt"
    #print("vocab --", vocab_file, " merge -- ", merge_file)
    return vocab_file, merge_file


def reload_hyper_params():
    checkpt_params_path = os.path.join(CHECKPOINT_PATH,
                                       f"hyperparams.json")
    hp, tokenfile = load_hyperparams(checkpt_params_path)
    vocab_size = hp.vocab_size
    num_layers = hp.num_layers
    context_length = hp.context_length
    batch_size = hp.batch_size
    num_heads  = hp.num_heads
    d_ff       = hp.d_ff
    d_model    = hp.d_model
    rope_theta = hp.rope_theta

    print("\n")
    print("Decoder with ")
    print("  vocab size: ", vocab_size)
    print("  context length: ", context_length)
    print("  num heads:  ", num_heads)
    print("  d_ff:       ", d_ff)
    print("  d_model:    ", d_model)
    print("  rope_theta: ", rope_theta)
    print("  tokenfile:  ", tokenfile)
    print("\n")
    return hp, tokenfile

def decoding(prompt_text, hp, vocab_path, merges_path, device=None):

    vocab_size = hp.vocab_size
    num_layers = hp.num_layers
    context_length = hp.context_length
    batch_size = hp.batch_size
    num_heads  = hp.num_heads
    d_ff       = hp.d_ff
    d_model    = hp.d_model
    rope_theta = hp.rope_theta


    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # initialized weights 0:
    model = TransformerModel(vocab_size, d_model,
                             context_length, rope_theta,
                             num_heads, d_ff, num_layers).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=1e-3,
                      weight_decay=0.01,
                      betas=(0.9, 0.999), eps=1e-8,)

    checkpoint_resume(model, optimizer, hp.checkpoint)

    tokenizer = BPETokenizer.from_files(vocab_path, merge_path)
    encoded_tokens = tokenizer.encode(prompt_text)

    # ... your existing code ends here ...
    token_ids = torch.tensor(encoded_tokens).unsqueeze(0).to(device) # Shape: (1, prompt_length)

    # 1. Define your max tokens to generate
    max_tokens = 50
    temperature = 0.7  # Controls randomness (optional)

    # 2. Put model in evaluation mode
    model.eval()
    # 3. Start generation loop

    with torch.no_grad():
      for _ in range(max_tokens):
         # limit the input context if it exceeds context length
         # (Takes the tail of input sequence of size 'context_length')
         input_context = token_ids[:, -context_length:]

         # Pass the token-ids into your model
         # Output logits shape: (Batch_Size, Current_Sequence_Length, Vocab_Size)
         logits = model(input_context)

         # Pluck out ONLY the logits for the very last token in the sequence
         # Shape changes from (1, T, Vocab_Size) -> (1, Vocab_Size)
         next_token_logits = logits[:, -1, :]

         # --- SAMPLING STRATEGY ---
         next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

         # Append the new token index to your running sequence matrix along the time dimension
         token_ids = torch.cat((token_ids, next_token), dim=1)

         # Optional: Stop early if the model generates an End-of-String (EOS) token ID
         if next_token.item() == "|<endoftext>|":
            break
    # After the loop finishes, token_ids holds the full generated sequence
    # Convert token IDs back to readable text
    full_text = tokenizer.decode(token_ids[0].tolist())
    print(f"\n--- Generated Text ---\n{full_text}")

def  prompt():
    # The input is automatically treated as a string type
    #user_text = input("Ask something: ")
    user_text = "When we think about a nice cozy evening, the first thing that comes to"
    return user_text + "|<endoftext>|"

if __name__ == "__main__":
    prompt_text = prompt()
    hp, tokenfile = reload_hyper_params()
    vocab_path, merge_path = get_bpe_params_for_dataset(tokenfile)
    decoding(prompt_text, hp, vocab_path, merge_path)
