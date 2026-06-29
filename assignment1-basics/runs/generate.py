import os
import time
import torch
from typing import Optional
from pathlib import Path

from cs336_basics.optim import AdamW
from cs336_basics.paths import CHECKPOINT_PATH
from cs336_basics.paths import EXP_PATH, OUT_PATH, set_device
from cs336_basics.configs import ConfigParams 
from cs336_basics.checkpoints import checkpoint_resume, load_hyperparams
from cs336_basics.transformer import TransformerModel
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.experiments_tracker import ExperimentTracker

def top_p_sampling(scaled_logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling to a probability vector.

    Args:
        probs: 1D tensor of probabilities (sum to 1).
        top_p: Cumulative probability threshold in (0, 1].

    Returns:
        Filtered probabilities (renormalized), same shape as probs.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    # 1. Sort logits and compute sorted probabilities
    s_logits, s_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(s_logits, dim=-1)

    # 2. Find threshold using cumulative sum
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 3. Filter: Keep tokens until cumulative sum exceeds top_p (always keep at least 1)
    keep_mask = cumulative_probs <= top_p
    keep_mask[..., 0] = True # Guaranteed safeguard to never filter out the top token

    # 4. Zero out excluded probabilities and re-normalize
    sorted_probs[~keep_mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # 5. Sample from sorted indices directly
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = torch.gather(s_indices, dim=-1, index=sampled_sorted_idx)
    return next_token

def get_tokenizer(tokenfile, prompt_text, device):
    tokens_topic = Path(tokenfile).stem

    vocab = OUT_PATH / f"{tokens_topic}-train_vocab.json"
    merges = OUT_PATH / f"{tokens_topic}-train_merges.txt"

    special_tokens=["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(vocab, merges,
                                        special_tokens)
    encoded_tokens = tokenizer.encode(prompt_text)

    token_ids = torch.tensor(encoded_tokens).unsqueeze(0).to(device)
    print("       shape of tokenids", token_ids.shape)
    eos_id = tokenizer.special_token_to_id.get("<|endoftext|>")

    return token_ids, eos_id, tokenizer


def load_running_config():
    checkpt_params_path = os.path.join(CHECKPOINT_PATH, "hyperparams.json")

    hp, tokenfile = load_hyperparams(checkpt_params_path)
    hp.show()
    print(f"  tokenfile: {tokenfile}\n")
    return hp, tokenfile

@torch.no_grad()
def generate_text(
        model: torch.nn.Module,
        prompt_token_ids: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int = 128,
        context_length: int = 256,
        generation_id: int =0,
        temperature: float=1.0, top_p:float =1.0
        )-> torch.Tensor:

    p_len = prompt_token_ids.shape[1]
    request_start_time = time.time()

    exp = ExperimentTracker(
         log_dir=EXP_PATH,
         service_name=tokenfile,
         config=None,  # dataclass will be serialized
         mode="decode",
     )

    if prompt_token_ids.dtype != torch.long:
        prompt_token_ids= pprompt_token_ids.to(torch.long)

    if max_new_tokens < 0:
        raise ValueError(
                f"max_new_tokens must be non-negative, got {max_new_tokens}")

    def _truncate(ctx_len, pt):
        if ctx_len is not None and pt.numel() > ctx_len:
            return pt[-ctx_len:]
        else:
            return pt

    model_mode = model.training
    model.eval()

    device = next(model.parameters()).device
    # pin tokens to device
    p_tokens = prompt_token_ids.to(device)

    # using @torch.no_grad() i dont need to use with torch.no_grad()
    for t_pos in range(max_new_tokens):
        token_start_time = time.time()
        input_context = _truncate(context_length, p_tokens)

        #input_context = token_ids[:, -context_length:]
        logits = model(input_context)

        token_latency_ms = (time.time() - token_start_time) * 1000
        next_token_logits = logits[:, -1, :]

        # Always track highest probability for logging safely
        probs_for_logging = torch.softmax(next_token_logits, dim=-1)
        highest_prob = torch.max(probs_for_logging).item()

        if 0.0 < temperature <= 1.0:
            scaled_logits = next_token_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if top_p < 1.0:
                next_token = top_p_sampling(probs, top_p)
        else: # 0 < temperature < 1.0:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        p_tokens = torch.cat((p_tokens, next_token), dim=1)

        # Pack metrics dictionary
        metrics = {
                "decode_run/token_pos": t_pos,
                "decode_run/token_latency_ms": token_latency_ms,
                "decode_run/top_token_probability": highest_prob,
            }
        exp.log(generation_id, metrics)

        if next_token.item() == eos_token_id:
            break
    if model_mode:
        model.train()

    total_request_time = time.time() - request_start_time
    tokens_generated   = p_tokens.shape[1] - p_len

    if total_request_time > 0:
        avg_tokens_per_sec = tokens_generated / total_request_time
    else:
        avg_tokens_per_sec = 0

    # Pack metrics dictionary
    metrics = {
        "inference_summary/total_latency_seconds": total_request_time,
        "inference_summary/generation_speed_tokens_per_sec": avg_tokens_per_sec,
        "inference_summary/output_sequence_length": p_tokens.shape[1]
    }
    exp.log(generation_id, metrics)
    exp.close()

    print(f"Finished run #{generation_id}! Generated {tokens_generated}",
          f" tokens at {avg_tokens_per_sec:.2f} tok/sec.")
    return p_tokens

def decoding(prompt_text:str,
             hp: ConfigParams,
             gen: int,
             tokenfile:str,
             temperature: float,
             top_p: float,
             device=None):

    from runs.train import  torch_dtype_from_string
    device = set_device(device)

    token_ids, eos_id, tokenizer = get_tokenizer(tokenfile,
                                                 prompt_text, device)

    # ---- 2) Build model (match training config) ----
    #model_dtype = cfg.model.torch_dtype.lower()
    model_dtype = "float32"
    dtype = torch_dtype_from_string(model_dtype)

    model = TransformerModel(
        hp.vocab_size,
        hp.d_model,
        hp.context_length,
        hp.rope_theta,
        hp.num_heads,
        hp.d_ff,
        hp.num_layers,
        device,
        dtype,
    ).to(device)

    optimizer = AdamW(model.parameters())

    checkpoint_resume(model, optimizer, hp.checkpoint)

    token_ids = generate_text(
        model=model,
        prompt_token_ids = token_ids,
        eos_token_id= eos_id,
        max_new_tokens=128,
        context_length=hp.context_length,
        generation_id=gen,
        temperature=temperature,
        top_p=top_p
    )


    full_text = tokenizer.decode(token_ids[0].tolist())
    print(f"\n--- Generated Text ---\n{full_text}")


if __name__ == "__main__":
    prompt_text = "Once, there was a little boy who was not feeling good"
    hyperparams, tokenfile = load_running_config()
    decoding(prompt_text, hyperparams,
             gen=0, tokenfile=tokenfile,
             temperature=1.0, top_p=0.9, device=None)
