import sys
from cs336_basics.transformer import TransformerModel

def dimEmbeddings(vocab_size, d_model):
    ans = vocab_size * d_model
    print("dimEmbeddings:", ans)
    return ans

def dimLinear(d_in, d_out):
    ans =  d_in * d_out
    print("dimLinear:", ans)
    return ans

def dimRms(d_model):
    # 2 einsum - one for sqr nd one for norm
    ans = d_model
    print("dimRms:", ans)
    return ans

def dimSwiglu(d_ff, d_model):
    first = d_model * d_ff
    second = d_ff * d_model
    # it is down twice
    ans =  2 * first + second
    print("dimSwiglu:", ans)
    return ans

def dimRope(b, seq, d_model):
    rotations = 2
    dirs = 4  # cos_sine
    ans =  rotations * (d_model / 2) * dirs
    print("dimRope:", ans)
    return ans

def dimScaled_dot(d_model):
    wts = d_model ** 2
    k_q_V = 3
    ans =  k_q_V * wts
    print("dimScaled_dot:", ans)
    return ans

def dimMHA(b, seq, heads, d_model, d_k):
    # rearrange to b head seq d_model  - k,q,v
    attn = dimScaled_dot(d_model)
    # projection output
    out_proj = dimLinear(d_model, d_model)
    ans =  attn + out_proj
    print("dimMHA:", "4xd_model^2")
    print("dimMHA:", ans)
    return ans

def dimMHARope(b, seq, heads, d_model, d_k):
    return dimMHA(b, seq, heads, d_model, d_k)

def dimXblock(b, seq, heads, d_model, d_ff):
    # 1. MHA
    attn = dimMHARope(b, seq, heads, d_model, d_ff)
    # 2. FFN
    """ A standard Transformer FFN expands d_model a hidden dimension d_ff
    then contracts it back.
     - First Linear Layer: d_model x d_ff
     - Second Linear Layer: d_ff x d_model
    For SwiGLU activation functionthe gated linear unit requires
    two up-projections, changing the formula to
        2 x (d_model x d_ff + d_ff x d_model)
    """

    swig = dimSwiglu(d_ff, d_model)

    # 3. Layer Normalization (RMSNorm)
    """ RMSNorm has one learnable scaling parameter per channel.
    rms should return d_model"""

    ln1 = dimRms(d_model)
    ln2 = dimRms(d_model)
    ans =  (swig + ln1 + ln2 + attn)
    print("dimXblock: 4*d_model^2 + 2*d_model* d_ff + 2*d_model")
    print("dimXblock:", ans)
    return ans

def dimXLayers(vocab_size, layers, b, seq, heads, d_model, d_ff):
    emb = dimEmbeddings(vocab_size, d_model)
    x = layers * dimXblock(b, seq, heads, d_model, d_ff)
    # Final RMSNorm:
    final_ln = dimRms(d_model)
    # Final LM head:
    logits = dimLinear(d_model, vocab_size)
    ans = (emb + x + final_ln + logits)
    # Total learnable parameters:
    # Total Params = Embeddings + LM Head   +
    #                All Transformer Layers +
    #                Final Norm
    # All Transformer Layers = Transformer Block x layers
    #      Transformer Block = attention + RMS + FFN
    # Embeddings + LMHead =  2 x vocab_size x d_model
    # attention = 4 x d_model^2
    #       RMS = 2xd_model x d_ff + 2xd_model) +
    #       FFN = 2xd_model
    # Final Norm = d_model
    # Total
    #     =  2 x vocab_size x d_model
    #     + layers(4 x d_model^2 + 2xd_model x d_ff + 2xd_model)
    #     + d_model
    print(f"dimXLayers Trainable Parameters: {ans:,}")
    inbil = ans/(10**9)
    print(f"Trainable Parameters :{inbil:,} B" )
    return ans

def activation_memory(vocab_size, layers, batch, seq, heads, d_model, d_ff):
    # 1. Multi-Head Attention (MHA) Activations
    # In the attention block, tensors are created at every step of calculation:
    #  a. Input LayerNorm:
    #   1 tensor of shape (batch, seq_len, d_model)
    #   Elements =  batch x seq x d_model
    #  b. Q, K, V Projections: Before multiplying by the weights,
    #     the input is duplicated/projected.
    #     Elements = 3 x batch x seq x d_model
    #  c. Attention Score Matrix QK^T: massive memory bottleneck.
    #     For every head, it creates a matrix of size seq x seq.
    #     Elements = batch x num_heads x seq^2
    #  d. Softmax Output: probabilities require another matrix of the same size.
    #     Elements = batch x num_heads x seq^2
    #  e. Context Vector (After V multiplication):
    #     Elements = batch x seq x d_model
    #  f. Output Projection Layer:
    #     Elements = batch x seq x d_model
    #  Attention Activations =
    #     6xbatch x seq x d_model + 2xbatch x heads x seq^2
    #
    # 2. SwiGLU Feed-Forward Network (FFN) Activations
    #       using SwiGLU, FFN has three linear projections : w1, w2,w3.
    #  a. FFN Input LayerNorm:
    #     Elements = batch x seq x d_model
    #  b. w1,w2 Projections: input is projected into two separate hidden spaces
    #     of size d_ff
    #     Elements = 2 x batch x d_ff
    #  c. Swish Activation Output projection:
    #     Elements = batch x d_ff
    #  d. Down Projection Input: Fed into the final linear layer.
    #     Elements = batch x seq x d_model
    # SwiGLU FFN Activations =
    #     2 x batch x seq x d_model + 3 x  batch x d_ff
    # Total Activation Elements for One Layer =
    #     2 x batch x seq x d_model + 3 x  batch x d_ff
    #     6 x batch x seq x d_model + 2xbatch x heads x seq^2
    #   = 8 x batch x seq x d_model + 3 x  batch x d_ff
    #   + 2 x batch x heads x seq^2


    #  Attention Activations =
    attention_act  = 6 * batch * seq * d_model
    attention_act  += 2 * batch * num_heads * (seq ** 2)
    print("\n ======Memory Computations ======")
    print(f" Attention Activations per layer: {attention_act:,}")
    swig  = 2 * batch * seq * d_model
    swig  += 3 * batch * d_ff
    print(f"SwiGLU FFN Activations per layer:  { swig:,}")

    total_elements = swig + attention_act
    _fp32_sz = 4
    print(f"\tTotal Elements per layer: {total_elements:,}")

    total_elements *= num_layers
    print(f"\tTotal Elements all layers: {total_elements:,}")
    ans = (total_elements * _fp32_sz)/(10 ** 9)
    print(f"\tActivation Memory GB: {ans:,}")
    return ans

def flopsAdamW(total_params, batch, seq):
    # An algebraic expression for each of parameters,
    # activations, gradients, and optimizer state, as well as the total.

    # 1.Weight Decay: p = p - lr x lambda p
    #   Operations: 2 mul, 1 sub = 3 FLOPs
    # 2.First Moment Update m = beta_1 * m + (1 - beta_1) *grad
    #   Operations: 1 sub, 1 mul, 1 add= 3 FLOPs
    # 3.Second Moment Update 
    #   Operations: 1 mul, 1 sub, 1 mul, 1 add= 4 FLOPs
    # 4.Denominator Calculation: 
    #   Operations: 1 square root, 1 addition = 2 FLOPs
    #5. Gradient Step Update:
    #   Operations: 1 division, 1 multiplication, 1 subtraction = 3 FLOPs
    flops = 3 + 3 + 4 + 2 + 3

    inbil = total_params/(10**9)
    ans =  inbil * flops
    forward  =  2 * inbil * batch * seq
    backward = 4 * inbil * batch * seq
    print(f"Total Gigaflops for {inbil} B params")
    print(f"\t    Optimizer:{ans:4f} GFLOPs" )
    print(f"\t Forward Pass:{forward:,.4f} GFLOPs" )
    print(f"\tBackward Pass:{backward:,.4f} GFLOPs" )
    return ans, forward, backward

def computeMFU(mfu, params, batch, steps, seq):
    # Here is the step-by-step mathematical breakdown.
    # Step 1: Calculate FLOPs Per Token  6 * N
    flops_per_token = 6 * params  #FLOPs

    #Step 2: Calculate Total Training Tokens
    # - Tokens per Step = Batch Size x  seq = tokens per step
    # - Total Tokens = params * steps
    tokens_per_step = batch * seq  # token/step
    total_tokens = tokens_per_step  * steps
    # Step 3: Calculate Total Training FLOPs
    # Now, multiply the total tokens by the FLOPs required per token:
    # Total Compute
    total_flops = total_tokens * flops_per_token
    c_secs = total_flops/mfu
    hours = c_secs/3600
    mfut = mfu/(10**12)
    print(f"For mfu {mfut} teraflops Time taken will be {hours:,.2f} hrs" )
    print(f"\t or {hours/24:,.2f} days" )



vocab_size    = 50257
context_length= 1024
num_layers    = 48
d_model       = 1600
num_heads     = 25
d_ff          = 4288 #(the nearest multiple of 64 to 83×1,600)
print("\n ======Parameters Computations ======")
total_params = dimXLayers(vocab_size, num_layers,
                 b=1, seq=context_length,
                 heads=num_heads, d_model=d_model,
                 d_ff=d_ff)

ans = activation_memory(vocab_size, num_layers,
                 batch=1, seq=context_length,
                 heads=num_heads, d_model=d_model,
                 d_ff=d_ff)
print("\n ======Giga FLOPs Computations ======")
ans, forward, backward  = flopsAdamW(total_params,
                                     batch = 1,
                                     seq=context_length)
mfu = 250 * (10**12) # tera flops
steps = 400 * (10**3)
computeMFU(mfu, total_params, batch=1024, steps=steps, seq=context_length)

"""
model = TransformerModel(vocab_size,
                         d_model,
                         context_length,
                         0.1,
                         num_heads,
                         d_ff,
                         num_layers)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")
"""
