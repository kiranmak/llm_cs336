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
    print(f"dimXLayers Trainable Parameters: {ans:,}")
    print("Need       Trainable Parameters: 1,640,452,800")
    return ans


vocab_size = 50257
context_length= 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288 #(the nearest multiple of 64 to 83×1,600)
ans = dimXLayers(vocab_size, num_layers,
                 b=1, seq=context_length,
                 heads=num_heads, d_model=d_model,
                 d_ff=d_ff)
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
