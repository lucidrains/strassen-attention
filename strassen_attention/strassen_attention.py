import torch
from torch import stack
from torch.nn import Module

from einops import einsum, rearrange
from opt_einsum import contract

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value = 30.):
    return (t / value).tanh() * value

# main attention function

def strassen_attend(
    q,   # b h i d
    k1,  # b h j d
    k2,  # b h k d
    v1,  # b h j dv
    v2,  # b h k dv,
    sim_clamp_value = None,
    causal = False,
    activate_fn = torch.exp
): # b h i dv

    scale = q.shape[-1] ** -0.5

    q = q * scale

    # three way dot product

    source = stack((q, q, k1))
    target = stack((k1, k2, k2))

    sims = einsum(source, target, '... i d, ... j d -> ... i j')
    
    # there could be an unaddressed instability issue in this paper, deal with it using the proven similarity softclamp from gemma2

    if exists(sim_clamp_value):
        sims = softclamp(sims, sim_clamp_value)

    sim_ij, sim_ik, sim_jk = sims

    # causal mask

    if causal:
        seq_len = sims.shape[-1]
        causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = q.device).triu(1)
        sim_ij = sim_ij.masked_fill(causal_mask, -torch.finfo(sims.dtype).max)
        sim_ik = sim_ik.masked_fill(causal_mask, -torch.finfo(sims.dtype).max)

    # do their efficient way

    # activation function, defaults to exponentiation

    exp_sims = activate_fn(stack((sim_ij, sim_ik, sim_jk)))

    # decomposed (n x n) X, Z, Y in paper

    exp_sim_ij, exp_sim_ik, exp_sim_jk = exp_sims

    # follow their notation of y and y_hat

    y = exp_sim_jk
    y_hat = contract('... j k, ... j d, ... k d -> ... j k d', exp_sim_jk, v1, v2)

    # complete it

    num = contract('... i j, ... j k d, ... i k -> ... i d', exp_sim_ij, y_hat, exp_sim_ik)

    den = contract('... i j, ... j k, ... i k -> ... i', exp_sim_ij, y, exp_sim_ik)

    return num / rearrange(den, '... -> ... 1')
