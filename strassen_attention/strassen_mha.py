import torch
from torch import stack
from torch.nn import Module, Linear, RMSNorm, Identity

from strassen_attention.strassen_attention import strassen_attend

from einops import rearrange
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class StrassenMHA(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,           # query / key head dimension
        dim_head_values = None,  # values head dimension (defaults to qk head dimension)
        heads = 8,               # query heads
        kv_heads = None,         # key / value heads (defaults to query heads)
        causal = False,
        qk_rmsnorm = True,
        pre_rmsnorm = False,
        attn_logits_clamp_value = 40.
    ):
        super().__init__()
        self.pre_norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        dim_head_values = default(dim_head_values, dim_head)
        kv_heads = default(kv_heads, heads)

        self.split_dims = (
            dim_head * heads,           # query
            dim_head * kv_heads,        # keys 1
            dim_head * kv_heads,        # keys 2
            dim_head_values * kv_heads, # values 1
            dim_head_values * kv_heads, # values 2
        )

        self.to_qkv = Linear(dim, sum(self.split_dims), bias = False)

        # qk rmsnorm, now used in a number of works, from stable diffusion 3 to alphagenome. it is fine

        self.qk_rmsnorm = qk_rmsnorm
        if qk_rmsnorm:
            self.q_norm = RMSNorm(dim_head)
            self.k1_norm = RMSNorm(dim_head)
            self.k2_norm = RMSNorm(dim_head)

        # gemma2 - like attention logit soft clamping

        self.attn_logits_clamp_value = attn_logits_clamp_value

        # split out attention heads

        self.split_q_heads = Rearrange('... n (h d) -> ... h n d', h = heads)
        self.split_kv_heads = Rearrange('... n (h d) -> ... h n d', h = kv_heads)

        self.causal = causal

        self.merge_heads = Rearrange('... h n d -> ... n (h d)')

        self.to_out = Linear(dim_head_values * kv_heads, dim, bias = False)

    def forward(
        self,
        x
    ):
        x = self.pre_norm(x)

        q, k1, k2, v1, v2 = self.to_qkv(x).split(self.split_dims, dim = -1)

        q = self.split_q_heads(q)
        k1, k2, v1, v2 = self.split_kv_heads(stack((k1, k2, v1, v2)))

        if self.qk_rmsnorm:
            q = self.q_norm(q)
            k1 = self.k1_norm(k1)
            k2 = self.k2_norm(k2)

        out = strassen_attend(q, k1, k2, v1, v2, causal = self.causal, sim_clamp_value = self.attn_logits_clamp_value)

        out = self.merge_heads(out)

        return self.to_out(out)
