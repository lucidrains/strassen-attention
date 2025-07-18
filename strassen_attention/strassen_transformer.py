from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList
from strassen_attention.strassen_mha import StrassenMHA

# simple feedforward

def FeedForward(dim, expansion = 4.):
    dim_inner = int(dim * expansion)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# transformer

class StrassenTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        dim_head = 64,
        causal = False,
        ff_expansion = 4.,
        rotary_embed = False,
        attn_kwargs: dict = dict(
            dim_head_values = None,
            kv_heads = None,
            qk_rmsnorm = True,
            attn_logits_clamp_value = 40.
        ),
        final_norm = True
    ):
        super().__init__()

        layers = []

        for _ in range(depth):
            attn = StrassenMHA(
                dim = dim,
                causal = causal,
                heads = heads,
                dim_head = dim_head,
                pre_rmsnorm = True,
                rotary_embed = rotary_embed,
                **attn_kwargs
            )

            ff = FeedForward(dim = dim, expansion = ff_expansion)

            layers.append(ModuleList([
                attn,
                ff
            ]))

        self.layers = ModuleList(layers)
        self.norm = nn.RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# feature map acceptor
# so can be used at the middle of a image or video unet easily

from einops import rearrange, pack, unpack

class FeatureMapWrapper(Module):
    def __init__(
        self,
        transformer: StrassenTransformer
    ):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        fmap # b c *dims
    ):

        fmap = rearrange(fmap, 'b d ... -> b ... d')

        fmap, packed_shape = pack((fmap,), 'b * c')

        attended = self.transformer(fmap)

        attended, = unpack(attended, packed_shape, 'b * c')

        return rearrange(attended, 'b ... d -> b d ...')
