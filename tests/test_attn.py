import pytest
import torch

def test_attn():
    from strassen_attention.strassen_attention import strassen_attend

    q = torch.randn(1, 8, 32, 16)
    k = torch.randn(1, 8, 32, 16)
    v = torch.randn(1, 8, 32, 16)

    attended = strassen_attend(
        q,
        k,
        k.clone(),
        v,
        v.clone(),
        sim_clamp_value = 50.
    )

    assert attended.shape == (1, 8, 32, 16)
