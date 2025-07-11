import pytest
import torch

@pytest.mark.parametrize('causal', (False, True))
def test_attn(causal):
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
        sim_clamp_value = 50.,
        causal = causal
    )

    assert attended.shape == (1, 8, 32, 16)

def test_cross_attn():
    from strassen_attention.strassen_attention import strassen_attend

    q = torch.randn(1, 8, 32, 16)
    k = torch.randn(1, 8, 64, 16)
    v = torch.randn(1, 8, 64, 16)

    attended = strassen_attend(
        q,
        k,
        k.clone(),
        v,
        v.clone(),
        sim_clamp_value = 50.,
    )

    assert attended.shape == (1, 8, 32, 16)

@pytest.mark.parametrize('causal', (False, True))
def test_mha(causal):
    from strassen_attention.strassen_mha import StrassenMHA

    mha = StrassenMHA(dim = 512)

    x = torch.randn(1, 256, 512)
    assert mha(x).shape == x.shape

def test_transformer():
    from strassen_attention.strassen_transformer import StrassenTransformer

    transformer = StrassenTransformer(dim = 512, depth = 2)

    x = torch.randn(1, 256, 512)
    assert transformer(x).shape == x.shape

    from strassen_attention.strassen_transformer import FeatureMapWrapper

    wrapped = FeatureMapWrapper(transformer)
    fmap = torch.randn(1, 512, 8, 8, 8)
    assert wrapped(fmap).shape == fmap.shape