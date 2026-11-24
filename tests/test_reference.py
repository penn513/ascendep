import torch
import pytest

from ascendep.model import LatentMoE, MoEConfig, ReferenceMoE


def test_hand_computed_swiglu_and_weighted_sum():
    layer = ReferenceMoE(MoEConfig(experts=2, top_k=2, hidden=1, ffn=1))
    with torch.no_grad():
        for e, expert in layer.experts.items():
            expert.gate.fill_(1)
            expert.up.fill_(2)
            expert.down.fill_(int(e) + 1)
    x = torch.tensor([[1.], [2.]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([[.25, .75], [.6, .4]], dtype=torch.float64, requires_grad=True)
    actual = layer(x, ids, weights)
    scalar = 2 * x.square() / (1 + (-x).exp())
    expected = scalar * torch.tensor([[1.75], [1.6]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad.abs().sum() > 0 and weights.grad.abs().sum() > 0
    altered = ids.clone()
    altered[0] = altered[0].flip(0)
    assert not torch.allclose(actual, layer(x, altered, weights))
    assert not torch.allclose(actual, layer(x, ids, weights + .1))


def test_reference_gradcheck():
    layer = ReferenceMoE(MoEConfig(experts=3, top_k=2, hidden=2, ffn=3))
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 2], [2, 1]])
    assert torch.autograd.gradcheck(lambda a, b: layer(a, ids, b), (x, weights))


def test_invalid_routing_and_latent_projection():
    layer = ReferenceMoE(MoEConfig(hidden=3, ffn=5))
    x = torch.randn(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="duplicate"):
        layer(x, torch.tensor([[0, 0], [1, 2]]), torch.ones(2, 2, dtype=torch.float64))
    latent = LatentMoE(7, layer)
    x = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    out = latent(x, torch.tensor([[0, 1], [2, 3]]), torch.ones(2, 2, dtype=torch.float64))
    assert out.shape == x.shape
    out.square().sum().backward()
    assert x.grad is not None
    assert latent.project_in.weight.grad.abs().sum() > 0
