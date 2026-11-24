import pytest
import torch

from ascendep.model import MoEConfig, ReferenceMoE
from ascendep.routing import Trace, make_routing, make_trace, routing_stats, stream_route_summary


@pytest.mark.parametrize("tokens,experts,k", [(16, 4, 2), (17, 7, 3), (8, 256, 8)])
def test_balanced_conservation(tokens, experts, k):
    ids, stats = make_routing(tokens, experts, k)
    counts = stats["counts"]
    assert sum(counts) == tokens * k
    assert max(counts) - min(counts) <= 1
    assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()


@pytest.mark.parametrize("target", [.2, 1., 10., 20.])
def test_default_skew_targets(target):
    ids, stats = make_routing(8192, 256, 8, target)
    assert stats["status"] == "matched", stats
    assert stats["error"] <= stats["tolerance"]
    assert stats["assignments"] == 8192 * 8
    assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()


def test_unreachable_and_replay_integrity(tmp_path):
    _, stats = make_routing(16, 4, 2, 10)
    assert stats["status"] == "unreachable"
    config = MoEConfig()
    trace = make_trace(config, world=2)
    path = tmp_path / "trace.pt"
    trace.save(path)
    replay = Trace.load(path)
    assert replay.digest() == trace.digest()
    model = ReferenceMoE(config)
    torch.testing.assert_close(model(trace.x, trace.ids, trace.weights), model(replay.x, replay.ids, replay.weights), rtol=0, atol=0)
    data = torch.load(path, weights_only=True)
    data["weights"][0, 0] += .01
    torch.save(data, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        Trace.load(path)


def test_rank_hotspot_placement_distinction():
    clustered = torch.tensor([[0], [0], [1], [1]])
    scattered = torch.tensor([[0], [0], [2], [2]])
    a, b = routing_stats(clustered, 4, 2), routing_stats(scattered, 4, 2)
    assert a["actual_maxvio"] == b["actual_maxvio"]
    assert a["owner_rank_max_mean"] > b["owner_rank_max_mean"]


def test_streamed_large_ep_configuration():
    for world in (8, 16, 32, 64, 128, 256):
        result = stream_route_summary(2048, 16, 4096, world)
        assert result["assignments"] == world * 4096 * 16
        assert result["actual_maxvio"] == 0
        assert result["owner_rank_max_mean"] == 1
        assert result["model_weights_allocated"] is False


def test_small_sample_noise_is_not_hidden():
    _, stats = make_routing(32, 256, 8, .2)
    assert stats["status"] == "not_matched"
    assert stats["error"] > stats["tolerance"]
