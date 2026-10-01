"""Averaged checkpoints must retain the feature schema used by real gameplay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from battle_models import choose_actions, load_policies
from mahjong_env.match import SouthMatch
from model import create_model
from run_autodl_average_campaign import average_checkpoints


def sources(tmp_path):
    torch.set_num_threads(2)
    model = create_model("tile_transformer", obs_channels=121)
    payload = {"model_name": "tile_transformer", "obs_channels": 121,
               "observation_key": "match_observation", "model_state_dict": model.state_dict()}
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(payload, a)
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    torch.save(payload, b)
    return a, b


def test_average_loads_through_real_battle_feature_selection(tmp_path):
    a, b = sources(tmp_path)
    out = tmp_path / "average.pt"
    average_checkpoints([a, b] * 4, out)
    args = SimpleNamespace(models=[f"candidate={out}", f"baseline={a}"], include_random=False)
    policy = load_policies(args, torch.device("cpu"))[0]
    assert policy.observation_key == "match_observation"
    observations = list(SouthMatch().reset(seed=362_000_000).values())
    actions = choose_actions(policy, observations, torch.device("cpu"), "greedy", 1,
                             np.random.default_rng(363_000_000))
    assert all(obs["action_mask"][action] for obs, action in zip(observations, actions))


def test_average_rejects_different_feature_semantics(tmp_path):
    a, b = sources(tmp_path)
    payload = torch.load(b, weights_only=True)
    payload["observation_key"] = "observation"
    torch.save(payload, b)
    with pytest.raises(ValueError, match="observation features differ"):
        average_checkpoints([a, b], tmp_path / "invalid.pt")
