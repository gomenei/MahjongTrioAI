import copy
from types import SimpleNamespace

import pytest

from ppo_train import validate_resume_settings
from run_autodl_training import assess_report, train_command


def report_fixture():
    return {
        "completed": True, "game_mode": "south",
        "rankings": [
            {"model_name": "candidate", "average_rank": 1.97,
             "deal_in_rate": 0.13, "invalid_actions": 0},
            {"model_name": "baseline", "average_rank": 2.03,
             "deal_in_rate": 0.13, "invalid_actions": 0},
        ],
        "pairwise": [{"model_a": "candidate", "model_b": "baseline",
                      "paired_seed_groups": 2000, "point_difference_ci95_low": 100,
                      "point_difference_ci95_high": 900,
                      "utility_difference_ci95_low": 0.01,
                      "utility_difference_ci95_high": 0.1}],
    }


def test_promotion_requires_valid_complete_evidence_on_both_metrics():
    report = report_fixture()
    assert assess_report(report, 1000)["pass"]
    assert not assess_report(report, 1000, errors=1)["pass"]
    for field, bad in (("completed", False), ("game_mode", "hand")):
        altered = copy.deepcopy(report)
        altered[field] = bad
        assert not assess_report(altered, 1000)["pass"]
    for field, bad in (("point_difference_ci95_low", -1),
                       ("utility_difference_ci95_low", -0.01),
                       ("paired_seed_groups", 1999)):
        altered = copy.deepcopy(report)
        altered["pairwise"][0][field] = bad
        assert not assess_report(altered, 1000)["pass"]
    for field, bad in (("invalid_actions", 1), ("deal_in_rate", 0.15),
                       ("average_rank", 2.03)):
        altered = copy.deepcopy(report)
        altered["rankings"][0][field] = bad
        assert not assess_report(altered, 1000)["pass"]


def test_reversed_pairwise_order_preserves_promotion_decision():
    report = report_fixture()
    pair = report["pairwise"][0]
    pair["model_a"], pair["model_b"] = pair["model_b"], pair["model_a"]
    for prefix in ("point", "utility"):
        low, high = f"{prefix}_difference_ci95_low", f"{prefix}_difference_ci95_high"
        pair[low], pair[high] = -pair[high], -pair[low]
    assert assess_report(report, 1000)["pass"]


def test_resume_prevents_silent_reward_or_reference_changes():
    payload = {"ppo_args": {"gamma": 1.0, "reference_kl_coef": 0.05},
               "ppo_opponent_pool_states": []}
    validate_resume_settings(payload, SimpleNamespace(gamma=1.0, reference_kl_coef=0.05))
    with pytest.raises(ValueError, match="gamma"):
        validate_resume_settings(payload, SimpleNamespace(gamma=0.995, reference_kl_coef=0.05))
    with pytest.raises(ValueError, match="reference_kl_coef"):
        validate_resume_settings(payload, SimpleNamespace(gamma=1.0, reference_kl_coef=0))


def test_smoke_and_training_use_separate_output_directories():
    args = SimpleNamespace(seed=20260916, device="cpu", env_workers=1,
                           hands_per_update=128, hours=4)
    smoke, smoke_command = train_command(args, smoke=True)
    formal, formal_command = train_command(args)
    assert smoke != formal
    assert "--fresh" not in formal_command
    assert smoke_command[smoke_command.index("--updates") + 1] == "3"
    assert formal_command[formal_command.index("--max-hours") + 1] == "4"
