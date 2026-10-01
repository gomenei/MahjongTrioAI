from pathlib import Path
from types import SimpleNamespace

import pytest

import run_autodl_generation_campaign as campaign


def test_next_generation_starts_from_current_champion():
    command = campaign.training_command(campaign.TRAIN_SEEDS[0], 20, "cuda")
    assert command[command.index("--base-model") + 1] == "model/autodl_verified_20260916.pt"
    assert command[command.index("--opponent-models") + 1] == campaign.BASE.as_posix()
    assert command[command.index("--max-hours") + 1] == "2.0"
    assert "--fresh" not in command


def test_average_ingredients_ignore_extra_resume_checkpoints(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    expected = []
    for minute, update in [(30, 70), (60, 134), (90, 199), (120, 263)]:
        path = directory / f"elapsed_{minute:04d}min_update_{update:06d}.pt"
        path.touch()
        expected.append(path)
    for name in ["update_000101.pt", "final_2.00h_update_000263.pt",
                 "elapsed_0060min_update_000135.pt"]:
        (directory / name).touch()
    assert campaign.time_checkpoints(tmp_path) == expected


@pytest.mark.parametrize("wrong_baseline", [False, True])
def test_evaluation_must_measure_against_installed_champion(tmp_path, monkeypatch, wrong_baseline):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    candidate = Path("candidate.pt")
    report = {"completed": True, "game_mode": "south", "rankings": [
        {"model_name": "candidate", "checkpoint": str(tmp_path / candidate),
         "average_rank": 1.98, "deal_in_rate": .13, "invalid_actions": 0},
        {"model_name": "baseline", "checkpoint": str(tmp_path / (campaign.BASE if wrong_baseline else campaign.CHAMPION)),
         "average_rank": 2.02, "deal_in_rate": .13, "invalid_actions": 0}],
        "pairwise": [{"model_a": "candidate", "model_b": "baseline", "paired_seed_groups": 12000,
                      "point_difference_ci95_low": 100, "point_difference_ci95_high": 900,
                      "utility_difference_ci95_low": .01, "utility_difference_ci95_high": .1}]}
    progress = {"lineups": {"a": {"errors": []}, "b": {"errors": []}}}
    commands = []
    monkeypatch.setattr(campaign, "run", lambda command: commands.append(command))
    monkeypatch.setattr(campaign, "read_json", lambda path: report if path.name == "battle_report.json" else progress)
    args = SimpleNamespace(env_workers=20, device="cuda")
    if wrong_baseline:
        with pytest.raises(ValueError, match="different baseline"):
            campaign.evaluate(args, candidate, Path("output"), campaign.CONFIRM_SEEDS[0], 6000)
    else:
        assert campaign.evaluate(args, candidate, Path("output"), campaign.CONFIRM_SEEDS[0], 6000)["pass"]
    assert "baseline=model/autodl_verified_20260916.pt" in commands[0]
