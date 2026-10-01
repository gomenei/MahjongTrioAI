import json
import sys

import pytest

import run_autodl_campaign as campaign


@pytest.mark.skipif(sys.platform == "win32", reason="Campaign uses a Linux process lock")
def test_campaign_skips_confirmation_without_candidate_and_does_not_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "BASE", campaign.Path("base.pt"))
    (tmp_path / "base.pt").write_bytes(b"test baseline")
    monkeypatch.setattr(sys, "argv", ["run_autodl_campaign.py", "--seeds", "1", "2"])
    launches = []

    class FakeProcess:
        pid = 99999999
        returncode = 0

        def __init__(self, command, **kwargs):
            stage = command[2]
            seed = command[command.index("--seed") + 1]
            launches.append((seed, stage))
            if stage == "screen":
                campaign.write_json(tmp_path / "battle_results" / f"league_v2_{seed}"
                                    / "selection.json", {"selected": None})

        def poll(self):
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", FakeProcess)
    campaign.main()
    assert launches == [("1", "train"), ("1", "screen"), ("2", "train"), ("2", "screen")]
    state = json.loads((tmp_path / "campaigns/league_v2/status.json").read_text())
    assert state["status"] == "completed"
    assert not state["strength_improvement_proven"]
    campaign.main()
    assert len(launches) == 4
