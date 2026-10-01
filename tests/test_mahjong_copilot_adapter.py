from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COPILOT_ROOT = PROJECT_ROOT / "integrations" / "MahjongCopilot"
if str(COPILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(COPILOT_ROOT))

from bot.mahjongtrio.bot_mahjongtrio import (  # noqa: E402
    BotMahjongTrioAI,
    MjaiFeatureState,
    mjai_to_tile,
    tile_to_mjai,
)
from common.utils import GameMode  # noqa: E402
from mahjong_env.feature import FeatureAgent  # noqa: E402


HAND = [
    "1m", "9m", "1p", "2p", "3p", "4p", "5pr",
    "5p", "6p", "7p", "1s", "2s", "E",
]


def start_kyoku_event(hand=None):
    hand = hand or HAND
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "4p",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [35000, 35000, 35000, 0],
        "tehais": [hand, ["?"] * 13, ["?"] * 13, ["?"] * 13],
    }


def test_tile_conversion_round_trip():
    for text in ("1m", "9m", "5p", "5pr", "5sr", "E", "N", "C"):
        assert tile_to_mjai(mjai_to_tile(text)) == text


def test_mjai_stream_builds_rich_observation_and_pon_mask():
    state = MjaiFeatureState(seat=0)
    state.apply({"type": "start_game", "id": 0})
    state.apply(start_kyoku_event())
    state.apply({"type": "tsumo", "actor": 1, "pai": "?"})
    observation = state.apply(
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False}
    )

    assert observation["rich_observation"].shape == (101, 30)
    assert observation["match_observation"].shape == (121, 30)
    np.testing.assert_array_equal(
        observation["match_observation"][:101],
        observation["rich_observation"],
    )
    assert observation["action_mask"].shape == (177,)
    assert observation["action_mask"][FeatureAgent.OFFSET_ACT["Pass"]] == 1
    # Calling a normal five while holding red+normal five is represented by
    # the red-five pon action in the existing 177-action vocabulary.
    assert observation["action_mask"][FeatureAgent.OFFSET_ACT["Peng"] + 27] == 1


def test_default_checkpoint_produces_copilot_reaction():
    checkpoint = PROJECT_ROOT / "ppo_runs" / "ppo_rich" / "best.pt"
    if not checkpoint.is_file():
        return

    bot = BotMahjongTrioAI(str(checkpoint), device="cpu", action_mode="greedy")
    bot.init_bot(0, GameMode.MJ3P)
    bot.set_runtime_context(
        {"seat": 0, "operationList": [{"type": 1, "combination": []}]},
        HAND,
        "9s",
    )
    reaction = bot.react_batch(
        [
            {"type": "start_game", "id": 0},
            start_kyoku_event(),
            {"type": "tsumo", "actor": 0, "pai": "9s"},
        ]
    )

    assert reaction["type"] in {
        "dahai", "reach", "ankan", "kakan", "nukidora", "hora"
    }
    assert reaction["actor"] == 0
    assert reaction["disable_copilot_randomize"] is True
    assert np.isclose(sum(weight for _, weight in reaction["meta_options"]), 1.0)
