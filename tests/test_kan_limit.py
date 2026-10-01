import copy

import pytest

from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.game import Error, ThreePlayerMahjong
from mahjong_env.tile import Meld, MeldType, Wind, can_declare_kan


def existing_packs(count):
    packs = [[], [], []]
    for index, text in enumerate(("1万", "9万", "东", "南")[:count]):
        tile = parse_tile_str(text)
        packs[1 + index % 2].append(Meld(MeldType.ClosedKan, [tile] * 4, tile))
    return packs


def make_agent(count):
    agent = FeatureAgent(Wind.South, 0)
    agent.packs = existing_packs(count)
    agent.hand = [parse_tile_str(text) for text in
                  ("白", "白", "白", "1筒", "2筒", "3筒", "4筒", "5筒", "6筒",
                   "1条", "2条", "3条", "北")]
    # These tests isolate kan legality from yaku and riichi calculations.
    agent._check_Hu = lambda isZimo: False
    agent._check_LiZhi = lambda: []
    return agent


@pytest.mark.parametrize("count, allowed", [(3, 1), (4, 0)])
def test_fourth_kan_allowed_fifth_kan_removed_from_every_action_type(count, allowed):
    agent = make_agent(count)
    white = agent.STRING_TO_INDEX["白"]
    obs = agent.request2obs("Draw 白")
    assert obs["action_mask"][agent.OFFSET_ACT["AnKang"] + white] == allowed
    assert obs["action_mask"][agent.OFFSET_ACT["Pei"]] == 1
    assert obs["action_mask"][:29].sum() > 0

    agent = make_agent(count)
    obs = agent.request2obs("Player 1 Play 白")
    assert obs["action_mask"][agent.OFFSET_ACT["Kang"] + white] == allowed
    assert obs["action_mask"][agent.OFFSET_ACT["Peng"] + white] == 1
    assert obs["action_mask"][agent.OFFSET_ACT["Pass"]] == 1

    agent = make_agent(count)
    tile = parse_tile_str("白")
    agent.packs[0].append(Meld(MeldType.Pon, [tile] * 3, tile))
    agent.hand = agent.hand[2:-1]
    obs = agent.request2obs("Draw 北")
    assert obs["action_mask"][agent.OFFSET_ACT["BuKang"] + white] == allowed
    assert obs["action_mask"][agent.OFFSET_ACT["Pei"]] == 1


def test_extracted_norths_do_not_consume_kan_capacity():
    packs = existing_packs(3)
    north = parse_tile_str("北")
    packs[0].extend(Meld(MeldType.Pei, [north], north) for _ in range(4))
    assert can_declare_kan(packs)
    white = parse_tile_str("白")
    packs[0].append(Meld(MeldType.OpenKan, [white] * 4, white))
    assert not can_declare_kan(packs)


@pytest.mark.parametrize("method", ["_kong", "_concealedKong", "_promoteKong"])
def test_engine_rejects_fifth_kan_before_mutating_tiles(method):
    game = ThreePlayerMahjong()
    game.reset()
    game.packs = existing_packs(4)
    game.hands[0] = [parse_tile_str("白") for _ in range(4)]
    before = copy.deepcopy((game.packs, game.hands, game.numdora))
    with pytest.raises(Error):
        getattr(game, method)(0, parse_tile_str("白"))
    assert (game.packs, game.hands, game.numdora) == before
