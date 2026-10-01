from copy import deepcopy

import pytest

from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.tile import Wind
from training.rules_candidate_feature import CandidateFeatureAgent
from training.rules_candidate_game import CandidateThreePlayerMahjong
from training.rules_candidate_kan import winning_kinds
from training.rules_candidate_match import build_wall


CASES = [
    ('valid', '1筒 1筒 1筒 2筒 3筒 4筒 5条 6条 7条 8条 9条 东 东', '1筒', '1筒', True),
    ('changed_waits', '4筒 4筒 4筒 5筒 6筒 7筒 2条 3条 4条 3筒 3筒 1筒 2筒', '4筒', '4筒', False),
    ('stored_quad', '1筒 1筒 1筒 1筒 2筒 3筒 4筒 5筒 6筒 7筒 8筒 9筒 东', '北', '1筒', False),
    ('red_draw', '5筒 5筒 5筒 1筒 2筒 3筒 1条 2条 3条 4条 5条 东 东', '红5筒', '红5筒', True),
    ('fifth_copy_wait', '2条 2条 2条 3条 4条 5条 4筒 5筒 6筒 5条 5条 6条 7条', '2条', '2条', False),
]


def fixture(raw, drawn, cls=CandidateFeatureAgent):
    agent = cls(Wind.East, 0)
    agent.request2obs('Wind 1')
    agent.request2obs('Deal ' + raw)
    agent.turn_count[0], agent.discard_called = 1, True
    agent.isLiZhi[0] = True
    obs = agent.request2obs('Draw ' + drawn)
    return agent, obs


@pytest.mark.parametrize('name,raw,drawn,kan,allowed', CASES)
def test_candidate_ankan_mask_and_execution(name, raw, drawn, kan, allowed):
    agent, obs = fixture(raw, drawn)
    action = 87 + agent._tile_to_index(parse_tile_str(kan))
    assert bool(obs['action_mask'][action]) == allowed
    game = CandidateThreePlayerMahjong()
    game.reset(walls=build_wall(992260917))
    game.hands[0] = [parse_tile_str(t) for t in raw.split()]
    game.agents[0], game.obs = agent, {0: obs}
    game.curTile = parse_tile_str(drawn)
    game.isLiZhi[0] = True
    before_hands, before_packs = deepcopy(game.hands), deepcopy(game.packs)
    waits_before = winning_kinds(game.hands[0], [])
    _, _, done = game.step({'player_0': action})
    if allowed:
        assert not done and game.state == 3
        assert len(game.packs[0]) == 1 and len(game.hands[0]) == 10
        assert winning_kinds(game.hands[0], game.packs[0]) == waits_before
    else:
        assert done and '非法动作' in game.result_message
        assert game.hands == before_hands and game.packs == before_packs


def test_legacy_mask_reproduces_changed_waits_bug_without_modification():
    _, raw, drawn, kan, _ = CASES[1]
    agent, obs = fixture(raw, drawn, cls=FeatureAgent)
    assert obs['action_mask'][87 + agent._tile_to_index(parse_tile_str(kan))]
