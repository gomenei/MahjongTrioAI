"""Focused boundary tests for the isolated candidate, including action execution."""
from copy import deepcopy

import numpy as np
import pytest

from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.tile import Wind
from training.rules_candidate_feature import CandidateFeatureAgent
from training.rules_candidate_game import CandidateThreePlayerMahjong
from training.rules_candidate_match import build_wall


HAND = '1筒 2筒 3筒 4筒 5筒 6筒 7筒 8筒 9筒 1条 2条 北 北'


def agent_position(*, left=4, score=35000, riichi=False, draw='东', seat=0,
                   agent_class=CandidateFeatureAgent):
    agent = agent_class(Wind.East, seat)
    agent.request2obs('Wind 1')
    agent.request2obs('Deal ' + HAND)
    agent.turn_count[0] = 1
    agent.discard_called = True
    agent.tileWall = left + 1
    agent.scores[seat] = score
    agent.isLiZhi[0] = riichi
    observation = agent.request2obs('Draw ' + draw)
    return agent, observation


def game_position(**kwargs):
    agent, observation = agent_position(**kwargs)
    game = CandidateThreePlayerMahjong()
    game.reset(walls=build_wall(987260917))
    game.agents[0] = agent
    game.hands[0] = [parse_tile_str(t) for t in HAND.split()]
    game.curTile = agent.curtile
    game.wall = game.wall[:14 + agent.tileWall]
    game.WallLast = agent.tileWall == 0
    game.turn_count[0] = 1
    game.discard_called = True
    game.scores = agent.scores.copy()
    game.isLiZhi[0] = agent.isLiZhi[0]
    game.obs = {0: observation}
    return game, observation


@pytest.mark.parametrize('left', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('score', [900, 1000, 35000])
def test_riichi_mask_requires_points_and_future_draw(left, score):
    _, observation = agent_position(left=left, score=score, seat=2)
    offered = bool(observation['action_mask'][145:174].any())
    assert offered == (left >= 3 and score >= 1000)
    assert observation['match_observation'].shape == (121, 30)
    assert np.isfinite(observation['match_observation']).all()


@pytest.mark.parametrize('left,score', [(2, 35000), (0, 35000), (4, 900)])
def test_core_rejects_riichi_without_mutating_hand(left, score):
    game, _ = game_position(left=left, score=score)
    hands, packs = deepcopy(game.hands), deepcopy(game.packs)
    action = FeatureAgent.OFFSET_ACT['Riichi'] + FeatureAgent.STRING_TO_INDEX['东']
    _, _, done = game.step({'player_0': action})
    assert done and '非法动作' in game.result_message
    assert game.hands == hands and game.packs == packs
    assert not game.isLiZhi[0] and game.riichi_sticks == 0


def test_core_accepts_riichi_at_exact_boundaries():
    game, observation = game_position(left=3, score=1000)
    action = FeatureAgent.OFFSET_ACT['Riichi'] + FeatureAgent.STRING_TO_INDEX['东']
    assert observation['action_mask'][action]
    _, _, done = game.step({'player_0': action})
    assert not done and game.state == 2
    assert game.isLiZhi[0] and game.riichi_sticks == 1
    assert game.agents[0].scores[0] == 0
    assert len(game.hands[0]) == 13


@pytest.mark.parametrize('riichi', [False, True])
@pytest.mark.parametrize('draw', ['东', '北'])
@pytest.mark.parametrize('left', [0, 1, 4])
def test_north_mask_requires_replacement_and_post_riichi_draw(riichi, draw, left):
    _, observation = agent_position(riichi=riichi, draw=draw, left=left)
    offered = bool(observation['action_mask'][174])
    assert offered == (left > 0 and (not riichi or draw == '北'))


def test_core_rejects_extracting_stored_north_after_riichi():
    game, _ = game_position(riichi=True, draw='东')
    hands, packs = deepcopy(game.hands), deepcopy(game.packs)
    _, _, done = game.step({'player_0': FeatureAgent.OFFSET_ACT['Pei']})
    assert done and '非法动作' in game.result_message
    assert game.hands == hands and game.packs == packs


@pytest.mark.parametrize('riichi,draw', [(False, '东'), (True, '北')])
def test_core_accepts_valid_north_extraction(riichi, draw):
    game, observation = game_position(riichi=riichi, draw=draw)
    assert observation['action_mask'][174]
    before = list(game.hands[0])
    _, _, done = game.step({'player_0': 174})
    assert not done and game.state == 3 and len(game.packs[0]) == 1
    assert len(game.hands[0]) == 13
    if riichi:
        assert game.hands[0] == before


def test_pon_still_requires_discard_before_extracting_north():
    agent = CandidateFeatureAgent(Wind.East, 0)
    agent.request2obs('Wind 1')
    agent.request2obs('Deal 1筒 2筒 3筒 4筒 5筒 6筒 7筒 8筒 9筒 北 北 白 白')
    agent.request2obs('Player 2 Play 白')
    observation = agent.request2obs('Player 0 Peng 白')
    assert observation['action_mask'][:29].any()
    assert not observation['action_mask'][29:].any()
    game, _ = game_position()
    game.state = 0
    _, _, done = game.step({'player_0': 174})
    assert done and '非法动作' in game.result_message


def test_legacy_agent_remains_frozen():
    _, observation = agent_position(left=1, score=900, agent_class=FeatureAgent)
    assert observation['action_mask'][145:174].any()
    _, observation = agent_position(riichi=True, draw='东', agent_class=FeatureAgent)
    assert observation['action_mask'][174]
