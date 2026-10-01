"""Decision observations must include the drawn tile and exclude called tiles."""
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np
import pytest

from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.tile import Wind
from training.rules_candidate_feature import CandidateFeatureAgent


@pytest.fixture(params=[FeatureAgent, CandidateFeatureAgent])
def agent(request):
    result=request.param(Wind.West,0)
    result.request2obs('Wind 1')
    result.request2obs('Deal 1万 1筒 3筒 4筒 8筒 1条 5条 8条 东 白 白 发 中')
    return result


def assert_hand(observation, tiles):
    expected=np.zeros((4,30))
    for tile,count in Counter(str(t) for t in tiles).items():
        expected[:count,FeatureAgent.STRING_TO_INDEX[tile]]=1
    for key in ('observation','rich_observation','match_observation'):
        np.testing.assert_array_equal(observation[key][2:6],expected)


@pytest.mark.parametrize('draw',['7筒','白','红5筒'])
def test_draw_and_discard_keep_all_three_feature_schemas_in_sync(agent,draw):
    observation=agent.request2obs('Draw '+draw)
    assert_hand(observation,agent.hand)
    assert observation['action_mask'][FeatureAgent.STRING_TO_INDEX[draw]]==1
    agent.request2obs('Player 0 Play '+draw)
    response=agent.request2obs('Player 1 Play 9筒')
    assert_hand(response,agent.hand)
    # Returned arrays must not change when the following discard changes state.
    assert observation['observation'][2:6].sum()==14


def test_north_replacement_draw_is_in_the_observation(agent):
    agent.request2obs('Draw 北')
    agent.request2obs('Player 0 Pei')
    observation=agent.request2obs('Draw 7筒')
    assert_hand(observation,agent.hand)
    assert observation['match_observation'][2,FeatureAgent.STRING_TO_INDEX['北']]==0
    assert observation['match_observation'][2,FeatureAgent.STRING_TO_INDEX['7筒']]==1


def test_closed_kan_removes_all_four_tiles_before_replacement_draw(agent):
    agent.hand=[parse_tile_str(t) for t in ('1筒','1筒','1筒','2筒','3筒','4筒','5筒','6筒','7筒','8筒','东','白','发')]
    agent._hand_embedding_update()
    agent.request2obs('Draw 1筒')
    agent.request2obs('Player 0 AnKang 1筒')
    assert agent.obs[2:6,FeatureAgent.STRING_TO_INDEX['1筒']].sum()==0
    observation=agent.request2obs('Draw 9筒')
    assert_hand(observation,agent.hand)
    assert observation['match_observation'][2:6].sum()==11


