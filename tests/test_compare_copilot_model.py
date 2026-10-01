import numpy as np
import pytest

from compare_copilot_model import (
    GameSnapshot,
    reaction_to_local_action,
    response_parts,
    tile_to_mjai,
    transition_events,
)
from mahjong_env.feature import FeatureAgent
from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.tile import Suit, Tile, Wind


def test_mjai_tile_format_preserves_red_five():
    assert tile_to_mjai(Tile(Suit.Pinzu, 5, is_red=True)) == "5pr"
    assert tile_to_mjai(Tile(Suit.Honors, 4)) == "N"


def test_mortal_discard_maps_to_same_legal_local_action():
    game = ThreePlayerMahjong()
    observations = game.reset(current_player=0)
    observation = observations["player_0"]
    action = next(
        int(value)
        for value in np.flatnonzero(observation["action_mask"] > 0)
        if value < FeatureAgent.OFFSET_ACT["Peng"]
    )
    tile = game.agents[0]._index_to_tile(action)
    mapped, issue = reaction_to_local_action(
        {"type": "dahai", "actor": 0, "pai": tile_to_mjai(tile)},
        observation,
        game.agents[0],
    )
    assert issue is None
    assert mapped == action


@pytest.mark.parametrize('suit,letter', [(Suit.Pinzu, 'p'), (Suit.Souzu, 's')])
@pytest.mark.parametrize('called_red,consume_red', [(True, False), (False, True), (False, False)])
def test_pon_red_choice_follows_consumed_hand_tiles(suit, letter, called_red, consume_red):
    agent = FeatureAgent(Wind.South, 0)
    agent.request2obs('Wind 1')
    # For a normal discard both red-consuming and normal-consuming pons exist.
    fives = ([Tile(suit, 5), Tile(suit, 5)] if called_red else
             [Tile(suit, 5), Tile(suit, 5), Tile(suit, 5, is_red=True)])
    filler = [Tile(Suit.Honors, x) for x in (1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7)]
    agent.hand = fives + filler[:13-len(fives)]
    agent._hand_embedding_update()
    called = Tile(suit, 5, is_red=called_red)
    observation = agent.request2obs('Player 1 Play '+str(called))
    reaction = {'type':'pon', 'actor':0, 'target':1, 'pai':tile_to_mjai(called),
                'consumed':[f'5{letter}r' if consume_red else f'5{letter}', f'5{letter}']}
    expected = agent.response2action('Peng '+str(Tile(suit, 5, is_red=consume_red)))
    assert observation['action_mask'][expected] == 1
    mapped, issue = reaction_to_local_action(reaction, observation, agent)
    assert issue is None
    assert mapped == expected


def test_internal_discard_and_next_draw_become_mjai_events():
    game = ThreePlayerMahjong()
    observations = game.reset(current_player=0)
    play = next(
        int(value)
        for value in np.flatnonzero(observations["player_0"]["action_mask"] > 0)
        if value < FeatureAgent.OFFSET_ACT["Peng"]
    )
    actions = {"player_0": play}
    before = GameSnapshot.capture(game)
    responses = response_parts(game, observations, actions)
    observations, _, done = game.step(actions)
    assert not done
    events, pending = transition_events(before, game, responses, None)
    assert pending is None
    assert [event["type"] for event in events] == ["dahai"]

    pass_actions = {
        name: FeatureAgent.OFFSET_ACT["Pass"] for name in observations
    }
    before = GameSnapshot.capture(game)
    responses = response_parts(game, observations, pass_actions)
    _, _, done = game.step(pass_actions)
    assert not done
    events, pending = transition_events(before, game, responses, pending)
    assert pending is None
    assert [event["type"] for event in events] == ["tsumo"]
    assert events[0]["actor"] == 1
