from unittest.mock import patch

import pytest

from mahjong_env.feature import parse_tile_str
from mahjong_env.tile import Meld, MeldType
from training.rules_candidate_match import CandidateSouthMatch
from training.rules_candidate_ron_fixtures import discard, position


def kans(game, distribution):
    # Synthetic response boundary; independent MJAI comparisons use full hands.
    tiles = iter(map(parse_tile_str, ['1万', '9万', '东', '南']))
    game.packs = [[Meld(MeldType.ClosedKan, [t]*4, t)
                   for t in [next(tiles) for _ in range(n)]] for n in distribution]


@pytest.mark.parametrize('counts,abort', [((2,2,0),True), ((3,0,1),True),
                                         ((1,1,2),True), ((4,0,0),False),
                                         ((1,1,1),False)])
def test_four_kan_response_boundary(counts, abort):
    match, _, winners = position()
    discard(match)
    game = match.game
    kans(game, counts)
    game.step({game.agent_names[s]:176 for s in winners})
    assert game.done == abort
    assert game.abortive_draw == ('suukansansen' if abort else None)


def test_north_does_not_count_as_fourth_kan():
    match, _, winners = position()
    discard(match)
    game = match.game
    kans(game, (2,1,0))
    north = parse_tile_str('北')
    game.packs[2].append(Meld(MeldType.Pei, [north], north))
    game.step({game.agent_names[s]:176 for s in winners})
    assert not game.done and game.abortive_draw is None


def test_ron_has_priority_over_four_kan_abort():
    match, _, winners = position()
    discard(match)
    game = match.game
    kans(game, (2,2,0))
    # Retain real legal winning hand on seat 2; synthetic kans belong to others.
    game.step({'player_1':176, 'player_2':175})
    assert game.winners == [2] and game.abortive_draw is None


def test_fourth_kan_response_still_draws_replacement():
    match, _, winners = position()
    game = match.game
    kans(game, (2,2,0))
    game.state = 3
    game.step({game.agent_names[s]:176 for s in winners})
    assert not game.done and game.state == 1 and game.abortive_draw is None


@pytest.mark.parametrize('round_index', [0,5,6,8])
def test_abort_repeats_dealer_without_noten_or_final_round_stop(round_index):
    scores = [25000]*3
    scores[round_index%3] = 53000
    match, _, _ = position(round_index=round_index, scores=scores, honba=2, sticks=2)
    game = match.game
    game.isLiZhi[0] = True
    game.riichi_sticks += 1
    game._finish_abortive_draw('suukansansen')
    with patch('training.rules_candidate_match.tenpai_seats', side_effect=AssertionError('Must not check tenpai')):
        match._settle_hand()
    assert not match.done
    assert (match.round_index,match.dealer,match.honba,match.riichi_sticks) == (round_index,round_index%3,3,3)
    assert match.history[-1].deltas == [-1000,0,0]
    assert sum(match.scores)+match.riichi_sticks*1000 == 105000


def nine_terminals():
    match = CandidateSouthMatch()
    match.reset(998260917)
    game = match.game
    game.hands[0] = list(map(parse_tile_str, '1万 9万 1筒 9筒 1条 9条 东 南 西 白 白 发 中'.split()))
    game.curTile = parse_tile_str('北')
    return match


def test_nine_terminals_optional_abort_and_177_action_contract():
    match = nine_terminals()
    game = match.game
    assert game.can_abort_nine_terminals(0)
    assert not game.done
    assert all(o['action_mask'].shape == (177,) for o in game.obs.values())
    game.abort_nine_terminals(0)
    match._settle_hand()
    assert not match.done and match.honba == 1 and match.dealer == 0
    assert match.history[-1].abortive_draw == 'kyushu_kyuhai'


@pytest.mark.parametrize('block', ['wrong_player','after_discard','after_call','reaction','few_unique'])
def test_nine_terminals_rejects_ineligible_without_mutation(block):
    game = nine_terminals().game
    player = 1 if block == 'wrong_player' else 0
    if block == 'after_discard': game.turn_count[0] = 1
    if block == 'after_call': game.discard_called = True
    if block == 'reaction': game.state = 2
    if block == 'few_unique': game.hands[0] = [parse_tile_str('1万')]*13
    assert not game.can_abort_nine_terminals(player)
    with pytest.raises(ValueError): game.abort_nine_terminals(player)
    assert not game.done and game.abortive_draw is None
