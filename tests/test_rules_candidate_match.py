"""Candidate-flow regressions; these do not certify full MahjongSoul rules."""
from unittest.mock import patch

import numpy as np
import pytest

from mahjong_env.match import SouthMatch
from mahjong_env.tile import Wind
from training.rules_candidate_match import CandidateSouthMatch, noten_deltas


def position(round_index=5, scores=(35000, 35000, 35000), honba=0, **kwargs):
    match = CandidateSouthMatch(**kwargs)
    match.reset(984260917)
    match.round_index = round_index
    match.dealer = round_index % 3
    match.scores = list(scores)
    match.honba = honba
    match._start_hand()
    return match


def ron(match, winner=0, discarder=1):
    game = match.game
    game.winner = winner
    game.win_by = '荣和'
    game.current_player = discarder
    game.fans[winner] = 1
    game.fus[winner] = 30
    game.result_message = 'settlement fixture: valid ron'
    match._settle_hand()


def draw(match, tenpai):
    match.game.winner = None
    match.game.result_message = '流局'
    with patch('training.rules_candidate_match.tenpai_seats', return_value=tenpai):
        match._settle_hand()


def test_official_below_target_south3_case_enters_west():
    match = position()
    ron(match)
    assert match.scores == [36000, 34000, 35000]
    assert not match.done
    assert (match.round_index, match.round_number, match.dealer) == (6, 0, 0)
    assert match.prevailing_wind == Wind.West
    observation = match._start_hand()['player_0']
    assert match.game.prevailing_wind == Wind.West
    assert all(a.prevailing_wind == Wind.West for a in match.agents)
    assert observation['match_observation'].shape == (121, 30)
    assert observation['action_mask'].shape == (177,)
    assert np.isfinite(observation['match_observation']).all()


@pytest.mark.parametrize('first_before,finished', [(38900, False), (39000, True)])
def test_south3_40000_boundary(first_before, finished):
    match = position(scores=(first_before, 34000, 71000-first_before))
    ron(match)
    assert match.done == finished


def test_south3_dealer_under_target_continues_same_round():
    match = position()
    ron(match, winner=2, discarder=1)
    assert not match.done
    assert (match.round_index, match.dealer, match.honba) == (5, 2, 1)


def test_candidate_dealer_at_target_can_finish():
    match = position(scores=(34000, 32500, 38500))
    ron(match, winner=2, discarder=1)
    assert match.done and match.scores[2] == 40000
    assert match.final_ranks == [2, 3, 1]


def test_candidate_tied_dealer_behind_initial_east_does_not_finish():
    match = position(scores=(40000, 26500, 38500))
    ron(match, winner=2, discarder=1)
    assert match.scores == [40000, 25000, 40000]
    assert not match.done
    assert match.dealer == 2


@pytest.mark.parametrize('tenpai,next_dealer', [([], 0), ([2], 2)])
def test_candidate_draw_increments_honba_with_or_without_renchan(tenpai, next_dealer):
    match = position(honba=2)
    draw(match, tenpai)
    assert not match.done
    assert match.honba == 3
    assert match.dealer == next_dealer
    assert sum(match.scores) + match.riichi_sticks*1000 == 105000


def test_candidate_west_non_dealer_reaching_target_ends():
    match = position(round_index=6, scores=(34000, 39000, 32000))
    ron(match, winner=1, discarder=2)
    assert match.done
    assert match.scores[1] == 40000


def test_candidate_west3_non_dealer_win_ends_without_north_round():
    match = position(round_index=8)
    ron(match)
    assert match.done
    assert match.prevailing_wind == Wind.West


def test_candidate_safety_limit_does_not_manufacture_final_ranks():
    match = position(round_index=0, max_hands=1)
    with pytest.raises(RuntimeError, match='safety limit'):
        ron(match, winner=1, discarder=2)
    assert not match.done
    assert match.final_ranks is None


def test_candidate_terminal_sticks_are_paid_exactly_once():
    match = position(round_index=8, scores=(35000, 34000, 34000))
    match.riichi_sticks = 2
    match.game.riichi_sticks = 2
    draw(match, [])
    assert match.done
    assert match.scores == [37000, 34000, 34000]
    assert match.riichi_sticks == 0
    assert sum(match.scores) == 105000


def test_legacy_entry_keeps_frozen_ending_behavior():
    match = SouthMatch()
    match.reset(984260917)
    match.round_index, match.dealer = 5, 2
    match._start_hand()
    ron(match)
    assert match.done
    assert match.scores == [36000, 34000, 35000]


def test_candidate_rejects_unvalidated_variants():
    with pytest.raises(ValueError):
        CandidateSouthMatch(rounds_per_wind=4)
    with pytest.raises(ValueError):
        CandidateSouthMatch(initial_score=25000)
    with pytest.raises(ValueError):
        CandidateSouthMatch().reset(984260917, current_player=2)


@pytest.mark.parametrize('tenpai,expected', [
    ([], [0, 0, 0]), ([0, 1, 2], [0, 0, 0]),
    ([0], [2000, -1000, -1000]), ([1], [-1000, 2000, -1000]),
    ([2], [-1000, -1000, 2000]), ([0, 1], [1000, 1000, -2000]),
    ([0, 2], [1000, -2000, 1000]), ([1, 2], [-2000, 1000, 1000]),
])
def test_candidate_noten_payments_match_public_three_player_defaults(tenpai, expected):
    assert noten_deltas(tenpai) == expected


def test_candidate_draw_at_39000_does_not_prematurely_end():
    match = position(scores=(34000, 34000, 37000))
    draw(match, [2])
    assert match.scores == [33000, 33000, 39000]
    assert not match.done
    assert (match.round_index, match.dealer, match.honba) == (5, 2, 1)
