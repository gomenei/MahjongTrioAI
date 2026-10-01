from types import SimpleNamespace

import pytest

from mahjong_env.scoring import score_win
from training.rules_candidate_metrics import finish_candidate_episode, summarize_candidate_match
from training.rules_candidate_ron_fixtures import discard, position


@pytest.mark.parametrize('dealer', range(3))
@pytest.mark.parametrize('discarder', range(3))
def test_double_ron_all_dealers_and_discarders(dealer, discarder):
    match, _, winners = position(round_index=dealer, discarder=discarder, honba=2, sticks=2)
    discard(match)
    game = match.game
    assert all(game.obs[w]['action_mask'][175] for w in winners)
    game.step({game.agent_names[w]:175 for w in reversed(winners)})
    assert game.winners == winners and game.winner == winners[0]
    assert all(game.fans[w] == 3 and game.fus[w] == 40 for w in winners)
    expected = [0]*3
    for index, w in enumerate(winners):
        delta, _ = score_win(fan=3, fu=40, winner=w, discarder=discarder,
                             dealer=dealer, honba=2 if index == 0 else 0)
        expected = [a+b for a,b in zip(expected,delta)]
    expected[winners[0]] += 2000
    match._settle_hand()
    assert match.history[-1].deltas == expected
    assert match.history[-1].winners == winners
    assert sum(match.scores) == 105000 and match.riichi_sticks == 0
    assert match.dealer == (dealer if dealer in winners else (dealer+1)%3)
    assert match.honba == (3 if dealer in winners else 0)


def test_second_ron_claim_is_validated_before_recording_any_winner():
    match, _, winners = position()
    discard(match)
    game = match.game
    game.hands[winners[1]] = []
    game.step({game.agent_names[w]:175 for w in winners})
    assert game.done and '非法动作' in game.result_message
    assert game.winner is None and game.winners == []
    assert game.fans == [0,0,0] and game.fus == [0,0,0]


@pytest.mark.parametrize('claims', [(1,), (2,), (1,2)])
def test_declaring_discard_ron_cancels_new_deposit(claims):
    match, _, winners = position(sticks=2)
    discard(match, riichi=True)
    game = match.game
    assert game.pending_riichi == 0 and game.riichi_sticks == 3
    game.step({game.agent_names[w]:175 if w in claims else 176 for w in winners})
    assert game.pending_riichi is None and not game.isLiZhi[0]
    assert game.riichi_sticks == 2
    assert all(a.scores[0] == match.scores[0] for a in game.agents)
    match._settle_hand()
    assert sum(match.scores) == 105000
    assert match.history[-1].winners == list(claims)


def test_declaration_surviving_discard_is_accepted():
    match, _, winners = position()
    discard(match, riichi=True)
    game = match.game
    game.step({game.agent_names[w]:176 for w in winners})
    assert game.pending_riichi is None and game.isLiZhi[0]
    assert game.riichi_sticks == 1


def test_metrics_credit_both_winners_and_one_deal_in():
    match, _, winners = position(round_index=5, discarder=2)
    discard(match)
    game = match.game
    game.step({game.agent_names[w]:175 for w in winners})
    match._settle_hand()
    assert match.done
    item = SimpleNamespace(task=SimpleNamespace(seat_models=(0,1,2)))
    stats = summarize_candidate_match(item, match)
    assert stats[0]['ron_wins'] == stats[1]['ron_wins'] == 1
    assert stats[2]['deal_ins'] == 1
    assert sum(s['point_sum'] for s in stats.values()) == 0
    args = SimpleNamespace(reward_scale=32000, rank_reward_weight=2., reward_clip=6.)
    for seat in range(3):
        result = finish_candidate_episode(SimpleNamespace(game=match, learner_seat=seat),args)
        assert result['win'] == int(seat in winners)
        assert result['deal_in'] == int(seat == 2)
        expected = result['points']/32000 + 2*{1:1,2:0,3:-1}[result['rank']]
        assert result['reward'] == pytest.approx(expected)


def test_dealer_as_second_winner_still_continues():
    match, _, winners = position(round_index=5, discarder=0, scores=[55000,28000,22000])
    discard(match)
    game = match.game
    game.step({game.agent_names[w]:175 for w in winners})
    match._settle_hand()
    assert winners == [1,2] and not match.done
    assert (match.round_index, match.dealer, match.honba) == (5,2,1)


def test_multiple_ron_from_kan_response_state():
    # Fixed response-state boundary; the full kan event chain has separate tests.
    match, _, winners = position(honba=1)
    discard(match)
    game = match.game
    game.state, game.isQiangGang = 3, True
    game.step({game.agent_names[w]:175 for w in winners})
    assert game.winners == winners
    assert all(game.fans[w] == 4 and game.fus[w] == 40 for w in winners)
    match._settle_hand()
    assert match.history[-1].deltas == [-16200,8200,8000]


def test_one_shared_policy_gets_both_wins_without_double_counting_deal_in():
    match, _, winners = position(round_index=5, discarder=2)
    discard(match)
    game = match.game
    game.step({game.agent_names[w]:175 for w in winners})
    match._settle_hand()
    stats = summarize_candidate_match(SimpleNamespace(task=SimpleNamespace(seat_models=(0,0,1))),match)
    assert stats[0]['wins'] == stats[0]['ron_wins'] == 2
    assert stats[1]['deal_ins'] == 1
    assert stats[0]['point_sum'] + stats[1]['point_sum'] == 0
