"""Fully specified closed hands for independent multi-ron settlement checks."""
from mahjong_env.feature import parse_tile_str
from training.rules_candidate_feature import CandidateFeatureAgent
from training.rules_candidate_match import CandidateSouthMatch


WINNING_HANDS = [
    '1筒 2筒 3筒 4筒 5筒 6筒 7筒 8筒 2条 2条 白 白 白',
    '1筒 2筒 3筒 4筒 5筒 6筒 7筒 8筒 4条 4条 发 发 发',
]
DISCARDER_HAND = '1万 1万 1万 9万 9万 9万 东 东 东 中 中 中 北'


def position(*, round_index=0, discarder=0, honba=0, sticks=0, scores=None):
    match = CandidateSouthMatch()
    match.reset(995260917)
    match.round_index, match.dealer = round_index, round_index % 3
    match.honba, match.riichi_sticks = honba, sticks
    match.scores = list(scores or [35000, 35000, 35000-sticks*1000])
    match._start_hand()
    game = match.game
    winners = [(discarder+i) % 3 for i in (1,2)]
    hands = [None]*3
    hands[discarder] = DISCARDER_HAND
    for seat, hand in zip(winners, WINNING_HANDS):
        hands[seat] = hand
    for seat, raw in enumerate(hands):
        old = game.agents[seat]
        agent = CandidateFeatureAgent(old.seatWind, seat)
        agent.set_match_context(round_number=match.round_number, honba=honba,
                                riichi_sticks=sticks, scores=match.scores,
                                dealer_seat=match.dealer, round_index=round_index)
        agent.request2obs(f'Wind {game.prevailing_wind.value}')
        agent.request2obs('Dora 南')
        agent.request2obs('Deal ' + raw)
        agent.turn_count, agent.discard_called = [1,1,1], True
        game.agents[seat] = agent
        game.hands[seat] = [parse_tile_str(t) for t in raw.split()]
    game.turn_count, game.discard_called = [1,1,1], True
    game.dora, game.lidora = [parse_tile_str('南')], []
    game.current_player, game.curTile, game.state = discarder, parse_tile_str('9筒'), 1
    for seat, agent in enumerate(game.agents):
        if seat != discarder:
            agent.request2obs(f'Player {discarder} Draw')
    game.obs = {discarder: game.agents[discarder].request2obs('Draw 9筒')}
    return match, hands, winners


def discard(match, *, riichi=False):
    game = match.game
    agent = game.agents[game.current_player]
    action = agent.OFFSET_ACT['Riichi' if riichi else 'Play'] + agent.STRING_TO_INDEX['9筒']
    assert game.obs[game.current_player]['action_mask'][action]
    return game.step({game.agent_names[game.current_player]:action})
