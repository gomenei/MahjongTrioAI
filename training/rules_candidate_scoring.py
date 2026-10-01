"""Settlement for candidate hands; frozen legacy scoring remains unchanged."""
from mahjong_env.scoring import score_win
from mahjong_env.tile import Wind


def hand_winners(game):
    winners = getattr(game, 'winners', None)
    if winners:
        return list(winners)
    return [] if game.winner is None else [int(game.winner)]


def winning_hand_deltas(game):
    winners = hand_winners(game)
    if not winners:
        return [0, 0, 0], {}
    dealer = next(s for s, a in enumerate(game.agents) if a.seatWind == Wind.East)
    tsumo = game.win_by == '自摸'
    if tsumo and len(winners) != 1:
        raise ValueError('Only one player can win by self draw')
    discarder = None if tsumo else int(game.current_player)
    if not tsumo:
        winners.sort(key=lambda seat: (seat-discarder) % 3)
    deltas, payments = [0, 0, 0], {}
    for index, winner in enumerate(winners):
        partial, payment = score_win(fan=game.fans[winner], fu=game.fus[winner],
                                    winner=winner, discarder=discarder, dealer=dealer,
                                    honba=game.honba if index == 0 else 0)
        deltas = [a+b for a, b in zip(deltas, partial)]
        payments[winner] = payment
    return deltas, payments
