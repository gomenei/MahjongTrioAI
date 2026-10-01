"""三麻点数结算；单局评测与南风场共用。"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from mahjong_env.tile import Wind


def round_up_100(value: float) -> int:
    return int(math.ceil(value / 100.0) * 100)


def basic_points(fan: int, fu: int) -> int:
    if fan < 0:
        return 8000 * (-fan)
    if fan >= 13:
        return 8000
    if fan >= 11:
        return 6000
    if fan >= 8:
        return 4000
    if fan >= 6:
        return 3000
    if fan == 5:
        return 2000
    return min(2000, int(fu) * (2 ** (int(fan) + 2)))


def score_win(
    *,
    fan: int,
    fu: int,
    winner: int,
    discarder: Optional[int],
    dealer: int,
    honba: int,
) -> Tuple[List[int], int]:
    """雀魂段位三麻自摸损：返回三家点差与赢家本手收入。"""
    base = basic_points(fan, fu)
    deltas = [0, 0, 0]
    if discarder is not None:
        multiplier = 6 if winner == dealer else 4
        payment = round_up_100(base * multiplier) + 200 * honba
        deltas[winner] += payment
        deltas[discarder] -= payment
        return deltas, payment

    for seat in range(3):
        if seat == winner:
            continue
        multiplier = 2 if winner == dealer or seat == dealer else 1
        payment = round_up_100(base * multiplier) + 100 * honba
        deltas[seat] -= payment
        deltas[winner] += payment
    return deltas, deltas[winner]


def winning_hand_deltas(game) -> Tuple[List[int], int]:
    if game.winner is None:
        return [0, 0, 0], 0
    winner = int(game.winner)
    dealer = next(
        seat
        for seat, agent in enumerate(game.agents)
        if agent.seatWind == Wind.East
    )
    discarder = None if game.win_by == "自摸" else int(game.current_player)
    return score_win(
        fan=game.fans[winner],
        fu=game.fus[winner],
        winner=winner,
        discarder=discarder,
        dealer=dealer,
        honba=game.honba,
    )
