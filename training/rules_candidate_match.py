"""Isolated match-flow candidate; not yet approved for training or app use.

See artifacts/PLAN_RULES_CANDIDATE_20260917.md for scope and unresolved rules.
The legacy environment is not imported through this class automatically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from training.rules_candidate_game import CandidateThreePlayerMahjong as ThreePlayerMahjong
from training.rules_candidate_scoring import hand_winners, winning_hand_deltas
from mahjong_env.tile import Suit, Tile, Wind
from mahjong_env.util import MahjongAgariCalculator


def build_wall(seed: int) -> List[Tile]:
    wall: List[Tile] = []
    for suit in (Suit.Manzu, Suit.Pinzu, Suit.Souzu):
        for value in range(1, 10):
            if suit == Suit.Manzu and 1 < value < 9:
                continue
            if suit in (Suit.Pinzu, Suit.Souzu) and value == 5:
                wall.append(Tile(suit, value, is_red=True))
                wall.extend(Tile(suit, value) for _ in range(3))
            else:
                wall.extend(Tile(suit, value) for _ in range(4))
    for value in range(1, 8):
        wall.extend(Tile(Suit.Honors, value) for _ in range(4))
    random.Random(seed).shuffle(wall)
    return wall


def tile_kinds() -> List[Tile]:
    tiles = [Tile(Suit.Manzu, 1), Tile(Suit.Manzu, 9)]
    for suit in (Suit.Pinzu, Suit.Souzu):
        tiles.extend(Tile(suit, value) for value in range(1, 10))
    tiles.extend(Tile(Suit.Honors, value) for value in range(1, 8))
    return tiles


def tenpai_seats(game: ThreePlayerMahjong) -> List[int]:
    result = []
    candidates = tile_kinds()
    for seat in range(3):
        own_tiles = list(game.hands[seat])
        for pack in game.packs[seat]:
            own_tiles.extend(pack.tiles)
        can_complete = False
        for tile in candidates:
            if own_tiles.count(tile) >= 4:
                continue
            try:
                if MahjongAgariCalculator(
                    game.hands[seat], game.packs[seat], tile
                ):
                    can_complete = True
                    break
            except Exception:
                continue
        if can_complete:
            result.append(seat)
    return result


def noten_deltas(tenpai: List[int]) -> List[int]:
    """三麻流局罚符总额 2000 点（雀魂公开客户端默认规则）。"""
    if len(tenpai) in (0, 3):
        return [0, 0, 0]
    result = [0, 0, 0]
    gain = 2000 // len(tenpai)
    loss = 2000 // (3 - len(tenpai))
    for seat in range(3):
        result[seat] = gain if seat in tenpai else -loss
    return result


@dataclass
class HandResult:
    index: int
    prevailing_wind: str
    round_number: int
    dealer: int
    honba: int
    winner: Optional[int]
    winners: List[int]
    win_by: Optional[str]
    discarder: Optional[int]
    tenpai: List[int]
    deltas: List[int]
    scores: List[int]
    message: str
    abortive_draw: Optional[str] = None


class CandidateSouthMatch:
    """Experimental South/West flow and targeted single-hand legality repairs."""

    agent_names = ThreePlayerMahjong.agent_names
    rules_version = "sanma_rules_candidate_v6"
    policy_action_contract = "177_actions_optional_nine_terminals_always_declined"

    def __init__(
        self,
        *,
        rounds_per_wind: int = 3,
        initial_score: int = 35000,
        max_hands: int = 150,
    ):
        if rounds_per_wind != 3:
            raise ValueError("The candidate supports three rounds per wind only")
        if initial_score != 35000 or max_hands < 1:
            raise ValueError("Use 35000 initial points and a positive safety limit")
        self.rounds_per_wind = rounds_per_wind
        self.total_rounds = rounds_per_wind * 2
        self.initial_score = int(initial_score)
        self.max_hands = int(max_hands)
        self.game: Optional[ThreePlayerMahjong] = None
        self.done = False

    def reset(self, seed: int, current_player: int = 0):
        # Absolute seat zero represents the initial East seat, including ties.
        # Evaluation rotates model assignments, not this rank-priority order.
        if current_player != 0:
            raise ValueError("Rotate model assignments; initial East must be seat zero")
        self.seed = int(seed)
        self.scores = [self.initial_score] * 3
        self.starting_scores = self.scores.copy()
        self.dealer = int(current_player)
        self.round_index = 0
        self.honba = 0
        self.riichi_sticks = 0
        self.hand_count = 0
        self.history: List[HandResult] = []
        self.done = False
        self.winner = None
        self.final_ranks = None
        self.result_message = ""
        return self._start_hand()

    @property
    def prevailing_wind(self) -> Wind:
        return (Wind.East, Wind.South, Wind.West)[self.round_index // 3]

    @property
    def round_number(self) -> int:
        return self.round_index % self.rounds_per_wind

    @property
    def agents(self):
        return self.game.agents if self.game is not None else []

    @property
    def current_player(self):
        return self.game.current_player if self.game is not None else self.dealer

    def _start_hand(self):
        self.game = ThreePlayerMahjong()
        config = {
            "prevailing_wind": self.prevailing_wind,
            "round_number": self.round_number,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "scores": self.scores,
            "round_index": self.round_index,
            "total_rounds": self.total_rounds,
        }
        hand_seed = self.seed + self.hand_count * 1_000_003
        return self.game.reset(
            walls=build_wall(hand_seed),
            current_player=self.dealer,
            config=config,
        )

    def _finish_match(self):
        # 终局仍留在场上的立直棒归当前一位，避免总点数在比赛结束时凭空减少。
        if self.riichi_sticks:
            provisional_ranking = sorted(
                range(3), key=lambda seat: (-self.scores[seat], seat)
            )
            self.scores[provisional_ranking[0]] += self.riichi_sticks * 1000
            self.riichi_sticks = 0
            if self.history:
                self.history[-1].scores = self.scores.copy()
        ranking = sorted(range(3), key=lambda seat: (-self.scores[seat], seat))
        self.final_ranks = [ranking.index(seat) + 1 for seat in range(3)]
        self.winner = ranking[0]
        self.done = True
        self.result_message = (
            f"南风场结束：scores={self.scores}, ranks={self.final_ranks}"
        )

    def _settle_hand(self):
        game = self.game
        old_honba = self.honba
        tenpai = []
        winners = hand_winners(game)
        abortive = getattr(game, 'abortive_draw', None)
        if "非法动作" in game.result_message:
            import re

            match = re.search(r"玩家\s*(\d+)", game.result_message)
            offender = int(match.group(1)) if match else 0
            deltas = [-8000 if seat == offender else 4000 for seat in range(3)]
            for seat, declared in enumerate(game.isLiZhi):
                if declared:
                    deltas[seat] -= 1000
            self.riichi_sticks = game.riichi_sticks
            dealer_continues = False
            force_finish = True
        elif game.winner is not None:
            deltas, _ = winning_hand_deltas(game)
            for seat, declared in enumerate(game.isLiZhi):
                if declared:
                    deltas[seat] -= 1000
            # Winners are ordered clockwise from the discarder by the core.
            deltas[winners[0]] += game.riichi_sticks * 1000
            self.riichi_sticks = 0
            dealer_continues = self.dealer in winners
            force_finish = False
        elif abortive:
            deltas = [-1000 if declared else 0 for declared in game.isLiZhi]
            self.riichi_sticks = game.riichi_sticks
            dealer_continues = True
            force_finish = False
        else:
            tenpai = tenpai_seats(game)
            deltas = noten_deltas(tenpai)
            for seat, declared in enumerate(game.isLiZhi):
                if declared:
                    deltas[seat] -= 1000
            self.riichi_sticks = game.riichi_sticks
            dealer_continues = self.dealer in tenpai
            force_finish = False

        self.scores = [score + delta for score, delta in zip(self.scores, deltas)]
        self.history.append(
            HandResult(
                index=self.hand_count,
                prevailing_wind=self.prevailing_wind.name,
                round_number=self.round_number,
                dealer=self.dealer,
                honba=old_honba,
                winner=game.winner,
                winners=winners,
                win_by=game.win_by,
                discarder=(
                    int(game.current_player)
                    if game.winner is not None and game.win_by != "自摸"
                    else None
                ),
                tenpai=tenpai,
                deltas=deltas,
                scores=self.scores.copy(),
                message=game.result_message,
                abortive_draw=abortive,
            )
        )
        self.hand_count += 1

        scheduled_final = self.round_index >= self.total_rounds - 1
        first_seat = min(range(3), key=lambda seat: (-self.scores[seat], seat))
        dealer_can_finish = first_seat == self.dealer and self.scores[self.dealer] >= 40000
        non_dealer_can_finish = max(self.scores) >= 40000 or self.round_index >= 8
        normal_finish = not abortive and scheduled_final and (
            (dealer_continues and dealer_can_finish)
            or (not dealer_continues and non_dealer_can_finish)
        )
        if (
            force_finish
            or any(score < 0 for score in self.scores)
            or normal_finish
        ):
            self._finish_match()
            return

        # A simulation limit is not a game rule and must not manufacture ranks.
        if self.hand_count >= self.max_hands:
            raise RuntimeError("Match safety limit exceeded before a rules-based ending")

        self.honba = old_honba + 1 if game.winner is None or dealer_continues else 0
        if dealer_continues:
            return
        else:
            self.dealer = (self.dealer + 1) % 3
            self.round_index += 1

    def step(self, actions: Dict[str, int]):
        if self.done:
            raise RuntimeError("南风场已经结束")
        observations, _, hand_done = self.game.step(actions)
        if not hand_done:
            return observations, {name: 0.0 for name in observations}, False
        self._settle_hand()
        if self.done:
            return {}, {}, True
        observations = self._start_hand()
        return observations, {name: 0.0 for name in observations}, False

    def get_public_state(self):
        state = self.game.get_public_state()
        state.update(
            scores=self.scores.copy(),
            prevailing_wind=self.prevailing_wind,
            round_number=self.round_number,
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
            match_done=self.done,
            final_ranks=self.final_ranks,
            hands_played=self.hand_count,
        )
        return state
