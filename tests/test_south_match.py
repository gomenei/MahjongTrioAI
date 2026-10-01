import unittest

import numpy as np

from battle_models import MatchTask, RunningGame, summarize_match
from mahjong_env.feature import FeatureAgent
from mahjong_env.match import HandResult, SouthMatch, noten_deltas


class MatchObservationTest(unittest.TestCase):
    def test_match_observation_keeps_rich_prefix(self):
        match = SouthMatch()
        observations = match.reset(seed=1234)
        observation = observations["player_0"]
        self.assertEqual(
            observation["match_observation"].shape,
            (FeatureAgent.MATCH_OBS_SIZE, 30),
        )
        np.testing.assert_array_equal(
            observation["match_observation"][:FeatureAgent.RICH_OBS_SIZE],
            observation["rich_observation"],
        )

    def test_match_context_contains_scores_and_rank(self):
        match = SouthMatch()
        observations = match.reset(seed=1234)
        context = observations["player_0"]["match_observation"]
        self.assertTrue(np.all(np.isfinite(context)))
        self.assertTrue(np.all((context >= 0) & (context <= 1)))
        rank_planes = context[
            FeatureAgent.OFFSET_MATCH["SELF_RANK"]:
            FeatureAgent.OFFSET_MATCH["SELF_RANK"] + 3
        ]
        self.assertEqual(int(rank_planes[:, 0].sum()), 1)


class SouthMatchSettlementTest(unittest.TestCase):
    def test_noten_payment_is_zero_sum(self):
        self.assertEqual(noten_deltas([]), [0, 0, 0])
        self.assertEqual(noten_deltas([0]), [3000, -1500, -1500])
        self.assertEqual(noten_deltas([0, 1]), [1500, 1500, -3000])

    def test_non_dealer_win_advances_round_and_dealer(self):
        match = SouthMatch()
        match.reset(seed=4321)
        game = match.game
        game.winner = 1
        game.win_by = "荣和"
        game.current_player = 2
        game.fans[1] = 1
        game.fus[1] = 30
        game.result_message = "玩家 1 荣和：1 番 30 符"
        game.isLiZhi = [False, False, False]
        match._settle_hand()
        self.assertEqual(match.round_index, 1)
        self.assertEqual(match.dealer, 1)
        self.assertEqual(sum(match.scores), 105000)
        self.assertEqual(match.history[-1].discarder, 2)

    def test_final_riichi_sticks_are_awarded_to_first_place(self):
        match = SouthMatch()
        match.reset(seed=9876)
        match.scores = [36000, 34000, 32000]
        match.riichi_sticks = 3
        match._finish_match()
        self.assertEqual(match.scores, [39000, 34000, 32000])
        self.assertEqual(match.riichi_sticks, 0)
        self.assertEqual(match.final_ranks, [1, 2, 3])

    def test_match_summary_counts_hand_win_tsumo_and_deal_in(self):
        match = SouthMatch()
        match.reset(seed=2468)
        match.hand_count = 3
        match.final_ranks = [1, 2, 3]
        match.scores = [40000, 35000, 30000]
        match.history = [
            HandResult(
                0, "East", 0, 0, 0, 0, "自摸", None, [],
                [4000, -2000, -2000], [39000, 33000, 33000], "玩家 0 自摸",
            ),
            HandResult(
                1, "East", 1, 1, 0, 1, "荣和", 0, [],
                [-2000, 2000, 0], [37000, 35000, 33000], "玩家 1 荣和",
            ),
            HandResult(
                2, "East", 2, 2, 0, None, None, None, [],
                [0, 0, 0], [37000, 35000, 33000], "流局",
            ),
        ]
        item = RunningGame(
            task=MatchTask(seed=2468, rotation=0, seat_models=(0, 1, 1)),
            game=match,
            observations={},
        )
        summary = summarize_match(item, match)
        self.assertEqual(summary[0]["played_hands"], 3)
        self.assertEqual(summary[0]["wins"], 1)
        self.assertEqual(summary[0]["tsumo_wins"], 1)
        self.assertEqual(summary[0]["ron_wins"], 0)
        self.assertEqual(summary[0]["deal_ins"], 1)
        self.assertEqual(summary[1]["played_hands"], 6)
        self.assertEqual(summary[1]["wins"], 1)
        self.assertEqual(summary[1]["ron_wins"], 1)


if __name__ == "__main__":
    unittest.main()
