import unittest

from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.game import Error, ThreePlayerMahjong
from mahjong_env.tile import Wind


def make_waiting_agent():
    """白刻子，筒子 6/9 两面听；两张都能完成牌型且有役。"""
    agent = FeatureAgent(Wind.South, 0)
    agent.request2obs("Wind 1")
    agent.hand = [
        parse_tile_str(text)
        for text in (
            "1筒", "2筒", "3筒", "4筒", "5筒", "6筒", "7筒", "8筒",
            "2条", "2条", "白", "白", "白",
        )
    ]
    agent._hand_embedding_update()
    agent.turn_count[0] = 1
    agent.discard_called = True
    agent.curtile = parse_tile_str("9筒")
    agent.curTile = str(agent.curtile)
    return agent


class FuritenTest(unittest.TestCase):
    def test_own_discard_blocks_every_ron_wait_but_not_tsumo(self):
        agent = make_waiting_agent()
        self.assertTrue(agent._check_Hu(isZimo=False))

        # 自己打过 6 筒后，即使现在别人打的是另一张听牌 9 筒，也不能荣和。
        agent.discarded_tile_kinds.add(agent._tile_kind(parse_tile_str("6筒")))
        self.assertTrue(agent.is_ron_furiten())
        self.assertFalse(agent._check_Hu(isZimo=False))
        self.assertTrue(agent._check_Hu(isZimo=True))

    def test_called_discard_stays_in_furiten_history(self):
        agent = FeatureAgent(Wind.South, 0)
        agent.request2obs("Wind 1")
        agent.hand = [parse_tile_str("6筒")]
        agent.request2obs("Player 0 Play 6筒")
        self.assertIn(agent._tile_kind(parse_tile_str("6筒")), agent.discarded_tile_kinds)

        agent.request2obs("Player 1 Peng 6筒")
        self.assertEqual(agent.history[0], [])
        self.assertIn(agent._tile_kind(parse_tile_str("6筒")), agent.discarded_tile_kinds)

    def test_temporary_furiten_ends_on_own_draw(self):
        agent = make_waiting_agent()
        agent.mark_ron_passed()
        self.assertTrue(agent.temporary_furiten)
        self.assertFalse(agent._check_Hu(isZimo=False))

        observation = agent.request2obs("Draw 9筒")
        self.assertFalse(agent.temporary_furiten)
        self.assertEqual(
            observation["action_mask"][FeatureAgent.OFFSET_ACT["Hu"]], 1
        )

    def test_riichi_passed_ron_is_permanent_until_round_end(self):
        agent = make_waiting_agent()
        agent.isLiZhi[0] = True
        agent.mark_ron_passed()
        agent.temporary_furiten = False
        self.assertTrue(agent.riichi_furiten)
        self.assertTrue(agent.is_ron_furiten())
        self.assertTrue(agent._check_Hu(isZimo=True))

    def test_game_records_pass_only_for_a_completing_tile(self):
        game = ThreePlayerMahjong()
        game.agents = [
            make_waiting_agent(),
            make_waiting_agent(),
            FeatureAgent(Wind.West, 2),
        ]
        game.agents[1].ron_shape_available = True
        game.agents[2].ron_shape_available = False

        game._mark_passed_ron({1: ["Pass"], 2: ["Pass"]})
        self.assertTrue(game.agents[1].temporary_furiten)
        self.assertFalse(game.agents[2].temporary_furiten)

    def test_no_yaku_completion_still_causes_temporary_furiten(self):
        agent = FeatureAgent(Wind.South, 0)
        agent.request2obs("Wind 1")
        agent.hand = [
            parse_tile_str(text)
            for text in (
                "1筒", "1筒", "1筒", "2筒", "3筒", "4筒", "5筒", "6筒",
                "7筒", "8筒", "9筒", "2条", "2条",
            )
        ]
        agent.turn_count[0] = 1
        agent.discard_called = True
        observation = agent.request2obs("Player 1 Play 7筒")
        self.assertTrue(agent.ron_shape_available)
        self.assertEqual(
            observation["action_mask"][FeatureAgent.OFFSET_ACT["Hu"]], 0
        )

        game = ThreePlayerMahjong()
        game.agents = [agent]
        game._mark_passed_ron({0: ["Pass"]})
        self.assertTrue(agent.temporary_furiten)

    def test_claiming_tile_clears_temporary_furiten(self):
        agent = FeatureAgent(Wind.South, 0)
        agent.request2obs("Wind 1")
        agent.hand = [
            parse_tile_str("6筒"),
            parse_tile_str("6筒"),
            parse_tile_str("1筒"),
        ]
        agent.temporary_furiten = True
        agent.request2obs("Player 1 Play 6筒")
        agent.request2obs("Player 0 Peng 6筒")
        self.assertFalse(agent.temporary_furiten)

    def test_red_five_and_normal_five_are_same_furiten_tile(self):
        agent = make_waiting_agent()
        normal = agent._tile_kind(parse_tile_str("5筒"))
        red = agent._tile_kind(parse_tile_str("红5筒"))
        self.assertEqual(normal, red)

    def test_game_core_rejects_ron_while_furiten(self):
        agent = make_waiting_agent()
        agent.temporary_furiten = True
        game = ThreePlayerMahjong()
        game.agents = [agent]
        with self.assertRaises(Error):
            game._checkHu(0, isZimo=False)


if __name__ == "__main__":
    unittest.main()
