import unittest

from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.tile import Wind


class LastTileActionMaskTest(unittest.TestCase):
    def make_tenpai_agent(self):
        agent = FeatureAgent(Wind.South, 0)
        agent.request2obs("Wind 1")
        agent.hand = [
            parse_tile_str(text)
            for text in (
                "1筒", "2筒", "3筒", "4筒", "5筒", "6筒", "7筒",
                "8筒", "2条", "2条", "白", "白", "白",
            )
        ]
        agent._hand_embedding_update()
        agent.turn_count[0] = 1
        agent.discard_called = True
        return agent

    def test_last_draw_does_not_offer_pei_or_riichi(self):
        agent = self.make_tenpai_agent()
        # Draw 会先把 tileWall 从 1 减到 0，这对应核心引擎的 WallLast。
        agent.tileWall = 1
        observation = agent.request2obs("Draw 北")
        mask = observation["action_mask"]

        self.assertEqual(mask[FeatureAgent.OFFSET_ACT["Pei"]], 0)
        riichi_start = FeatureAgent.OFFSET_ACT["Riichi"]
        riichi_end = FeatureAgent.OFFSET_ACT["Pei"]
        self.assertEqual(mask[riichi_start:riichi_end].sum(), 0)

    def test_non_last_draw_still_offers_pei_and_riichi(self):
        agent = self.make_tenpai_agent()
        agent.tileWall = 2
        observation = agent.request2obs("Draw 北")
        mask = observation["action_mask"]

        self.assertEqual(mask[FeatureAgent.OFFSET_ACT["Pei"]], 1)
        riichi_start = FeatureAgent.OFFSET_ACT["Riichi"]
        riichi_end = FeatureAgent.OFFSET_ACT["Pei"]
        self.assertGreater(mask[riichi_start:riichi_end].sum(), 0)


if __name__ == "__main__":
    unittest.main()
