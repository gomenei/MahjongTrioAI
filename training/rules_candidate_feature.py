"""Isolated legality changes; production and frozen experiments retain their agent."""

from mahjong_env.feature import FeatureAgent
from mahjong_env.tile import Suit, Tile
from training.rules_candidate_kan import can_riichi_ankan


class CandidateFeatureAgent(FeatureAgent):
    def request2obs(self, request):
        result = super().request2obs(request)
        if request.startswith('Draw ') and self.isLiZhi[0]:
            # The base agent just appended the actual drawn tile to the hand.
            before = self.hand[:-1]
            start = self.OFFSET_ACT['AnKang']
            end = self.OFFSET_ACT['BuKang']
            self.valid = [a for a in self.valid if not start <= a < end
                          or can_riichi_ankan(before, self.packs[0], self.curtile,
                                              self._index_to_tile(a - start))]
            return self._obs()
        return result

    def _check_LiZhi(self):
        # Three normal draws remain before this player can draw again.
        # scores use absolute seats, whereas isLiZhi uses relative seats.
        if self.tileWall < 3 or self.scores[self.player] < 1000:
            return []
        return super()._check_LiZhi()

    def _check_Pei(self):
        if self.isLiZhi[0] and self.curtile != Tile(Suit.Honors, 4):
            return False
        return super()._check_Pei()
