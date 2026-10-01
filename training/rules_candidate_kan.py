"""Wait-preservation check for the isolated post-riichi concealed-kan candidate."""
from mahjong_env.tile import Meld, MeldType, Suit, Tile
from mahjong_env.util import MahjongAgariCalculator


def winning_kinds(hand, packs):
    kinds = [Tile(Suit.Manzu, 1), Tile(Suit.Manzu, 9)]
    kinds += [Tile(suit, value) for suit in (Suit.Pinzu, Suit.Souzu) for value in range(1, 10)]
    kinds += [Tile(Suit.Honors, value) for value in range(1, 8)]
    visible = list(hand) + [t for pack in packs for t in pack.tiles]
    return {(t.suit.value, t.value) for t in kinds
            if visible.count(t) < 4 and MahjongAgariCalculator(hand, packs, t)}


def can_riichi_ankan(hand_before_draw, packs, drawn, kan_tile):
    """The drawn fourth tile must form a kan without changing a nonempty wait set."""
    if kan_tile != drawn or hand_before_draw.count(kan_tile) != 3:
        return False
    before = winning_kinds(hand_before_draw, packs)
    if not before:
        return False
    consumed = [t for t in hand_before_draw if t == kan_tile] + [drawn]
    after_hand = [t for t in hand_before_draw if t != kan_tile]
    after_packs = [*packs, Meld(MeldType.ClosedKan, consumed, kan_tile)]
    return before == winning_kinds(after_hand, after_packs)
