"""Compare fast shape calculations with the original mahjong library inputs."""

import random

from mahjong.agari import Agari
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.tile import Meld, MeldType, Suit, Tile, Wind
from mahjong_env.util import (
    MahjongAgariCalculator, MahjongXiangTingCalculator,
    _shanten_for_counts, _tiles_to_34_counts, group_tiles_by_suit,
)


def original_counts(tiles):
    return TilesConverter.string_to_34_array(**group_tiles_by_suit(tiles))


def test_shape_counts_and_shanten_match_library_including_red_fives():
    wall = [Tile(suit, value, copy == 0 and value == 5)
            for suit in Suit for value in range(1, 8 if suit == Suit.Honors else 10)
            for copy in range(4)]
    rng = random.Random(9751)
    for _ in range(250):
        hand = rng.sample(wall, rng.choice((1, 2, 4, 5, 7, 8, 10, 11, 13, 14)))
        expected_counts = original_counts(hand)
        assert _tiles_to_34_counts(hand) == expected_counts
        assert MahjongXiangTingCalculator(hand) == Shanten().calculate_shanten(expected_counts)
        # Reusing the caller's mutable list must not return stale cached results.
        hand.reverse()
        assert MahjongXiangTingCalculator(hand) == Shanten().calculate_shanten(expected_counts)
        hand[0] = rng.choice(wall)
        if max(_tiles_to_34_counts(hand)) <= 4:
            assert MahjongXiangTingCalculator(hand) == Shanten().calculate_shanten(original_counts(hand))
    assert _shanten_for_counts.cache_info().hits > 0


def test_agari_matches_library_with_pon_kans_and_extracted_north():
    for kind, copies in ((MeldType.Pon, 3), (MeldType.OpenKan, 4), (MeldType.ClosedKan, 4)):
        pack = [Meld(kind, [parse_tile_str("白") for _ in range(copies)]),
                Meld(MeldType.Pei, [parse_tile_str("北")], parse_tile_str("北"))]
        hand = [parse_tile_str(text) for text in
                ("1筒", "2筒", "3筒", "4筒", "红5筒", "6筒", "7筒", "8筒", "2条", "2条")]
        for win in ("9筒", "6筒", "北", "1条"):
            tile = parse_tile_str(win)
            expected = Agari().is_agari(original_counts(hand + [tile] + pack[0].tiles),
                                       [[31] * copies])
            assert MahjongAgariCalculator(hand, pack, tile) == expected


def test_river_lookup_preserves_original_tile_indices():
    agent = FeatureAgent(Wind.East, 0)
    for text, index in agent.STRING_TO_INDEX.items():
        assert agent._tile_to_index(parse_tile_str(text)) == index
