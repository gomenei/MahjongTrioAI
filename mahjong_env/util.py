from mahjong_env.feature import Suit, Tile
from typing import List
import re
from mahjong_env.tile import Wind, MeldType, Tile, Suit, Meld
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.constants import EAST, NORTH, SOUTH, WEST
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld as meld
from typing import Dict, List
from collections import defaultdict
from functools import lru_cache
from mahjong.hand_calculating.yaku_list.yakuman import (
    KokushiMusou
)
from mahjong.shanten import Shanten
from mahjong.agari import Agari


def convert_string_to_wall(tile_str: str) -> List[Tile]:
    """将字符串格式的牌转换为牌山列表"""
    wall = []
    pattern = re.compile(r'([0-9]+[mpsz])')  # 匹配数字+花色的组合（如8p、1z）
    tokens = pattern.findall(tile_str)  # 分割为['8p','1z','4p','8p',...]
    
    for token in tokens[::-1]:  # 逆序解析（假设牌山从末尾取牌）
        value = int(token[:-1])
        suit_char = token[-1].lower()
        
        # 确定花色
        suit_map = {
            'm': Suit.Manzu,
            'p': Suit.Pinzu,
            's': Suit.Souzu,
            'z': Suit.Honors
        }
        suit = suit_map.get(suit_char)
        if not suit:
            raise ValueError(f"无效的花色字符: {suit_char}")
        
        # 处理红宝牌（红5标记为0，如0p=红5筒）
        is_red = (value == 0) and (suit != Suit.Honors)  # 字牌无红宝牌
        actual_value = 5 if value == 0 else value  # 0转换为5
        
        wall.append(Tile(suit, actual_value, is_red))
    return wall

def convert_string_to_tile(tile_str: str) -> Tile:
    """
    将单张麻将牌字符串转换为 Tile 对象
    格式示例: 
        - "5m" → 5万 
        - "0p" → 红5筒 (is_red=True)
        - "7z" → 7字牌（东风）
    
    Args:
        tile_str: 输入的牌字符串（如 "5m"）
    
    Returns:
        Tile: 对应的麻将牌对象
    
    Raises:
        ValueError: 输入格式无效时抛出异常
    """
    if not tile_str or len(tile_str) < 2:
        raise ValueError(f"无效的牌字符串格式: {tile_str}")

    # 解析数值和花色
    value_part = tile_str[:-1]
    suit_char = tile_str[-1].lower()

    try:
        value = int(value_part)
    except ValueError:
        raise ValueError(f"牌字符串的数值部分必须是数字: {tile_str}")

    # 确定花色
    suit_map = {
        'm': Suit.Manzu,   # 万
        'p': Suit.Pinzu,   # 筒
        's': Suit.Souzu,   # 条
        'z': Suit.Honors   # 字牌
    }
    suit = suit_map.get(suit_char)
    if not suit:
        raise ValueError(f"无效的花色字符: {suit_char}")

    # 处理红宝牌（红5标记为0，如 "0p"=红5筒）
    is_red = (value == 0) and (suit != Suit.Honors)  # 字牌无红宝牌
    actual_value = 5 if (value == 0 and suit != Suit.Honors) else value

    # 检查数值范围
    if suit == Suit.Honors:
        if not (1 <= actual_value <= 7):
            raise ValueError(f"字牌数值必须在1-7之间: {tile_str}")
    else:
        if not (1 <= actual_value <= 9):
            raise ValueError(f"数牌数值必须在1-9之间: {tile_str}")

    return Tile(suit, actual_value, is_red)

def group_tiles_by_suit(tiles: List[Tile]) -> Dict[str, str]:
    # 按 Suit 分类
    grouped = defaultdict(list)
    for tile in tiles:
        if tile.suit == Suit.Manzu:
            suit_key = "man"
        elif tile.suit == Suit.Pinzu:
            suit_key = "pin"
        elif tile.suit == Suit.Souzu:
            suit_key = "sou"
        elif tile.suit == Suit.Honors:
            suit_key = "honors"
        else:
            raise ValueError(f"Invalid suit: {tile.suit}")
        value_str = str(tile.value)
        grouped[suit_key].append(value_str)
    # 排序并组合成字符串
    result = {}
    for suit_key, values in grouped.items():
        sorted_values = sorted(values)  # 保证顺序（如 '345' 而非 '354'）
        result[suit_key] = "".join(sorted_values)
    return result

#计算番数和符数
def MahjongFanCalculator(hand: List[Tile],pack: List[Meld],winTile: Tile,prevailing_wind: Wind,seat_wind: Wind,dora: List[Tile],lidora: List[Tile],
isLiangLiZhi: bool,isLiZhi: bool,isYiFa: bool,isLingShang: bool,isZimo: bool,isLast: bool,isQiangGang: bool,isTianHe: bool,isDiHe: bool) -> tuple[int, int]:
    # The scorer applies this same shape test. Most draw/discard checks cannot
    # win; reject those before constructing its many yaku/configuration objects.
    if not MahjongAgariCalculator(hand, pack, winTile):
        return 0, 0
    calculator = HandCalculator()
    doras = 0
    melds = []
    hands = hand.copy()
    hands.append(winTile)
    for p in pack:
        if p.type == MeldType.Pon:
            meld_type = meld.PON
            opened = True
        elif p.type == MeldType.OpenKan:
            meld_type = meld.KAN
            opened = True
        elif p.type == MeldType.ClosedKan:
            meld_type = meld.KAN
            opened = False
        elif p.type == MeldType.Pei:
            doras += 1
            continue
        else:
            continue
        grouped = group_tiles_by_suit(p.tiles)
        hands.extend(p.tiles)
        tiles = TilesConverter.string_to_136_array(**grouped)
        melds.append(meld(meld_type, tiles, opened=opened))
    grouped = group_tiles_by_suit(hands)
    # print(grouped)
    tiles_136 = TilesConverter.string_to_136_array(**grouped)
    grouped = group_tiles_by_suit([winTile])
    wintile = TilesConverter.string_to_136_array(**grouped)[0]
    winds = [EAST, SOUTH, WEST]
    player_wind = winds[seat_wind.value - 1]
    round_wind = winds[prevailing_wind.value - 1]
    options = OptionalRules(
        has_open_tanyao=True,
    )
    config = HandConfig(
        is_tsumo=isZimo,
        is_riichi=isLiZhi,
        is_ippatsu=isYiFa,
        is_daburu_riichi=isLiangLiZhi,
        is_open_riichi=False,
        is_rinshan=isLingShang,
        is_chankan=isQiangGang,
        is_haitei=isLast and isZimo,
        is_houtei=isLast and (not isZimo),
        is_tenhou=isTianHe,
        is_chiihou=isDiHe,
        is_renhou=False,
        player_wind=player_wind,
        round_wind=round_wind,
        options=options
    )
    result = calculator.estimate_hand_value(
        tiles=tiles_136,
        win_tile=wintile,
        melds=melds,
        config=config,
    )
    for p in pack:
        if p.type == MeldType.Pei:
            hands.append(p.taken_tile)
    for h in hands:
        if h.is_red:
            doras += 1

        for d in dora:
            if d == h:
                doras += 1

        if isLiZhi:
            for d in lidora:
                if d == h:
                    doras += 1
    # HandCalculator 的错误结果没有 han/fu。除常见的未和牌、无役外，
    # 互斥役配置等也会返回错误对象，必须在访问 result.han 前统一处理。
    if result.error:
        return 0, 0
    # print(result)
    if result.han % 13 == 0 and len(result.yaku) <= (result.han // 13):
        if result.fu == 0:
            return -(result.han // 13), 25
        else:
            return -(result.han // 13), result.fu

    return result.han + doras, result.fu
    
#计算向听数
def MahjongXiangTingCalculator(hand: List[Tile]) -> int:
    return _shanten_for_counts(tuple(_tiles_to_34_counts(hand)))


@lru_cache(maxsize=65536)
def _shanten_for_counts(counts: tuple) -> int:
    # Immutable keys survive hand mutations and merge red/ordinary fives only
    # for shape calculations. Each worker owns a bounded, independent cache.
    return Shanten().calculate_shanten(list(counts))


def _tiles_to_34_counts(tiles: List[Tile]) -> List[int]:
    counts = [0] * 34
    for tile in tiles:
        counts[_tile_to_34_index(tile)] += 1
    return counts


def MahjongAgariCalculator(
    hand: List[Tile], pack: List[Meld], win_tile: Tile
) -> bool:
    """只判断牌型是否完成，不要求有役。

    振听看的是整组听牌，而不只是当前打出的牌。这里必须绕过番数计算，
    否则某张能完成牌型但本身无役时可能被漏掉。拔北不是面子，不加入牌型。
    """
    tiles = hand.copy()
    tiles.append(win_tile)
    melds_34 = []
    for p in pack:
        if p.type == MeldType.Pei:
            continue
        if p.type not in {MeldType.Pon, MeldType.OpenKan, MeldType.ClosedKan}:
            continue
        tiles.extend(p.tiles)
        melds_34.append([_tile_to_34_index(tile) for tile in p.tiles])
    tiles_34 = _tiles_to_34_counts(tiles)
    return Agari().is_agari(tiles_34, melds_34)


def _tile_to_34_index(tile: Tile) -> int:
    if tile.suit == Suit.Manzu:
        return tile.value - 1
    if tile.suit == Suit.Pinzu:
        return 9 + tile.value - 1
    if tile.suit == Suit.Souzu:
        return 18 + tile.value - 1
    return 27 + tile.value - 1
