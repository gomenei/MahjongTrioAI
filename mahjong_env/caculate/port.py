from rule import evaluate_hand, load_patterns
from tile import Tile, Suit, Wind, Meld, MeldType
from typing import List

def _tile_to_str(tile: Tile) -> str:
    """将Tile对象转换为rule.py的字符串表示如1m, 0p, 5z"""
    if tile.is_red and tile.suit != Suit.Honors and tile.value == 5:
        return f"0{tile.suit.name[0].lower()}"  # 红宝牌
    suit_map = {
        Suit.Manzu: "m",
        Suit.Pinzu: "p",
        Suit.Souzu: "s",
        Suit.Honors: "z"
    }
    value = tile.value
    return f"{value}{suit_map[tile.suit]}"

def _parse_melds(pack: List[Meld]) -> str:
    """将Meld列表转换为副露字符串outer"""
    meld_strs = []
    for meld in pack:
        if meld.type == MeldType.Pei:
            continue  # 拔北被剔除
        tiles = meld.tiles
        if meld.type == MeldType.ClosedKan:  # 暗杠，格式如0330m
            suit = _tile_to_str(tiles[0])[-1]
            meld_strs.append(f"0{tiles[0].value}{tiles[0].value}0{suit}")
        elif meld.type == MeldType.OpenKan:  # 明杠，格式如4444m
            meld_strs.append("".join([_tile_to_str(t) for t in tiles]))
        elif meld.type == MeldType.Pon:  # 刻子
            meld_strs.append("".join([_tile_to_str(t) for t in tiles]))
    return " ".join(meld_strs)


#计算番数和符数
def MahjongFanCalculator(hand: List[Tile],pack: List[Meld],winTile: Tile,prevailing_wind: Wind,seat_wind: Wind,dora: List[Tile],lidora: List[Tile],
isLiangLiZhi: bool,isLiZhi: bool,isYiFa: bool,isLingShang: bool,isZimo: bool,isLast: bool,isQiangGang: bool,isTianHe: bool,isDiHe: bool) -> tuple[int, int]:
    # 转换手牌和进张
    inner_tiles = "".join([_tile_to_str(t) for t in hand])
    jinzhang = _tile_to_str(winTile)
    # 转换副露
    outer_str = _parse_melds(pack)
    # 转换宝牌
    dora_str = "".join([_tile_to_str(t) for t in dora])
    lidora_str = "".join([_tile_to_str(t) for t in lidora])
    # 转换场风和自风Wind枚举转0-3
    input = {
        "inner": inner_tiles,
        "jinzhang": jinzhang,
        "outer": outer_str,
        "selfwind": seat_wind.value - 1,  # Wind.East对应0
        "placewind": prevailing_wind.value - 1,
        "dora": dora_str,
        "innerdora": lidora_str,
        "beidora": sum(1 for m in pack if m.type == MeldType.Pei),  # 拔北数量
        "isReach": isLiZhi,
        "isWReach": isLiangLiZhi,
        "isYiFa": isYiFa,
        "isTsumo": isZimo,
        "haidi": isLast,
        "hedi": isLast,
        "isLingShang": isLingShang,
        "isQiangGang": isQiangGang,
        "tianhe": isTianHe,
        "dihe": isDiHe
    }
    load_patterns()
    result = evaluate_hand(input)
    return result.get("fan", 0), result.get("fu", 0)
