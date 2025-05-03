import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
from mahjong_env.tile import Tile, Suit, Meld, Wind

def parse_tile_str(tile_str: str) -> Tile:
    """
    将中文 Tile 字符串（如 "5万", "红5筒", "白"）转为 Tile 对象
    
    Args:
        tile_str (str): Tile 的中文字符串表示（如 "红5索"、"东"）
    
    Returns:
        Tile: 对应的 Tile 对象
    
    Raises:
        ValueError: 如果字符串格式不对
    """
    # 先去空格
    tile_str = tile_str.strip()
    
    # 1. 处理字牌（东、南、白等）
    honor_tiles = {
        "东": 1, "南": 2, "西": 3, "北": 4,
        "白": 5, "发": 6, "中": 7,
    }
    
    if tile_str in honor_tiles:
        return Tile(Suit.Honors, honor_tiles[tile_str], is_red=False)

    # 2. 处理数牌（万、筒、条）
    # 检查 "红"（如 "红5筒" → is_red=True）
    is_red = tile_str.startswith("红")
    if is_red:
        tile_str = tile_str[1:]  # 去掉 "红"

    # 匹配数字 + 花色（万/筒/条）
    import re
    match = re.match(r"^(\d+)([万筒条])$", tile_str)
    if not match:
        raise ValueError(f"Invalid tile format: {tile_str}")

    value = int(match.group(1))
    suit_str = match.group(2)

    # 万/筒/条 → Suit 枚举
    suit_map = {"万": Suit.Manzu, "筒": Suit.Pinzu, "条": Suit.Souzu}
    suit = suit_map.get(suit_str)
    if not suit:
        raise ValueError(f"Unknown suit: {suit_str}")

    return Tile(suit, value, is_red=is_red)

class FeatureAgent():

    OBS_SIZE = 3
    ACT_SIZE = 143

    OFFSET_OBS = {
        'SEAT_WIND' : 0,
        'PREVALENT_WIND' : 1,
        'HAND' : 2,
    }

    OFFSET_ACT = {
        'Play' : 0,
        'Peng' : 29,
        'Kang' : 58,
        'AnKang' : 87,
        'BuKang' : 114,
        'Riichi' : 143,
        'Pei' : 172,
        'Hu' : 173,
        'Pass' : 174,
    }

    STRING_TO_INDEX = {
            "1万": 0,
            "9万": 1,
            "1筒": 2,
            "2筒": 3,
            "3筒": 4,
            "4筒": 5,
            "5筒": 6,
            "6筒": 7,
            "7筒": 8,
            "8筒": 9,
            "9筒": 10,
            "1条": 11,
            "2条": 12,
            "3条": 13,
            "4条": 14,
            "5条": 15,
            "6条": 16,
            "7条": 17,
            "8条": 18,
            "9条": 19,
            "东": 20,
            "南": 21,
            "西": 22,
            "北": 23,
            "白": 24,
            "发": 25,
            "中": 26,
            "红5筒": 27, 
            "红5条": 28,  
        }

    def __init__(self, seatWind: Wind):
        self.seatWind = seatWind
        self.obs = np.zeros((self.OBS_SIZE, 28))
        # self.obs[self.OFFSET_OBS['SEAT_WIND']][self._tile_to_index(seatWind)] = 1

    def _tile_to_index(self, tile: Tile):
        if tile.is_honor:
            honor_map = {1: 21, 2: 22, 3: 23, 5: 24, 6: 25, 7: 26}
            return honor_map[tile.value]
        else:
            if (tile.suit == Suit.Manzu) and (tile.value == 1):
                return 0
            else:
                if (tile.is_red):
                    return 27 if tile.suit == Suit.Pinzu else 28
                return (tile.suit.value * 9) + (tile.value - 1) - 7
    
    def _index_to_tile(self, index: int):
        if 20 <= index <= 26:
            honor_reverse_map = {
                20: 1,  # 东
                21: 2,  # 南
                22: 3,  # 西
                23: 4,  # 北
                24: 5,  # 白
                25: 6,  # 发
                26: 7,  # 中
            }
            return Tile(Suit.Honors, honor_reverse_map[index])
        if index == 27:
            return Tile(Suit.Pinzu, 5, is_red=True)
        if index == 28:
            return Tile(Suit.Souzu, 5, is_red=True)
        if index == 0:
            return Tile(Suit.Manzu, 1, is_red=False)
        
        tile_value = (index + 7) % 9 + 1
        suit_value = (index + 7) // 9 + 1

        suit = Suit(suit_value)
        return Tile(suit, tile_value)

    def request2obs(self, request):
        pass

    def action2response(self, action):
        if action < self.OFFSET_ACT['Peng']:
            return 'Play ' + str(self._index_to_tile(action - self.OFFSET_ACT['Play']))
        if action < self.OFFSET_ACT['Kang']:
            return 'Peng ' + str(self._index_to_tile(action - self.OFFSET_ACT['Peng']))
        if action < self.OFFSET_ACT["AnKang"]: 
            return "Kang " + str(self._index_to_tile(action - self.OFFSET_ACT["Kang"]))
        if action < self.OFFSET_ACT["BuKang"]:  # 85 <= action < 112 → AnKang
            return "AnKang " + str(self._index_to_tile(action - self.OFFSET_ACT["AnKang"]))
        if action < self.OFFSET_ACT["Riichi"]:  # 112 <= action < 139 → BuKang
            return "BuKang " + str(self._index_to_tile(action - self.OFFSET_ACT["BuKang"]))
        if action < self.OFFSET_ACT["Pei"]:
            return "Riichi " + str(self._index_to_tile(action - self.OFFSET_ACT["Riichi"]))
        if action == self.OFFSET_ACT["Pei"]:  # 139 → Pei
            return "Pei"
        if action == self.OFFSET_ACT["Hu"]:  # 141 → Hu
            return "Hu"
        if action == self.OFFSET_ACT["Pass"]:  # 142 → Pass
            return "Pass"
        
        else:
            raise ValueError("Invalid action index!")

    def response2action(self, response):
        # 去除首尾空格，按空格拆分为 [Action, Tile(可选)]
        parts = response.strip().split(maxsplit=1)
        action_name = parts[0]  # "Play", "Peng", "Riichi" 等
        try:
            offset = self.OFFSET_ACT[action_name]  # 获取基础偏移量
        except KeyError:
            raise ValueError(f"Unknown action: {action_name}")
        # 不带牌的情况（Pei/Riichi/Hu/Pass）
        if action_name in {"Pei", "Hu", "Pass"}:
            return offset
        # 解析 Tile 部分（需自定义 _parse_tile_str() 方法）
        try:
            index = self.STRING_TO_INDEX[parts[1]]
            if action_name == 'AnKang' and index == 27:
                index = 6
            if action_name == 'AnKang' and index == 28:
                index = 15
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid tile format: {parts[1]}")
        return offset + index

    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.copy(),
            'action_mask': mask
        }
