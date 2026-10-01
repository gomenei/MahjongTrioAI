import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
from mahjong_env.tile import Tile, Suit, Meld, Wind, MeldType, can_declare_kan
from collections import defaultdict
from mahjong_env.util import (
    MahjongAgariCalculator,
    MahjongFanCalculator,
    MahjongXiangTingCalculator,
)

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

    OBS_SIZE = 6
    RICH_OBS_SIZE = 101
    MATCH_OBS_SIZE = 121
    ACT_SIZE = 177
    LEGACY_OBSERVATION_KEY = 'observation'
    RICH_OBSERVATION_KEY = 'rich_observation'
    MATCH_OBSERVATION_KEY = 'match_observation'
    OBSERVATION_CONTRACT = 'current_hand_after_draw_and_kan_v1'

    OFFSET_MATCH = {
        'WIND': 101,          # 东/南场 one-hot
        'ROUND': 103,         # 当前场第 1--4 局 one-hot
        'HONBA': 107,         # 本场数，归一化标量
        'RIICHI_STICKS': 108, # 场上立直棒，归一化标量
        'SCORES': 109,        # 自家/下家/对家分数
        'SCORE_DIFF': 112,    # 下家/对家相对自家的点差
        'SELF_RANK': 114,     # 当前自家顺位 one-hot
        'DEALER': 117,        # 庄家相对位置 one-hot
        'PROGRESS': 120,      # 半庄进度
    }

    OFFSET_RICH = {
        # 0..5 与原 observation 逐位完全一致。
        'LEGACY': 0,
        'RIVER': 6,          # 3 家 x 4 层 unary 张数
        'RIVER_TURN': 18,    # 3 家 x 4 个舍牌顺序 bucket
        'MELD': 30,          # 3 家 x 4 层 unary 公开张数（不含拔北）
        'MELD_TYPE': 42,     # 3 家 x 3 类：碰/明杠/暗杠
        'MELD_TURN': 51,     # 3 家 x 4 个副露顺序 bucket
        'PEI': 63,           # 3 家 x 4 层 unary 拔北数
        'PEI_TURN': 75,      # 3 家 x 4 个拔北顺序 bucket
        'DORA': 87,          # 5 层 unary 宝牌
        'RIICHI': 92,        # 3 家立直宣言牌
        'REMAINING': 95,     # 6 个剩余牌数 bucket
    }

    OFFSET_OBS = {
        'PREVALENT_WIND' : 0,
        'SEAT_WIND' : 1,
        'HAND' : 2,
    }

    OFFSET_ACT = {
        'Play' : 0,
        'Peng' : 29,
        'Kang' : 58,
        'AnKang' : 87,
        'BuKang' : 116,
        'Riichi' : 145,
        'Pei' : 174,
        'Hu' : 175,
        'Pass' : 176,
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

    OFFSET_HONOR = 19

    def __init__(self, seatWind: Wind, player: int):
        self.seatWind = seatWind
        self.player = player
        self.obs = np.zeros((self.OBS_SIZE, 30))
        self.packs : List[List[Meld]] = [[] for i in range(3)]    # packs记录副录，0代表自家，1代表下家，2代表对家
        self.turn_count = [0 for _ in range(3)]
        self.isLiZhi = [False for _ in range(3)]    # 是否立直
        self.isLiangLiZhi = [False for _ in range(3)]   # 是否两立直
        self.LiZhi_turn = [-1 for _ in range(3)] # 立直巡目
        self.isYiFa = [False for _ in range(3)]
        self.discard_called = False             # 是否有人鸣牌
        self.isLingShang = False
        self.isQiangGang = False
        self.hand : List[Tile] = []
        self.history = [[] for i in range(3)]
        self.history_orders = [[] for i in range(3)]
        # 可见牌河中的牌会在被碰/杠后移除，但振听判定必须永久记住
        # 自己实际打过的所有牌。赤五与普通五属于同一种牌。
        self.discarded_tile_kinds = set()
        self.temporary_furiten = False
        self.riichi_furiten = False
        # 当前他家牌是否能完成牌型。即使无役、没有 Hu 动作，见逃仍会
        # 形成同巡振听；暗杠等本来不可抢的事件不会设置此标记。
        self.ron_shape_available = False
        self.riichi_history_order = [-1 for _ in range(3)]
        self.riichi_tile_indices = [-1 for _ in range(3)]
        self.discard_serial = 0
        self.pack_turns = [[] for i in range(3)]
        self.dora : List[Tile] = []
        # 三麻开局第一次摸牌后，雀魂的 left_tile_count 为 54。
        # Draw/Player Draw 会立即减 1，因此初值应为 55，而不是 58。
        self.tileWall = 55
        self.prevailing_wind = Wind.East
        self.round_number = 0
        self.honba = 0
        self.riichi_sticks = 0
        self.scores = [35000, 35000, 35000]
        self.dealer_seat = 0
        self.match_round_index = 0
        self.match_total_rounds = 6
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_HONOR + seatWind.value] = 1

    def set_match_context(
        self,
        *,
        round_number: int,
        honba: int,
        riichi_sticks: int,
        scores,
        dealer_seat: int,
        round_index: int = 0,
        total_rounds: int = 6,
    ):
        """设置一小局之外的公开比赛状态。玩家顺序在编码时转为自家视角。"""
        if len(scores) != 3:
            raise ValueError("三麻比赛必须提供三家分数")
        self.round_number = int(round_number)
        self.honba = int(honba)
        self.riichi_sticks = int(riichi_sticks)
        self.scores = [int(score) for score in scores]
        self.dealer_seat = int(dealer_seat)
        self.match_round_index = int(round_index)
        self.match_total_rounds = max(1, int(total_rounds))

    def _tile_to_index(self, tile: Tile):
        if tile.is_honor:
            honor_map = {1: 20, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26}
            return honor_map[tile.value]
        else:
            if (tile.suit == Suit.Manzu) and (tile.value == 1):
                return 0
            else:
                if (tile.is_red):
                    return 27 if tile.suit == Suit.Pinzu else 28
                return (tile.suit.value * 9) + (tile.value - 1) - 7 - 9
    
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
        t = request.split()
        if t[0] == 'Wind':
            winds = [Wind.East, Wind.South, Wind.West]
            self.prevailing_wind = winds[int(t[1]) - 1]
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_HONOR + int(t[1])] = 1
            return
        if t[0] == 'Dora':
            self.dora.append(parse_tile_str(t[1]))
            return
        if t[0] == 'Deal':
            for tile in t[1:]:
                self.hand.append(parse_tile_str(tile))
            self._hand_embedding_update()
            return
        if t[0] == 'LiuJu':
            self.valid = []
            self.ron_shape_available = False
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Riichi, Pei, Play, AnKang, BuKang
            self.tileWall -= 1
            # 同巡振听持续到自己的下一次摸牌（普通摸牌、岭上或拔北补牌）。
            self.temporary_furiten = False
            self.ron_shape_available = False
            self.curTile = t[1]
            self.curtile = parse_tile_str(t[1])
            self.valid = []
            if self._check_Hu(isZimo=True):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.hand.append(self.curtile)
            # The policy must see the drawn tile as well as its legal actions.
            self._hand_embedding_update()
            # 牌山已经没有可摸牌时，核心规则会拒绝立直和拔北。
            # action mask 必须与 ThreePlayerMahjong.step 的 WallLast 判断一致，
            # 否则模型即使只从合法动作中选择也可能被判非法动作。
            # Riichi
            if self.tileWall > 0 and not self.isLiZhi[0]:
                tiles_lizhi = self._check_LiZhi()
                for tile in tiles_lizhi:
                    self.valid.append(self.OFFSET_ACT['Riichi'] + self._tile_to_index(tile))
            # Pei
            if self.tileWall > 0 and self._check_Pei():
                self.valid.append(self.OFFSET_ACT['Pei'])
            can_kan = self.tileWall > 0 and can_declare_kan(self.packs)
            for tile in self.hand:
                # Play
                if not self.isLiZhi[0]:
                    self.valid.append(self.OFFSET_ACT['Play'] + self._tile_to_index(tile))
                else:
                    self.valid.append(self.OFFSET_ACT['Play'] + self._tile_to_index(self.curtile))
                # AnKang
                if self.hand.count(tile) == 4 and can_kan:
                    self.valid.append(self.OFFSET_ACT['AnKang'] + self._tile_to_index(tile))
            # BuKang
            if can_kan:
                for pack in self.packs[0]:
                    if pack.type != MeldType.Pon:
                        continue
                    # taken_tile 是当初从牌河碰来的牌，赤牌身份未必与现在
                    # 手中用于补杠的第四张牌相同。动作编号必须使用手牌实体。
                    for hand_tile in self.hand:
                        same_kind = (
                            hand_tile.suit == pack.taken_tile.suit
                            and hand_tile.value == pack.taken_tile.value
                        )
                        if not same_kind:
                            continue
                        action = self.OFFSET_ACT['BuKang'] + self._tile_to_index(hand_tile)
                        if action not in self.valid:
                            self.valid.append(action)
            return self._obs()
        # Player N Hu/Riichi/Draw/Play/Peng/Kang/AnKang/BuKang/Pei
        p = (int(t[1]) - self.player + 3) % 3
        if t[2] == 'Draw':
            self.tileWall -= 1
            self.ron_shape_available = False
            return
        if t[2] == 'Invalid':
            self.valid = []
            self.ron_shape_available = False
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            self.ron_shape_available = False
            return self._obs()
        if t[2] == 'Play' or t[2] == 'Riichi':
            if t[2] == 'Riichi':
                declaring_seat = int(t[1])
                self.riichi_sticks += 1
                self.scores[declaring_seat] -= 1000
            self.tileFrom = p
            self.curTile = t[3]
            self.curtile = parse_tile_str(self.curTile)
            self.history[p].append(self.curTile)
            self.discard_serial += 1
            self.history_orders[p].append(self.discard_serial)
            if t[2] == 'Riichi':
                self.isLiZhi[p] = True
                self.isLiangLiZhi[p] = (self.turn_count[p] == 0) and (not self.discard_called)
                self.LiZhi_turn[p] = self.turn_count[p]
                self.riichi_history_order[p] = self.discard_serial
                self.riichi_tile_indices[p] = self._tile_to_index(self.curtile)
            self.turn_count[p] += 1
            self.isLingShang = False
            if p == 0:
                self.discarded_tile_kinds.add(self._tile_kind(self.curtile))
                if t[2] == 'Play':
                    self.isYiFa[0] = False
                else:
                    self.isYiFa[0] = True
                for i, h in enumerate(self.hand):
                    if h == self.curtile and h.is_red == self.curtile.is_red:
                        self.hand.pop(i)
                        break
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Peng/Kang/Pass
                self.valid = []
                self.ron_shape_available = self.can_complete_with_current_tile()
                if self._check_Hu(isZimo=False):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if self.tileWall > 0 and (not self.isLiZhi[0]):
                    if self.hand.count(self.curtile) >= 2:
                        # 要碰的牌是红五，可执行的动作应该不带红五
                        if self.curtile.value == 5 and (not self.curtile.is_honor):
                            if self.curtile.is_red:
                                self.valid.append(self.OFFSET_ACT['Peng'] + self.STRING_TO_INDEX[self.curTile[1:]])
                            else:
                                red_in_hand = False
                                for h in self.hand:
                                    if h == self.curtile and h.is_red:
                                        red_in_hand = True
                                        break
                                if red_in_hand:
                                    self.valid.append(self.OFFSET_ACT['Peng'] + self.STRING_TO_INDEX["红" + self.curTile])
                                    if self.hand.count(self.curtile) >= 3:
                                        self.valid.append(self.OFFSET_ACT['Peng'] + self.STRING_TO_INDEX[self.curTile])
                                else:
                                    self.valid.append(self.OFFSET_ACT['Peng'] + self.STRING_TO_INDEX[self.curTile])
                        else:
                            self.valid.append(self.OFFSET_ACT['Peng'] + self._tile_to_index(self.curtile))
                    if self.hand.count(self.curtile) >= 3 and can_declare_kan(self.packs):
                        self.valid.append(self.OFFSET_ACT['Kang'] + self.STRING_TO_INDEX[self.curTile[1:] if "红" in self.curTile else self.curTile])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Peng':
            tile = parse_tile_str(t[3])
            if tile.is_red:
                tiles = [self.curtile] * 2 + [tile]
            else:
                tiles = [tile] * 2 + [self.curtile]
            self.packs[p].append(Meld(MeldType.Pon, tiles, self.curtile))
            self.pack_turns[p].append(self.discard_serial)
            self._remove_called_discard_from_history()
            self.isYiFa = [False for _ in range(3)]
            self.discard_called = True
            if p == 0:
                # 标准日麻中，同巡振听在自己取得鸣牌时解除。
                self.temporary_furiten = False
                self.ron_shape_available = False
                # Available: Play
                self.valid = []
                for i, h in enumerate(self.hand):
                    if (h == tile) and (h.is_red == tile.is_red):
                        self.hand.pop(i)
                        break
                self.hand.remove(self.curtile)
                self._hand_embedding_update()
                for tile in self.hand:
                    self.valid.append(self.OFFSET_ACT['Play'] + self._tile_to_index(tile))
                return self._obs()
        if t[2] == 'Kang':
            if (not self.curtile.is_honor) and self.curtile.value == 5:
                tiles = [Tile(self.curtile.suit, self.curtile.value, is_red=False)] * 3 + [Tile(self.curtile.suit, self.curtile.value, is_red=True)]
            else:
                tiles = [self.curtile] * 4
            self.packs[p].append(Meld(MeldType.OpenKan, tiles, self.curtile))
            self.pack_turns[p].append(self.discard_serial)
            self._remove_called_discard_from_history()
            self.isYiFa = [False for _ in range(3)]
            self.discard_called = True
            if p == 0:
                self.temporary_furiten = False
                self.ron_shape_available = False
                self.isLingShang = True
                for i in range(3):
                    self.hand.remove(self.curtile)
                self._hand_embedding_update()
            return
        if t[2] == 'AnKang':
            self.curTile = t[3]
            self.curtile = parse_tile_str(self.curTile)
            if (not self.curtile.is_honor) and self.curtile.value == 5:
                tiles = [Tile(self.curtile.suit, self.curtile.value, is_red=False)] * 3 + [Tile(self.curtile.suit, self.curtile.value, is_red=True)]
            else:
                tiles = [self.curtile] * 4
            self.packs[p].append(Meld(MeldType.ClosedKan, tiles, self.curtile))
            self.pack_turns[p].append(self.discard_serial)
            self.isYiFa = [False for _ in range(3)]
            self.discard_called = True
            if p == 0:
                self.isLingShang = True
                for i in range(4):
                    self.hand.remove(self.curtile)
                self._hand_embedding_update()
            else:
                # 国士无双抢暗杠
                self.ron_shape_available = False
                self.valid = [self.OFFSET_ACT['Pass']]
                return self._obs()
            return
        if t[2] == 'BuKang':
            self.curTile = t[3]
            self.curtile = parse_tile_str(self.curTile)
            idx = -1
            for i in range(len(self.packs[p])):
                if self.packs[p][i].type == MeldType.Pon and self.packs[p][i].taken_tile == self.curtile:
                    idx = i
            if idx < 0: raise Exception
            old_pack = self.packs[p][idx]
            tiles = [*old_pack.tiles, self.curtile]
            # 补杠后仍保留原碰牌的来源和横置牌；self.curtile 只是第四张牌。
            self.packs[p][idx] = Meld(
                MeldType.OpenKan,
                tiles,
                old_pack.taken_tile,
                old_pack.from_player,
            )
            self.pack_turns[p][idx] = self.discard_serial
            self.isYiFa = [False for _ in range(3)]
            self.discard_called = True
            if p == 0:
                self.isLingShang = True
                for hand_index, hand_tile in enumerate(self.hand):
                    same_physical_tile = (
                        hand_tile.suit == self.curtile.suit
                        and hand_tile.value == self.curtile.value
                        and hand_tile.is_red == self.curtile.is_red
                    )
                    if same_physical_tile:
                        self.hand.pop(hand_index)
                        break
                else:
                    raise ValueError(f"手中没有补杠牌：{self.curtile}")
                self._hand_embedding_update()
                return
            else:
                self.valid = []
                self.isQiangGang = True
                self.ron_shape_available = self.can_complete_with_current_tile()
                if self._check_Hu(isZimo=False):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                else:
                    self.isQiangGang = False
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Pei':
            self.curtile = Tile(Suit.Honors, value=4)
            self.curTile = str(self.curtile)
            pack_turn = self.discard_serial
            self.turn_count[p] += 1
            self.packs[p].append(Meld(MeldType.Pei, [self.curtile], self.curtile))
            self.pack_turns[p].append(pack_turn)
            self.discard_called = True
            if p == 0:
                self.isYiFa = [False for _ in range(3)]
                self.isLingShang = True
                self.hand.remove(self.curtile)
                self._hand_embedding_update()
                return
            else:
                self.valid = []
                self.ron_shape_available = self.can_complete_with_current_tile()
                if self._check_Hu(isZimo=False):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                else:
                    self.isYiFa = [False for _ in range(3)]
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()

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
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid tile format: {parts[1]}")
        return offset + index

    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        rich_observation = self._build_rich_observation()
        return {
            self.LEGACY_OBSERVATION_KEY: self.obs.copy(),
            self.RICH_OBSERVATION_KEY: rich_observation,
            self.MATCH_OBSERVATION_KEY: self._build_match_observation(
                rich_observation
            ),
            'action_mask': mask
        }

    def _build_match_observation(self, rich_observation=None):
        """构造 121x30 场况特征；前 101 层与 rich_observation 完全一致。"""
        result = np.zeros((self.MATCH_OBS_SIZE, 30), dtype=np.float32)
        if rich_observation is None:
            rich_observation = self._build_rich_observation()
        result[:self.RICH_OBS_SIZE] = rich_observation

        wind_index = 0 if self.prevailing_wind == Wind.East else 1
        result[self.OFFSET_MATCH['WIND'] + wind_index].fill(1)
        round_index = min(max(self.round_number, 0), 3)
        result[self.OFFSET_MATCH['ROUND'] + round_index].fill(1)
        result[self.OFFSET_MATCH['HONBA']].fill(min(max(self.honba, 0), 10) / 10)
        result[self.OFFSET_MATCH['RIICHI_STICKS']].fill(
            min(max(self.riichi_sticks, 0), 5) / 5
        )

        relative_scores = [
            self.scores[(self.player + offset) % 3] for offset in range(3)
        ]
        for offset, score in enumerate(relative_scores):
            result[self.OFFSET_MATCH['SCORES'] + offset].fill(
                min(max(score, 0), 70000) / 70000
            )
        for offset, score in enumerate(relative_scores[1:]):
            normalized = 0.5 + (score - relative_scores[0]) / 100000
            result[self.OFFSET_MATCH['SCORE_DIFF'] + offset].fill(
                min(max(normalized, 0), 1)
            )

        # 同分时以固定座位号作为稳定 tie-break，确保编码可复现。
        ranking = sorted(range(3), key=lambda seat: (-self.scores[seat], seat))
        self_rank = ranking.index(self.player)
        result[self.OFFSET_MATCH['SELF_RANK'] + self_rank].fill(1)
        relative_dealer = (self.dealer_seat - self.player + 3) % 3
        result[self.OFFSET_MATCH['DEALER'] + relative_dealer].fill(1)
        denominator = max(self.match_total_rounds - 1, 1)
        result[self.OFFSET_MATCH['PROGRESS']].fill(
            min(max(self.match_round_index / denominator, 0), 1)
        )
        return result

    def _remove_called_discard_from_history(self):
        """鸣牌后让特征牌河与游戏核心中实际显示的牌河保持一致。"""
        source = getattr(self, 'tileFrom', None)
        if source is None or not self.history[source]:
            return
        last_tile = parse_tile_str(self.history[source][-1])
        same_physical_tile = (
            last_tile == self.curtile
            and last_tile.is_red == self.curtile.is_red
        )
        if same_physical_tile:
            self.history[source].pop()
            self.history_orders[source].pop()

    def _build_rich_observation(self):
        """构造 101x30 全 one-hot 公共信息特征，前 6 层与旧特征逐位相同。"""
        rich = np.zeros((self.RICH_OBS_SIZE, 30), dtype=np.float32)
        rich[:self.OBS_SIZE] = self.obs

        # 牌河张数使用 4 层 unary，额外一层保存该牌最近
        # 出现在全局第几次舍牌，早期与晚期现物会被区分。
        for player in range(3):
            counts = defaultdict(int)
            latest = defaultdict(int)
            for tile_text, order in zip(self.history[player], self.history_orders[player]):
                tile_index = self.STRING_TO_INDEX[tile_text]
                counts[tile_index] += 1
                latest[tile_index] = max(latest[tile_index], order)
            base = self.OFFSET_RICH['RIVER'] + player * 4
            turn_base = self.OFFSET_RICH['RIVER_TURN'] + player * 4
            for tile_index, count in counts.items():
                rich[base:base + min(count, 4), tile_index] = 1
                turn_bucket = min(latest[tile_index], 63) // 16
                rich[turn_base + turn_bucket, tile_index] = 1

        # 副露张数、类型和发生时间分开。拔北不混入碰杠，
        # 使用自己的 12 个 unary 平面。
        meld_type_offset = {
            MeldType.Pon: 0,
            MeldType.OpenKan: 1,
            MeldType.ClosedKan: 2,
        }
        for player, player_packs in enumerate(self.packs):
            pei_count = 0
            pei_latest = 0
            for pack_index, pack in enumerate(player_packs):
                turn = (
                    self.pack_turns[player][pack_index]
                    if pack_index < len(self.pack_turns[player]) else 0
                )
                if pack.type == MeldType.Pei:
                    pei_count += 1
                    pei_latest = max(pei_latest, turn)
                    continue
                if pack.type not in meld_type_offset:
                    continue
                tile_counts = defaultdict(int)
                for tile in pack.tiles:
                    tile_counts[self._tile_to_index(tile)] += 1
                count_base = self.OFFSET_RICH['MELD'] + player * 4
                type_plane = (
                    self.OFFSET_RICH['MELD_TYPE']
                    + player * 3
                    + meld_type_offset[pack.type]
                )
                turn_base = self.OFFSET_RICH['MELD_TURN'] + player * 4
                for tile_index, count in tile_counts.items():
                    rich[count_base:count_base + min(count, 4), tile_index] = 1
                    rich[type_plane, tile_index] = 1
                    turn_bucket = min(turn, 63) // 16
                    rich[turn_base + turn_bucket, tile_index] = 1
            pei_base = self.OFFSET_RICH['PEI'] + player * 4
            if pei_count:
                rich[pei_base:pei_base + min(pei_count, 4), self.STRING_TO_INDEX['北']] = 1
                pei_turn_base = self.OFFSET_RICH['PEI_TURN'] + player * 4
                pei_turn_bucket = min(pei_latest, 63) // 16
                rich[
                    pei_turn_base + pei_turn_bucket,
                    self.STRING_TO_INDEX['北'],
                ] = 1

        # 宝牌最多五张，按翻出数量做 unary 编码。
        dora_counts = defaultdict(int)
        for tile in self.dora:
            dora_counts[self._tile_to_index(tile)] += 1
        dora_base = self.OFFSET_RICH['DORA']
        for tile_index, count in dora_counts.items():
            rich[dora_base:dora_base + min(count, 5), tile_index] = 1

        # 立直平面在宣言牌位置为1，第30格表示已立直；
        # 即使宣言牌后来被鸣走，立直状态仍不会丢失。
        for player in range(3):
            plane = self.OFFSET_RICH['RIICHI'] + player
            if self.isLiZhi[player]:
                rich[plane, 29] = 1
                tile_index = self.riichi_tile_indices[player]
                if tile_index >= 0:
                    rich[plane, tile_index] = 1

        remaining_bucket = min(max(self.tileWall, 0) // 10, 5)
        rich[self.OFFSET_RICH['REMAINING'] + remaining_bucket, 29] = 1
        return rich

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : ] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[str(tile)] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[str(tile)], self.STRING_TO_INDEX[str(tile)]] = 1
    
    def _check_Pei(self):
        tile = Tile(Suit.Honors, value=4)
        if tile in self.hand:
            return True
        else:
            return False
    
    def _check_LiZhi(self):
        tiles = []
        for p in self.packs[0]:
            if p.type in {MeldType.Pon, MeldType.OpenKan}:
                return tiles
        for i, h in enumerate(self.hand):
            hands = self.hand.copy()
            hands.pop(i)
            xiangting = MahjongXiangTingCalculator(hands)
            if xiangting == 0:
                tiles.append(h)
        return tiles

    def _check_Hu(self, isZimo):
        try:
            # 岭上牌与海底互斥；拔北/杠后的补充牌即使令剩余牌数归零，
            # 也只按岭上判定，不能同时向番符库传入海底。
            isLast = (self.tileWall == 0) and (not self.isLingShang)
            isTianHe = self.turn_count[0] == 0 and self.seatWind == Wind.East and (not self.discard_called)
            isDiHe = self.turn_count[0] == 0 and self.seatWind != Wind.East and (not self.discard_called) and isZimo
            fan, fu = MahjongFanCalculator(
                hand = self.hand,                           # 手牌
                pack = self.packs[0],                           # 副录（包含碰、明杠、暗杠、拨北）
                winTile = self.curtile,                     # 炮张
                prevailing_wind = self.prevailing_wind,     # 场风
                seat_wind = self.seatWind,                  # 自风
                dora = [],                                  # dora
                lidora = [],                                # 里dora
                isLiangLiZhi = self.isLiangLiZhi[0],           # 是否两立直
                isLiZhi = self.isLiZhi[0],                     # 是否立直
                isYiFa = self.isYiFa[0],                       # 是否一发
                isLingShang = self.isLingShang,             # 是否岭上
                isZimo = isZimo,                            # 是否自摸
                isLast = isLast,                            # 是否海底
                isQiangGang = self.isQiangGang,             # 是否抢杠
                isTianHe = isTianHe,                        # 是否天和
                isDiHe = isDiHe,                            # 是否地和
            )
            if fan == 0:
                return False
            # 振听只禁止荣和，不禁止自摸。
            if (not isZimo) and self.is_ron_furiten():
                return False
            return True
        except Exception as e:
            raise e

    @staticmethod
    def _tile_kind(tile: Tile):
        """振听按牌种判断，红五和普通五视为相同。"""
        return tile.suit, tile.value

    def _is_discard_furiten(self) -> bool:
        """自己的任一舍牌命中当前整组听牌时，所有荣和牌都被禁止。"""
        for suit, value in self.discarded_tile_kinds:
            candidate = Tile(suit, value)
            if MahjongAgariCalculator(self.hand, self.packs[0], candidate):
                return True
        return False

    def is_ron_furiten(self) -> bool:
        return (
            self.temporary_furiten
            or self.riichi_furiten
            or self._is_discard_furiten()
        )

    def can_complete_with_current_tile(self) -> bool:
        tile = getattr(self, "curtile", None)
        if tile is None:
            return False
        return MahjongAgariCalculator(self.hand, self.packs[0], tile)

    def mark_ron_passed(self):
        """放弃一次原本合法的荣和：本巡振听；立直后则持续到本局结束。"""
        self.temporary_furiten = True
        if self.isLiZhi[0]:
            self.riichi_furiten = True
