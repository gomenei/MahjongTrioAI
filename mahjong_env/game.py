from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
from mahjong_env.tile import Wind, Suit, MeldType, Meld, Tile, can_declare_kan
from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.util import MahjongFanCalculator
import random
import numpy as np

class Error(Exception):
    pass

class ThreePlayerMahjong:
    """三麻（三人麻将）游戏核心类"""
    agent_names = ['player_%d' % i for i in range(3)]

    def __init__(self):
        # 初始化游戏状态
        self.prevailing_wind : Wind    # 场风
        self.round_number : int               # 当前局数
        self.honba : int                      # 本场数
        self.riichi_sticks : int              # 立直棒数量
        self.turn_count : List[int]           # 巡目
        self.discard_called : bool            # 是否有人鸣牌
        self.isLiZhi : List[bool]             # 是否有人立直
        self.isLiangLiZhi: List[bool]         # 是否两立直
        self.isYiFa : List[bool]              # 是否一发
        self.isLingShang : bool                 # 是否是岭上牌
        self.isQiangGang : bool                 # 是否是抢杠牌
        self.isAddDora : bool                  # 是否之前明杠、加杠需要加dora

        # 玩家管理
        self.current_player : int             # 当前玩家索引 (0-2)
        self.hands: List[List[Tile]]
        self.packs: List[List[Meld]]
        
        # 牌山管理
        self.wall: List[Tile]
        self.dead_wall: List[Tile]
        self.dora: List[Tile]
        self.lidora: List[Tile]
        self.numdora: int

        # 玩家实例
        self.players: List[FeatureAgent]    # 三个玩家对象
        self.is_ron_occurred : bool        # 有人胡牌
        
        # 测试用
        self.debug = False
    
    # 初始化游戏
    def reset(self, walls: List[Tile] = None, current_player = 0, config = None):
        if config:
            self.prevailing_wind = config['prevailing_wind']
            self.round_number = config['round_number']
            self.honba = config['honba']
            self.riichi_sticks = config['riichi_sticks']
        else:
            self.prevailing_wind = Wind.East        # 场风
            self.round_number = 0                   # 当前局数（0 表示东一）
            self.honba = 0                          # 本场数
            self.riichi_sticks = 0                  # 立直棒数量
        self.current_player = current_player    # 当前玩家索引 (0-2)，初始庄家
        self.turn_count = [0 for _ in range(3)]     # 巡目
        self.isLiZhi = [False for _ in range(3)]    # 是否立直
        self.isLiangLiZhi = [False for _ in range(3)]
        self.isYiFa = [False for _ in range(3)]     # 一发
        self.isLingShang = False
        self.isQiangGang = False
        self.isAddDora = False
        self.discard_called = False
        self.reward = None
        self.done = False
        self.discards = [[] for _ in range(3)]
        self.riichi_discard_indices = [None, None, None]
        self.scores = list(config.get('scores', [35000, 35000, 35000])) if config else [35000, 35000, 35000]
        self.winner = None
        self.win_by = None
        self.result_message = ""

        self.fans = [0, 0, 0]
        self.fus = [0, 0, 0]

        # 初始化玩家
        wind_order = [Wind.East, Wind.South, Wind.West]
        self.agents = []
        for i in range(3):
            index = wind_order.index(Wind.East)
            player_wind = wind_order[(index + i - self.current_player + 3) % 3]
            self.agents.append(FeatureAgent(player_wind, i))
        if self.debug:
            print("Wind %d" % self.prevailing_wind.value)
        for agent in self.agents:
            agent.request2obs('Wind %d' % self.prevailing_wind.value)
            agent.set_match_context(
                round_number=self.round_number,
                honba=self.honba,
                riichi_sticks=self.riichi_sticks,
                scores=self.scores,
                dealer_seat=self.current_player,
                round_index=(
                    config.get('round_index', 0) if config else 0
                ),
                total_rounds=(
                    config.get('total_rounds', 6) if config else 6
                ),
            )
        
        # 生成牌墙
        if walls:
            self.wall = walls
        else:
            self.initialize_wall()

        # 翻出dora
        self.numdora = 0
        self.dead_wall = self.wall[:18]
        self._plusdora()
        # 发牌
        self._deal()
        return self._obs()

    def initialize_wall(self):
        # 洗牌
        # 添加数牌（万/筒/条）各36张（1-9 × 4）
        # 万字牌只加1万，9万
        self.wall = []
        for suit in [Suit.Manzu, Suit.Pinzu, Suit.Souzu]:
            for value in range(1, 10):  # 1-9
                if (suit == Suit.Manzu) and (value > 1) and (value < 9):
                    continue
                self.wall.extend([Tile(suit, value) for _ in range(4)])
       
        for value in range(1, 8):  # 1=东 2=南 3=西 4=北 5=白 6=发 7=中
                self.wall.extend([Tile(Suit.Honors, value) for _ in range(4)])

        # 3. 标记红宝牌（赤五）
        def mark_red_tile(suit: Suit, value: int):
            """找到指定的牌并标记为红牌"""
            for tile in self.wall:
                if tile.suit == suit and tile.value == value:
                    tile.is_red = True
                    break
        
        mark_red_tile(Suit.Pinzu, 5)
        mark_red_tile(Suit.Souzu, 5)

        # 4. 洗牌
        random.shuffle(self.wall)
        # self.wall = self.wall[:-14]
    
    def _plusdora(self):
        if self.numdora >= 5:
            raise Exception("太多的杠了")
        if self.numdora == 0:
            self.dora = []
            self.lidora = []
        new_dora = self.dead_wall[8 + self.numdora * 2].get_dora()
        self.dora.append(new_dora)
        self.lidora.append(self.dead_wall[9 + self.numdora * 2].get_dora())
        self.numdora += 1
        # 宝牌是公开信息，同步给三个特征 Agent。里宝牌在和牌前
        # 不可见，因此不放入 observation。
        for agent in getattr(self, "agents", []):
            agent.request2obs(f"Dora {new_dora}")

    def step(self, action_dict: List):
        '''
        actions = {
            'player_0': action0,
            'player_1': action1...
        }
        '''
        try:
            if self.state == 0:
                # After Chi/Peng, prepare to Play
                response = self.agents[self.current_player].action2response(action_dict[self.agent_names[self.current_player]]).split()
                if response[0] == 'Play':
                    tile = parse_tile_str(response[1])
                    self._discard(self.current_player, tile)
                else:
                    raise Error(self.current_player)
            elif self.state == 1:
                # After Draw, prepare to Hu/Play/AnKang/BuKang/Pei/Riichi
                response = self.agents[self.current_player].action2response(action_dict[self.agent_names[self.current_player]]).split()
                # print(response)
                if len(response) > 1:
                    tile = parse_tile_str(response[1])
                if response[0] == 'Hu':
                    self._checkHu(self.current_player, isZimo=True)
                else:
                    self.isLingShang = False
                    if self.isAddDora:
                        self.isAddDora = False
                        self._plusdora()
                    if response[0] == 'Play':
                        self.hands[self.current_player].append(self.curTile)
                        self._discard(self.current_player, tile)
                    elif response[0] == 'AnKang' and not self.WallLast and can_declare_kan(self.packs):
                        self.hands[self.current_player].append(self.curTile)
                        self._concealedKong(self.current_player, tile)
                    elif response[0] == 'BuKang' and not self.WallLast and can_declare_kan(self.packs):
                        self.hands[self.current_player].append(self.curTile)
                        self._promoteKong(self.current_player, tile)
                    elif response[0] == 'Pei' and not self.WallLast:
                        self.hands[self.current_player].append(self.curTile)
                        self._drawNorth(self.current_player)
                    elif response[0] == 'Riichi' and not self.WallLast:
                        self.hands[self.current_player].append(self.curTile)
                        self._riichi(self.current_player, tile)
                    else:
                        raise Error(self.current_player)
            elif self.state == 2:
                # After Play/Riichi, prepare to Peng/Kang/Hu/Pass
                responses = {i : self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(3) if i != self.current_player}
                t = {i : responses[i].split() for i in responses}
                self._mark_passed_ron(t)

                for j in range(1, 3):
                    i = (self.current_player + j) % 3
                    if t[i][0] == 'Hu':
                        self._checkHu(i, isZimo=False)
                        break
                else:
                    for j in range(1, 3):
                        i = (self.current_player + j) % 3
                        if t[i][0] == 'Kang' and self._canDrawTile() and not self.WallLast and can_declare_kan(self.packs):
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == 'Peng' and not self.WallLast:
                            tile = parse_tile_str(t[i][1])
                            self._peng(i, self.curTile, tile)
                            break
                    else:
                        i = (self.current_player + 1) % 3
                        for j in range(1, 3):
                            i = (self.current_player + j) % 3
                            if t[i][0] != 'Pass': raise Error(i)
                        if self.WallLast:
                            self.obs = {i : self.agents[i].request2obs('LiuJu') for i in range(3)}
                            self.reward = [0, 0, 0]
                            self.done = True
                            self.result_message = "流局"
                        else:
                            self.current_player = (self.current_player + 1) % 3
                            self._draw(self.current_player)

            elif self.state == 3:
                # After AnKang/BuKang/BoBei, prepare to Hu/Pass
                responses = {i : self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(3) if i != self.current_player}
                self._mark_passed_ron(responses)
                for j in range(1, 3):
                    i = (self.current_player + j) % 3
                    if responses[i] == 'Hu':
                        self.isLingShang = False
                        self._checkHu(i, isZimo=False)
                        break
                else:
                    for j in range(1, 3):
                        i = (self.current_player + j) % 3
                        if responses[i] != 'Pass': raise Error(i)
                    self.isYiFa = [False for _ in range(3)]
                    self.isQiangGang = False
                    if self.isAddDora:
                        self.isAddDora = False
                        self._plusdora()
                    self._draw(self.current_player, from_bottom=True)
        except Error as e:
            player = e.args[0]
            self.obs = {i : self.agents[i].request2obs('Player %d Invalid' % player) for i in range(3)}
            self.reward = [10] * 3
            self.reward[player] = -8000
            self.done = True
            self.result_message = "玩家 %d 提交了非法动作" % player
        return self._obs(), self._reward(), self._done()

    def _obs(self):
        return {self.agent_names[k] : v for k, v in self.obs.items()}
    
    def _reward(self):
        if self.reward: return {self.agent_names[k] : self.reward[k] for k in self.obs}
        return {self.agent_names[k] : 0 for k in self.obs}
    
    def _done(self):
        return self.done
    
    def _canDrawTile(self) -> bool:
        return len(self.wall) > 14

    def _mark_passed_ron(self, responses):
        """见逃任何完成牌型的牌都振听，包括当前无役而不能荣和的牌。"""
        for player, response in responses.items():
            action = response[0] if isinstance(response, (list, tuple)) else response
            if action != 'Hu' and self.agents[player].ron_shape_available:
                self.agents[player].mark_ron_passed()

    '''处理玩家出牌的逻辑'''
    def _discard(self, player: int, tile: Tile):
        if tile not in self.hands[player]: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "Discard", tile)
        for i, hand in enumerate(self.hands[player]):
            if (hand == tile) and (hand.is_red == tile.is_red):
                self.hands[player].pop(i)
                break
        self.WallLast = not self._canDrawTile()
        self.turn_count[player] += 1
        self.isYiFa[player] = False
        self.isLingShang = False
        self.isQiangGang = False
        self.curTile = tile
        self.discards[player].append(tile)
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(3) if i != player}

    def _remove_called_discard(self, player: int, tile: Tile):
        if not self.discards[player]:
            return
        last = self.discards[player][-1]
        if last == tile and last.is_red == tile.is_red:
            called_index = len(self.discards[player]) - 1
            self.discards[player].pop()
            if self.riichi_discard_indices[player] == called_index:
                self.riichi_discard_indices[player] = None
    '''处理玩家初始抓牌'''
    def _deal(self):
        self.hands = [[], [], []]
        self.packs = [[], [], []]
        # 先抓3次4张，再抓1次1张
        for _ in range(3): 
            for i in range(3):
                for _ in range(4):
                    if len(self.wall) > 0:
                        tile = self.wall.pop()
                        self.hands[(self.current_player + i) % 3].append(tile)
                    else:
                        raise Error("牌山已空，无法继续抓牌")
        
        for i in range(3):
            if len(self.wall) > 0:
                tile = self.wall.pop()
                self.hands[(self.current_player + i) % 3].append(tile)
            else:
                raise Error("牌山已空，无法继续抓牌")
        for i in range(3):
            self.agents[i].request2obs(' '.join(['Deal', *[str(h) for h in self.hands[i]]]))
        self.drawAboutKong = False
        if self.debug:
            for i in range(3):
                print(self.agent_names[i], ' '.join(['Deal', *[str(h) for h in self.hands[i]]]))
        self._draw(self.current_player)
    '''处理玩家摸一张牌'''
    def _draw(self, player: int, from_bottom: bool=False):
        if from_bottom:
            tile = self.wall.pop(0)
        else:
            tile = self.wall.pop()
        if self.debug:
            print(self.agent_names[player], "Draw", tile)
        # self.hands[player].append(tile)
        self.WallLast = not self._canDrawTile()
        self.state = 1
        self.curTile = tile
        for i in range(3):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player : self.agents[player].request2obs('Draw %s' % tile)}

    """处理玩家碰牌，curtile是打出的牌，actiontile是玩家的红五牌"""
    def _peng(self, player: int, curtile: Tile, actiontile: Tile):
        from_player = self.current_player
        if self.hands[player].count(curtile) < 2: raise Error(player)
        for i, hand in enumerate(self.hands[player]):
            if (hand == actiontile) and (hand.is_red == actiontile.is_red):
                self.hands[player].pop(i)
                break
        else:
            raise Error(player)
        self.hands[player].remove(curtile)

        if self.debug:
            print(self.agent_names[player], "Peng", curtile)

        if actiontile.is_red:
            tiles = [curtile] * 2 + [actiontile]
        else:
            tiles = [actiontile] * 2 + [curtile]
        self.packs[player].append(Meld(MeldType.Pon, tiles, curtile, from_player))
        self._remove_called_discard(from_player, curtile)
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.state = 0
        self.current_player = player
        for i in range(3):
            if i != player:
                self.agents[i].request2obs('Player %d Peng %s' % (player, actiontile))
        self.obs = {player : self.agents[player].request2obs('Player %d Peng %s' % (player, actiontile))}

    def _kong(self, player: int, tile: Tile):
        if not can_declare_kan(self.packs):
            raise Error(player)
        from_player = self.current_player
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        if self.debug:
            print(self.agent_names[player], "Kang", tile)
        if (not tile.is_honor) and tile.value == 5:
            tiles = [Tile(tile.suit, tile.value, is_red=False)] * 3 + [Tile(tile.suit, tile.value, is_red=True)]
        else:
            tiles = [tile] * 4
        self.packs[player].append(Meld(MeldType.OpenKan, tiles, tile, from_player))
        self._remove_called_discard(from_player, tile)
        self.discard_called = True
        self.isQiangGang = False
        self.isLingShang = True
        self.isAddDora = False
        self.isYiFa = [False for _ in range(3)]
        self.current_player = player
        for agent in self.agents:
            agent.request2obs('Player %d Kang' % player)
        self._plusdora()
        self._draw(player, from_bottom=True)

    def _concealedKong(self, player: int, tile: Tile):
        if not can_declare_kan(self.packs):
            raise Error(player)
        # self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)

        if self.debug:
            print(self.agent_names[player], "AnKang", tile)
        if (not tile.is_honor) and tile.value == 5:
            tiles = [Tile(tile.suit, tile.value, is_red=False)] * 3 + [Tile(tile.suit, tile.value, is_red=True)]
        else:
            tiles = [tile] * 4
        self.packs[player].append(Meld(MeldType.ClosedKan, tiles, taken_tile=tile))
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.state = 3
        self.current_player = player
        self.isQiangGang = False
        self.isLingShang = True
        self.curTile = tile
        self._plusdora()
        self.obs = {
            i: self.agents[i].request2obs('Player %d AnKang %s' % (player, tile))
            for i in range(3) if i != player
        }
        self.agents[player].request2obs('Player %d AnKang %s' % (player, tile))
    
    def _promoteKong(self, player: int, tile: Tile):
        if not can_declare_kan(self.packs):
            raise Error(player)
        # self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i].type == MeldType.Pon and self.packs[player][i].taken_tile == tile:
                idx = i
        if idx < 0: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "BuKang", tile)

        # Tile.__eq__ 有意忽略赤牌，不能用 list.remove，否则普通五和赤五
        # 同时存在时可能删错实体牌。
        removed_tile = None
        for hand_index, hand_tile in enumerate(self.hands[player]):
            same_physical_tile = (
                hand_tile.suit == tile.suit
                and hand_tile.value == tile.value
                and hand_tile.is_red == tile.is_red
            )
            if same_physical_tile:
                removed_tile = self.hands[player].pop(hand_index)
                break
        if removed_tile is None:
            raise Error(player)

        old_pack = self.packs[player][idx]
        tiles = [*old_pack.tiles, removed_tile]
        self.packs[player][idx] = Meld(
            MeldType.OpenKan,
            tiles,
            old_pack.taken_tile,
            old_pack.from_player,
        )
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.state = 3
        self.current_player = player
        self.isQiangGang = True
        self.isLingShang = True
        self.isAddDora = True
        self.curTile = tile
        self.agents[player].request2obs('Player %d BuKang %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d BuKang %s' % (player, tile)) for i in range(3) if i != player}

    def _drawNorth(self, player: int):
        tile = Tile(Suit.Honors, value=4)
        if tile not in self.hands[player]: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "Pei")
        self.hands[player].remove(tile)
        self.packs[player].append(Meld(MeldType.Pei, [tile], tile))
        self.turn_count[player] += 1
        self.state = 3
        self.current_player = player
        self.discard_called = True
        self.isQiangGang = False
        self.isLingShang = True
        self.curTile = tile
        self.agents[player].request2obs('Player %d Pei' % player)
        self.obs = {i : self.agents[i].request2obs('Player %d Pei' % player) for i in range(3) if i != player}

    def _riichi(self, player: int, tile: Tile):
        if tile not in self.hands[player]: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "Riichi", tile)
        
        for i, hand in enumerate(self.hands[player]):
            if (hand == tile) and (hand.is_red == tile.is_red):
                self.hands[player].pop(i)
                break
        self.riichi_sticks += 1
        self.isLiZhi[player] = True
        self.isLiangLiZhi[player] = (self.turn_count[player] == 0) and (not self.discard_called)
        self.turn_count[player] += 1
        self.isYiFa[player] = True 
        self.WallLast = not self._canDrawTile()
        self.isLingShang = False
        self.isQiangGang = False
        self.curTile = tile
        self.discards[player].append(tile)
        self.riichi_discard_indices[player] = len(self.discards[player]) - 1
        self.state = 2
        self.agents[player].request2obs('Player %d Riichi %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d Riichi %s' % (player, tile)) for i in range(3) if i != player}        
        pass

    def _checkHu(self, player: int, isZimo: bool):
        try:
            if (not isZimo) and self.agents[player].is_ron_furiten():
                raise Error(player)
            # 计算番数和符数
            isTianHe = self.turn_count[player] == 0 and self.agents[player].seatWind == Wind.East and (not self.discard_called)
            isDiHe = self.turn_count[player] == 0 and self.agents[player].seatWind != Wind.East and (not self.discard_called) and isZimo
            # 岭上牌与海底互斥。尤其是最后一次拔北后的补充牌，
            # 只能按岭上计算，不能同时设置 is_haitei。
            isLast = (len(self.wall) == 14) and (not self.isLingShang)
            fan, fu = MahjongFanCalculator(
                hand = self.hands[player],                  # 手牌
                pack = self.packs[player],                  # 副录（包含碰、明杠、暗杠、拨北）
                winTile = self.curTile,                     # 炮张
                prevailing_wind = self.prevailing_wind,     # 场风
                seat_wind = self.agents[player].seatWind,   # 自风
                dora = self.dora,                           # dora
                lidora = self.lidora,                       # 里dora
                isLiangLiZhi = self.isLiangLiZhi[player],   # 是否两立直
                isLiZhi = self.isLiZhi[player],             # 是否立直
                isYiFa = self.isYiFa[player],               # 是否一发
                isLingShang = self.isLingShang,             # 是否岭上
                isZimo = isZimo,                            # 是否自摸
                isLast = isLast,                            # 是否海底
                isQiangGang = self.isQiangGang,             # 是否抢杠
                isTianHe = isTianHe,                        # 是否天和
                isDiHe = isDiHe,                            # 是否地和
            )
            if fan == 0:
                raise Error(player)

            self.fans[player] = fan
            self.fus[player] = fu
            self.winner = player
            self.win_by = "自摸" if isZimo else "荣和"
            self.reward = [0, 0, 0]
            self.reward[player] = 1
            self.done = True
            fan_text = "%d 倍役满" % (-fan) if fan < 0 else "%d 番" % fan
            self.result_message = "玩家 %d %s：%s %d 符" % (
                player, self.win_by, fan_text, fu
            )
            self.obs = {}
            if self.debug:
                if fan != self.fans[player] or fu != self.fus[player]:
                    hands = self.hands[player]
                    sorted(hands)
                    for h in hands:
                        print(h, end=' ')
                    print()
                    print(self.agent_names[player], "Hu, Fans:", fan, "Fus:", fu)
                    print("MahjongSoul", "Fans:", self.fans[player], "Fus:", self.fus[player])
                else:
                    print(self.agent_names[player], "Hu")
        except Error:
            raise
        except Exception:
            raise Error(player)

    def get_public_state(self):
        """返回供 GUI 使用的牌局快照。"""
        visible_hands = [hand.copy() for hand in self.hands]
        drawn_tiles = [None, None, None]
        if not self.done and self.state == 1:
            visible_hands[self.current_player].append(self.curTile)
            drawn_tiles[self.current_player] = self.curTile

        return {
            "hands": visible_hands,
            "drawn_tiles": drawn_tiles,
            "packs": [pack.copy() for pack in self.packs],
            "discards": [river.copy() for river in self.discards],
            "riichi_discard_indices": self.riichi_discard_indices.copy(),
            "current_player": self.current_player,
            "state": self.state,
            "dora": self.dora.copy(),
            "remaining": max(0, len(self.wall) - 14),
            "prevailing_wind": self.prevailing_wind,
            "round_number": self.round_number,
            "round_number_zero_based": True,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "riichi": self.isLiZhi.copy(),
            "scores": self.scores.copy(),
            "seat_winds": [agent.seatWind for agent in self.agents],
            "done": self.done,
            "winner": self.winner,
            "result_message": self.result_message,
            "view_player": 0,
            "player_names": ["你", "AI 1", "AI 2"],
            "event": self.result_message,
            "event_id": sum(len(river) for river in self.discards) + sum(self.turn_count),
        }

    def render(self):
        for i in range(3):
            print(self.agent_names[i])
            sorted_hand = sorted(self.hands[i].copy())
            for hand in sorted_hand:
                print(hand, end=' ')
            print()
    
    def set_fan_fu(self, player, fan, fu):
        self.fans[player] = fan
        self.fus[player] = fu
