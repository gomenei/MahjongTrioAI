from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
from mahjong_env.tile import Wind, Suit, MeldType, Meld, Tile
from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.caculate.port import MahjongFanCalculator
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
        self.isYiFa : List[bool]              # 是否一发
        self.isLingShang : bool                 # 是否是岭上牌
        self.isQiangGang : bool                 # 是否是抢杠牌

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
        self.debug = True
    
    # 初始化游戏
    def reset(self, walls: List[Tile] = None, current_player = 0, config = None):
        if config:
            self.prevailing_wind = config['prevailing_wind']
            self.round_number = config['round_number']
            self.honba = config['honba']
            self.riichi_sticks = config['riichi_sticks']
        else:
            self.prevailing_wind = Wind.East        # 场风
            self.round_number = 1                   # 当前局数
            self.honba = 0                          # 本场数
            self.riichi_sticks = 0                  # 立直棒数量
            self.current_player = current_player    # 当前玩家索引 (0-2)
        self.turn_count = [0 for _ in range(3)]     # 巡目
        self.isLiZhi = [False for _ in range(3)]    # 是否立直
        self.isYiFa = [False for _ in range(3)]     # 番
        self.discard_called = False
        self.reward = None
        self.done = False

        # 初始化玩家
        wind_order = [Wind.East, Wind.South, Wind.West]
        self.agents = []
        for i in range(3):
            index = wind_order.index(self.prevailing_wind)
            player_wind = wind_order[(index + i) % 3]
            self.agents.append(FeatureAgent(player_wind))
        # 生成牌墙
        if walls:
            self.wall = walls
        else:
            self.initialize_wall()

        # 翻出dora
        self.numdora = 0
        self.dead_wall = self.wall[-18:]
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
        self.dora.append(self.dead_wall[8 + self.numdora * 2].get_dora())
        self.lidora.append(self.dead_wall[9 + self.numdora * 2].get_dora())
        self.numdora += 1

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
                elif response[0] == 'Play':
                    self.hands[self.current_player].append(self.curTile)
                    self._discard(self.current_player, tile)
                elif response[0] == 'AnKang' and not self.WallLast:
                    self.hands[self.current_player].append(self.curTile)
                    self._concealedKong(self.current_player, tile)
                elif response[0] == 'BuKang' and not self.WallLast:
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

                for j in range(1, 3):
                    i = (self.current_player + j) % 3
                    if t[i][0] == 'Hu':
                        self._checkHu(i, isZimo=False)
                        break
                else:
                    for j in range(1, 3):
                        i = (self.current_player + j) % 3
                        if t[i][0] == 'Kang' and self._canDrawTile() and not self.WallLast:
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == 'Peng' and not self.WallLast:
                            self._peng(i, self.curTile)
                            break
                    else:
                        i = (self.current_player + 1) % 3
                        for j in range(1, 3):
                            i = (self.current_player + j) % 3
                            if t[i][0] != 'Pass': raise Error(i)
                        if self.WallLast:
                            self.obs = {i : self.agents[i].request2obs('Huang') for i in range(3)}
                            self.reward = [0, 0, 0, 0]
                            self.done = True
                        else:
                            self.current_player = (self.current_player + 1) % 3
                            self._draw(self.current_player)

            elif self.state == 3:
                # After AnKang/BuKang/BoBei, prepare to Hu/Pass
                responses = {i : self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(3) if i != self.current_player}
                for j in range(1, 3):
                    i = (self.current_player + j) % 3
                    if responses[i] == 'Hu':
                        self._checkHu(i, isZimo=True)
                        break
                else:
                    for j in range(1, 3):
                        i = (self.current_player + j) % 3
                        if responses[i] != 'Pass': raise Error(i)
                    self._draw(self.current_player, from_bottom=True)
        except Error as e:
            player = e.args[0]
            self.obs = {i : self.agents[i].request2obs('Player %d Invalid' % player) for i in range(3)}
            self.reward = [10] * 3
            self.reward[player] = -8000
            self.done = True
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

    '''处理玩家出牌的逻辑'''
    def _discard(self, player: int, tile: Tile):
        if tile not in self.hands[player]: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "Discard", tile)
        self.hands[player].remove(tile)
        self.WallLast = not self._canDrawTile()
        self.isYiFa[player] = False
        self.isLingShang = False
        self.isQiangGang = False
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(3) if i != player}
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
        # self.agents[i].request2obs(' '.join(['Deal', *hand]))
        self.drawAboutKong = False
        if self.debug:
            self.render()
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
        self.turn_count[player] += 1
        self.WallLast = not self._canDrawTile()
        self.state = 1
        self.curTile = tile
        for i in range(3):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player : self.agents[player].request2obs('Draw %s' % tile)}

    def _peng(self, player: int, tile: Tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3: raise Error(player)
        for i in range(3): self.hands[player].remove(tile)

        if self.debug:
            print(self.agent_names[player], "Peng", tile)

        self.packs[player].append(Meld(MeldType.Pon, [tile] * 3, tile))
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.state = 0
        self.current_player = player
        for i in range(3):
            if i != player:
                self.agents[i].request2obs('Player %d Peng' % player)
        self.obs = {player : self.agents[player].request2obs('Player %d Peng' % player)}

    def _kong(self, player: int, tile: Tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        if self.debug:
            print(self.agent_names[player], "Kang", tile)
        self.packs[player].append(Meld(MeldType.OpenKan, [tile] * 4, tile))
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.current_player = player
        for agent in self.agents:
            agent.request2obs('Player %d Kang' % player)
        self._plusdora()
        self._draw(player, from_bottom=True)

    def _concealedKong(self, player: int, tile: Tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)

        if self.debug:
            print(self.agent_names[player], "AnKang", tile)

        self.packs[player].append(Meld(MeldType.ClosedKan, [tile] * 4, taken_tile=tile))
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.state = 3
        self.current_player = player
        self.isQiangGang = False
        self.isLingShang = False
        self.curTile = tile
        for i in range(3):
            if i != player:
                self.agents[i].request2obs('Player %d AnKang' % player)
        self.agents[player].request2obs('Player %d AnKang %s' % (player, tile))
        self._plusdora()
    
    def _promoteKong(self, player: int, tile: Tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i].type == MeldType.Pon and self.packs[player][i].taken_tile == tile:
                idx = i
        if idx < 0: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "BuKang", tile)

        self.hands[player].remove(tile)
        self.packs[player][idx] = Meld(MeldType.OpenKan, [tile] * 4, tile)
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.state = 3
        self.current_player = player
        self.isQiangGang = True
        self.isLingShang = False
        self.curTile = tile
        self.agents[player].request2obs('Player %d BuKang %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d BuKang %s' % (player, tile)) for i in range(3) if i != player}
        self._plusdora()

    def _drawNorth(self, player: int):
        tile = Tile(Suit.Honors, value=4)
        if tile not in self.hands[player]: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "Pei")
        self.hands[player].remove(tile)
        self.discard_called = True
        self.isYiFa = [False for _ in range(3)]
        self.packs[player].append(Meld(MeldType.Pei, [tile], tile))
        self.state = 3
        self.current_player = player
        self.isQiangGang = False
        self.isLingShang = True
        self.curTile = tile
        self.agents[player].request2obs('Player %d Pei' % player)
        self.obs = {i : self.agents[i].request2obs('Player %d Pei' % player) for i in range(3) if i != player}

    def _riichi(self, player: int, tile: Tile):
        if tile not in self.hands[player]: raise Error(player)
        if self.debug:
            print(self.agent_names[player], "Riichi", tile)
        self.hands[player].remove(tile)
        self.riichi_sticks += 1
        self.isLiZhi[player] = True
        self.isYiFa[player] = True 
        self.WallLast = not self._canDrawTile()
        self.isLingShang = False
        self.isQiangGang = False
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Riichi %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d Riichi %s' % (player, tile)) for i in range(3) if i != player}        
        pass

    def _checkHu(self, player: int, isZimo: bool):
        try:
            # 计算番数和符数
            isLiangLiZhi = self.isLiZhi[player] and self.turn_count[player] == 1 and (not self.discard_called)
            isTianHe = self.turn_count[player] == 1 and self.agents[player].seatWind == self.prevailing_wind
            isDiHe = self.turn_count[player] == 0 and self.agents[player].seatWind != self.prevailing_wind
            isLast = len(self.wall) == 14
            # for hand in self.hands[player]:
            #     print(str(hand), end='')
            # print(self.packs[player])
            # print(self.curTile)
            # print(self.prevailing_wind)
            # print(self.agents[player].seatWind)
            # print(self.dora)
            # print(self.lidora)
            fan, fu = MahjongFanCalculator(
                hand = self.hands[player],                  # 手牌
                pack = self.packs[player],                  # 副录（包含碰、明杠、暗杠、拨北）
                winTile = self.curTile,                     # 炮张
                prevailing_wind = self.prevailing_wind,     # 场风
                seat_wind = self.agents[player].seatWind,   # 自风
                dora = self.dora,                           # dora
                lidora = self.lidora,                       # 里dora
                isLiangLiZhi = isLiangLiZhi,                # 是否两立直
                isLiZhi = self.isLiZhi[player],             # 是否立直
                isYiFa = self.isYiFa[player],               # 是否一发
                isLingShang = self.isLingShang,             # 是否岭上
                isZimo = isZimo,                            # 是否自摸
                isLast = isLast,                            # 是否海底
                isQiangGang = self.isQiangGang,             # 是否抢杠
                isTianHe = isTianHe,                        # 是否天和
                isDiHe = isDiHe,                            # 是否地和
            )
            if self.debug:
                print(self.agent_names[player], "Hu, Fans:", fan, "Fus:", fu)
        except Exception as e:
            raise Error(player)

    def render(self):
        for i in range(3):
            print(self.agent_names[i])
            sorted_hand = sorted(self.hands[i].copy())
            for hand in sorted_hand:
                print(hand, end=' ')
            print()
