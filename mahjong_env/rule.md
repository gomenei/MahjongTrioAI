### env说明

```
 OFFSET_ACT = {
        'Play' : 0,
        'Peng' : 29,
        'Kang' : 58,
        'AnKang' : 85,
        'BuKang' : 112,
        'DrawNorth' : 139,
        'Riichi' : 140,
        'Ron' : 141,
        'Pass' : 142,
    }

```

```
# 计算番数和符数
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
    isLingShang = self.isLingShang[player],     # 是否岭上
    isZimo = False,                             # 是否自摸
    isLast = False,                             # 是否海底
    isQiangGang = False,                        # 是否抢杠
    isTianHe = False,                           # 是否天和
    isDiHe = False,                             # 是否地和
)

MahjongFanCalculator(
    hand: List[Tile],
    pack: List[Meld],
    winTile: Tile,
    prevailing_wind: Wind,
    seat_wind: Wind,
    dora: List[Tile],
    lidora: List[Tile],
    is...: bool
)
```

还差红五,流局的逻辑,振听的逻辑，

