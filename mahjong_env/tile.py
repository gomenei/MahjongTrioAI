from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class Suit(Enum):
    Manzu = auto()
    Pinzu = auto()
    Souzu = auto()
    Honors = auto()
class Wind(Enum):
    East = auto()
    South = auto()
    West = auto()
    North = auto()
class MeldType(Enum):
    Chi = auto()
    Pon = auto()
    OpenKan = auto()
    ClosedKan = auto()

@dataclass(order=True)  # 自动生成比较运算符
class Tile:
    suit: Suit
    value: int
    is_red: bool = False
    def __post_init__(self):
        """数字牌有效性检查（1-9）"""
        if not self.is_honor and (self.value < 1 or self.value > 9):
            raise ValueError("Tile value must be 1-9 for non-honor tiles")
    @property
    def is_honor(self) -> bool:
        return self.suit == Suit.Honors
    
    @property
    def is_terminal(self) -> bool:
        return not self.is_honor and (self.value == 1 or self.value == 9)
    
    def __str__(self) -> str:
        """实现toString()功能"""
        if self.is_honor:
            honor_names = ["东", "南", "西", "北", "白", "发", "中"]
            return honor_names[self.value - 1]
        else:
            suit_names = {Suit.Manzu: "万", Suit.Pinzu: "筒", Suit.Souzu: "条"}
            return f"{self.value}{suit_names[self.suit]}{'(红)' if self.is_red else ''}"
@dataclass
class Meld:
    type: MeldType
    tiles: List[Tile]
    taken_tile: Optional[Tile] = None  # 被吃/碰/杠的牌