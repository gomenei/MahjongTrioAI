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
    Pei = auto()

@dataclass
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

    def get_dora(dora_idt: 'Tile') -> 'Tile':
        if dora_idt.is_honor:
            if dora_idt.value <= 4:
                next_val = dora_idt.value % 4 + 1
                return Tile(Suit.Honors, next_val)
            else:
                next_val = dora_idt.value + 1
                if next_val > 7:
                    next_val = 5
                return Tile(Suit.Honors, next_val)
        elif dora_idt.suit == Suit.Manzu:
            return Tile(Suit.Manzu, 1 if dora_idt.value == 9 else 9)
        else:
            if dora_idt.value == 9:
                next_val = 1
            else:
                next_val = dora_idt.value + 1
            return Tile(dora_idt.suit, next_val)

    
    def __eq__(self, other: 'Tile') -> bool:
        return (self.value == other.value) and (self.suit == other.suit)

    def __lt__(self, other: 'Tile') -> bool:
        if self.suit != other.suit:
            return self.suit.value < other.suit.value
        if self.value == 5 and other.value == 5:
            if self.is_red != other.is_red:
                return other.is_red
        
        return self.value < other.value

    def __str__(self) -> str:
        """实现toString()功能"""
        if self.is_honor:
            honor_names = ["东", "南", "西", "北", "白", "发", "中"]
            return honor_names[self.value - 1]
        else:
            suit_names = {Suit.Manzu: "万", Suit.Pinzu: "筒", Suit.Souzu: "条"}
            return f"{'红' if self.is_red else ''}{self.value}{suit_names[self.suit]}"
@dataclass
class Meld:
    type: MeldType
    tiles: List[Tile]
    taken_tile: Optional[Tile] = None  # 被吃/碰/杠的牌
    from_player: Optional[int] = None  # 鸣牌来源，用于标准副露摆放


def can_declare_kan(packs: List[List[Meld]]) -> bool:
    """At most four kans across all players; extracted North tiles are not kans.

    This limit is independent of whether a ruleset uses four-kan abortive draws.
    Count declared melds rather than dora indicators, which may be revealed later.
    """
    return sum(pack.type in (MeldType.OpenKan, MeldType.ClosedKan)
               for player_packs in packs for pack in player_packs) < 4
    
