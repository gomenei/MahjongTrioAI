import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

from mahjong_env.feature import parse_tile_str
from mahjong_env.tile import Meld, MeldType, Tile, Wind


def parse_log_tile(value: str) -> Tile:
    """将雀魂的 0p/0s 赤五格式转换为项目内的 Tile。"""
    if value == "0m":
        value = "红5万"
    elif value == "0p":
        value = "红5筒"
    elif value == "0s":
        value = "红5条"
    else:
        suffix = {"m": "万", "p": "筒", "s": "条", "z": None}
        if len(value) == 2 and value[1] == "z":
            value = ["东", "南", "西", "北", "白", "发", "中"][int(value[0]) - 1]
        elif len(value) == 2 and value[1] in suffix:
            value = value[0] + suffix[value[1]]
    return parse_tile_str(value)


def same_tile(left: Tile, right: Tile) -> bool:
    return (
        left.suit == right.suit
        and left.value == right.value
        and left.is_red == right.is_red
    )


def remove_tile(hand: List[Tile], tile: Tile, from_right: bool = False) -> bool:
    indices = range(len(hand) - 1, -1, -1) if from_right else range(len(hand))
    for index in indices:
        if same_tile(hand[index], tile):
            hand.pop(index)
            return True
    # 雀魂的部分记录不会区分被鸣走的普通五与赤五，允许降级匹配。
    for index, candidate in enumerate(hand):
        if candidate == tile:
            hand.pop(index)
            return True
    return False


class ReplaySession:
    """把雀魂 JSON 牌谱转换成可以任意定位的画面快照。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rounds: List[List[Dict]] = []
        self.round_labels: List[str] = []
        self.round_index = 0
        self.step_index = 0
        self.view_player = 0
        self.player_names = ["玩家 0", "玩家 1", "玩家 2"]
        self._load()

    def _load(self):
        with self.path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        accounts = document.get("head", {}).get("accounts", [])
        for account in accounts:
            seat = account.get("seat")
            if seat in (0, 1, 2):
                self.player_names[seat] = account.get("nickname") or f"玩家 {seat}"

        actions = document["data"]["data"]["actions"]
        state = None
        snapshots = None
        turn_counts = [0, 0, 0]

        for action in actions:
            result = action.get("result")
            if not result:
                continue
            name = result.get("name", "").split(".")[-1]
            data = result.get("data", {})

            if name == "RecordNewRound":
                state = self._new_round_state(data)
                turn_counts = [0, 0, 0]
                snapshots = []
                self.rounds.append(snapshots)
                self.round_labels.append(state["round_label"])
                self._snapshot(snapshots, state, "配牌", turn_counts)
                continue

            if state is None or snapshots is None:
                continue

            if name == "RecordDealTile":
                seat = int(data["seat"])
                tile = parse_log_tile(data["tile"])
                state["hands"][seat].append(tile)
                state["drawn_tiles"][seat] = tile
                state["current_player"] = seat
                state["remaining"] = data.get("left_tile_count", state["remaining"])
                if data.get("doras"):
                    state["dora"] = [parse_log_tile(value) for value in data["doras"]]
                event = f"{self.player_names[seat]} 摸 {tile}"
                self._snapshot(snapshots, state, event, turn_counts)

            elif name == "RecordDiscardTile":
                seat = int(data["seat"])
                tile = parse_log_tile(data["tile"])
                remove_tile(state["hands"][seat], tile, bool(data.get("moqie")))
                state["drawn_tiles"][seat] = None
                state["discards"][seat].append(tile)
                state["current_player"] = seat
                if data.get("is_liqi") or data.get("is_wliqi"):
                    state["riichi"][seat] = True
                    state["riichi_discard_indices"][seat] = len(state["discards"][seat]) - 1
                turn_counts[seat] += 1
                prefix = "立直打" if data.get("is_liqi") or data.get("is_wliqi") else "打"
                event = f"{self.player_names[seat]} {prefix} {tile}"
                self._snapshot(snapshots, state, event, turn_counts)

            elif name == "RecordChiPengGang":
                self._apply_open_meld(state, data)
                seat = int(data["seat"])
                kind = {0: "吃", 1: "碰", 2: "明杠"}.get(data.get("type"), "鸣牌")
                self._snapshot(snapshots, state, f"{self.player_names[seat]} {kind}", turn_counts)

            elif name == "RecordAnGangAddGang":
                seat = int(data["seat"])
                tile = parse_log_tile(data["tiles"])
                if data.get("type") == 3:
                    for _ in range(4):
                        remove_tile(state["hands"][seat], tile)
                    state["packs"][seat].append(Meld(MeldType.ClosedKan, [tile] * 4, tile))
                    kind = "暗杠"
                else:
                    remove_tile(state["hands"][seat], tile)
                    self._promote_meld(state["packs"][seat], tile)
                    kind = "加杠"
                state["drawn_tiles"][seat] = None
                state["current_player"] = seat
                self._snapshot(snapshots, state, f"{self.player_names[seat]} {kind} {tile}", turn_counts)

            elif name == "RecordBaBei":
                seat = int(data["seat"])
                north = parse_log_tile("4z")
                remove_tile(state["hands"][seat], north, bool(data.get("moqie")))
                state["drawn_tiles"][seat] = None
                state["packs"][seat].append(Meld(MeldType.Pei, [north], north))
                state["current_player"] = seat
                self._snapshot(snapshots, state, f"{self.player_names[seat]} 拔北", turn_counts)

            elif name == "RecordHule":
                winners = data.get("hules", [])
                names = [self.player_names[int(item["seat"])] for item in winners]
                state["scores"] = data.get("scores", state["scores"])
                event = " / ".join(names) + " 和牌"
                state["result_message"] = event
                self._snapshot(snapshots, state, event, turn_counts)

            elif name == "RecordNoTile":
                state["result_message"] = "荒牌流局"
                self._snapshot(snapshots, state, "荒牌流局", turn_counts)

            elif name == "RecordLiuJu":
                state["result_message"] = "途中流局"
                self._snapshot(snapshots, state, "途中流局", turn_counts)

        if not self.rounds:
            raise ValueError("没有在该文件中找到 RecordNewRound 牌局记录")

    def _new_round_state(self, data: Dict) -> Dict:
        dealer = int(data.get("ju", 0)) % 3
        hands = [
            [parse_log_tile(value) for value in data.get(f"tiles{seat}", [])]
            for seat in range(3)
        ]
        drawn_tiles = [None, None, None]
        if len(hands[dealer]) % 3 == 2 and hands[dealer]:
            drawn_tiles[dealer] = hands[dealer][-1]
        winds = [Wind.East, Wind.South, Wind.West]
        wind = winds[min(int(data.get("chang", 0)), 2)]
        wind_text = ["东", "南", "西"][wind.value - 1]
        round_number = int(data.get("ju", 0)) + 1
        honba = int(data.get("ben", 0))
        return {
            "hands": hands,
            "drawn_tiles": drawn_tiles,
            "packs": [[], [], []],
            "discards": [[], [], []],
            "riichi_discard_indices": [None, None, None],
            "current_player": int(data.get("operation", {}).get("seat", dealer)),
            "state": 1,
            "dora": [parse_log_tile(value) for value in data.get("doras", [])],
            "remaining": int(data.get("left_tile_count", 54)),
            "prevailing_wind": wind,
            "round_number": round_number,
            "honba": honba,
            "riichi_sticks": int(data.get("liqibang", 0)),
            "riichi": [False, False, False],
            "scores": data.get("scores", [35000, 35000, 35000]),
            "seat_winds": [winds[(seat - dealer) % 3] for seat in range(3)],
            "done": False,
            "winner": None,
            "result_message": "",
            "round_label": f"{wind_text}{round_number}局 {honba}本场",
            "event": "配牌",
            "turn_counts": [0, 0, 0],
        }

    def _apply_open_meld(self, state: Dict, data: Dict):
        seat = int(data["seat"])
        tiles = [parse_log_tile(value) for value in data.get("tiles", [])]
        froms = data.get("froms", [seat] * len(tiles))
        taken_tile = None
        source_seat = None
        for tile, source in zip(tiles, froms):
            if int(source) == seat:
                remove_tile(state["hands"][seat], tile)
            else:
                taken_tile = tile
                source_seat = int(source)
        if taken_tile is None and tiles:
            taken_tile = tiles[-1]
        if source_seat is not None and state["discards"][source_seat]:
            river = state["discards"][source_seat]
            if same_tile(river[-1], taken_tile):
                called_index = len(river) - 1
                river.pop()
                if state["riichi_discard_indices"][source_seat] == called_index:
                    state["riichi_discard_indices"][source_seat] = None
        meld_type = {
            0: MeldType.Chi,
            1: MeldType.Pon,
            2: MeldType.OpenKan,
        }.get(data.get("type"), MeldType.Pon)
        state["packs"][seat].append(Meld(meld_type, tiles, taken_tile, source_seat))
        state["drawn_tiles"][seat] = None
        state["current_player"] = seat

    @staticmethod
    def _promote_meld(melds: List[Meld], tile: Tile):
        for index, meld in enumerate(melds):
            if meld.type == MeldType.Pon and meld.taken_tile == tile:
                melds[index] = Meld(
                    MeldType.OpenKan, meld.tiles + [tile], tile, meld.from_player
                )
                return
        melds.append(Meld(MeldType.OpenKan, [tile] * 4, tile))

    @staticmethod
    def _snapshot(snapshots, state, event, turn_counts):
        state["event"] = event
        state["turn_counts"] = turn_counts.copy()
        snapshot = copy.deepcopy(state)
        snapshot["event_id"] = len(snapshots)
        snapshots.append(snapshot)

    @property
    def state(self) -> Dict:
        result = copy.deepcopy(self.rounds[self.round_index][self.step_index])
        result["view_player"] = self.view_player
        result["player_names"] = self.player_names.copy()
        result["replay_step"] = self.step_index
        result["replay_total"] = len(self.rounds[self.round_index])
        result["replay_round"] = self.round_index
        result["replay_round_total"] = len(self.rounds)
        result["source_file"] = self.path.name
        return result

    def seek(self, step: int):
        maximum = len(self.rounds[self.round_index]) - 1
        self.step_index = max(0, min(int(step), maximum))

    def change_step(self, delta: int):
        self.seek(self.step_index + delta)

    def change_round(self, delta: int):
        self.round_index = max(0, min(self.round_index + delta, len(self.rounds) - 1))
        self.step_index = 0

    def set_view(self, player: int):
        self.view_player = int(player) % 3
