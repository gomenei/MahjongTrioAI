"""Run MahjongTrioAI checkpoints behind MahjongCopilot's MJAI interface."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import copy
import logging
import sys
import threading
from typing import Iterable

import numpy as np
import torch

from bot.bot import Bot
from common.log_helper import LOGGER
from common.mj_helper import MjaiType, MSType, cvt_ms2mjai
from common.utils import BotNotSupportingMode, GameMode


COPILOT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = COPILOT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Append instead of prepend so MahjongCopilot's own gui/common/bot packages win.
    sys.path.append(str(PROJECT_ROOT))

from mahjong_env.feature import FeatureAgent  # noqa: E402
from mahjong_env.tile import MeldType, Suit, Tile, Wind  # noqa: E402
from training.models import load_checkpoint_model  # noqa: E402


ACTION_SIZE = FeatureAgent.ACT_SIZE


def mjai_to_tile(text: str) -> Tile:
    """Convert an MJAI tile (5pr/E/...) into MahjongTrioAI's Tile."""
    if text == "?":
        raise ValueError("A hidden MJAI tile cannot be converted to a local tile")
    honors = {"E": 1, "S": 2, "W": 3, "N": 4, "P": 5, "F": 6, "C": 7}
    if text in honors:
        return Tile(Suit.Honors, honors[text])
    if len(text) not in (2, 3) or text[0] not in "123456789" or text[1] not in "mps":
        raise ValueError(f"Unsupported MJAI tile: {text!r}")
    suit = {"m": Suit.Manzu, "p": Suit.Pinzu, "s": Suit.Souzu}[text[1]]
    return Tile(suit, int(text[0]), is_red=text.endswith("r"))


def tile_to_mjai(tile: Tile) -> str:
    """Convert MahjongTrioAI's Tile into MJAI notation."""
    if tile.is_honor:
        return {1: "E", 2: "S", 3: "W", 4: "N", 5: "P", 6: "F", 7: "C"}[tile.value]
    suit = {Suit.Manzu: "m", Suit.Pinzu: "p", Suit.Souzu: "s"}[tile.suit]
    suffix = "r" if tile.is_red else ""
    return f"{tile.value}{suit}{suffix}"


def same_physical_tile(left: Tile, right: Tile) -> bool:
    return left == right and left.is_red == right.is_red


def same_tile_kind(left: Tile, right: Tile) -> bool:
    return left.suit == right.suit and left.value == right.value


class MjaiFeatureState:
    """Translate a player-view MJAI stream into the existing 101x30 feature state."""

    def __init__(self, seat: int):
        self.seat = seat
        self.feature: FeatureAgent | None = None
        self.last_observation: dict | None = None
        self.pending_reach: set[int] = set()
        self.last_draw: tuple[int, str] | None = None
        self.last_discard: tuple[int, str] | None = None
        self.decision_source: tuple[int, str] | None = None

    def reset(self) -> None:
        self.feature = None
        self.last_observation = None
        self.pending_reach.clear()
        self.last_draw = None
        self.last_discard = None
        self.decision_source = None

    @staticmethod
    def _actual_dora(marker: str) -> str:
        return tile_to_mjai(Tile.get_dora(mjai_to_tile(marker)))

    @staticmethod
    def _internal_tile(text: str) -> str:
        return str(mjai_to_tile(text))

    def _request(self, request: str) -> dict | None:
        if self.feature is None:
            raise RuntimeError(f"Received MJAI action before start_kyoku: {request}")
        observation = self.feature.request2obs(request)
        if observation is not None:
            self.last_observation = observation
        return observation

    def apply(self, event: dict) -> dict | None:
        event_type = event.get("type")
        if event_type == MjaiType.START_GAME:
            self.reset()
            return None
        if event_type == MjaiType.START_KYOKU:
            return self._start_kyoku(event)
        if self.feature is None:
            LOGGER.debug("Ignoring MJAI event before start_kyoku: %s", event)
            return None

        if event_type == MjaiType.DORA:
            dora = self._actual_dora(event["dora_marker"])
            return self._request(f"Dora {self._internal_tile(dora)}")

        if event_type == MjaiType.TSUMO:
            actor = int(event["actor"])
            pai = event.get("pai", "?")
            self.last_draw = (actor, pai)
            self.decision_source = (actor, pai)
            if actor == self.seat:
                if pai == "?":
                    raise ValueError("MahjongCopilot hid the controlled player's draw")
                return self._request(f"Draw {self._internal_tile(pai)}")
            return self._request(f"Player {actor} Draw")

        if event_type == MjaiType.REACH:
            self.pending_reach.add(int(event["actor"]))
            return None

        if event_type == MjaiType.REACH_ACCEPTED:
            return None

        if event_type == MjaiType.DAHAI:
            actor = int(event["actor"])
            pai = event["pai"]
            action = "Riichi" if actor in self.pending_reach else "Play"
            self.pending_reach.discard(actor)
            self.last_discard = (actor, pai)
            self.decision_source = (actor, pai)
            return self._request(
                f"Player {actor} {action} {self._internal_tile(pai)}"
            )

        if event_type == MjaiType.PON:
            actor = int(event["actor"])
            consumed = event.get("consumed", [])
            action_tile = next((tile for tile in consumed if tile.endswith("r")), None)
            if action_tile is None:
                action_tile = consumed[0] if consumed else event["pai"]
            return self._request(
                f"Player {actor} Peng {self._internal_tile(action_tile)}"
            )

        if event_type == MjaiType.DAIMINKAN:
            actor = int(event["actor"])
            return self._request(
                f"Player {actor} Kang {self._internal_tile(event['pai'])}"
            )

        if event_type == MjaiType.ANKAN:
            actor = int(event["actor"])
            consumed = event.get("consumed", [])
            tile = next((item for item in consumed if not item.endswith("r")), consumed[0])
            self.decision_source = (actor, tile)
            return self._request(
                f"Player {actor} AnKang {self._internal_tile(tile)}"
            )

        if event_type == MjaiType.KAKAN:
            actor = int(event["actor"])
            pai = event["pai"]
            self.decision_source = (actor, pai)
            return self._request(
                f"Player {actor} BuKang {self._internal_tile(pai)}"
            )

        if event_type == MjaiType.NUKIDORA:
            actor = int(event["actor"])
            self.decision_source = (actor, "N")
            return self._request(f"Player {actor} Pei")

        if event_type == MjaiType.HORA:
            actor = int(event.get("actor", self.seat))
            return self._request(f"Player {actor} Hu")

        if event_type in (MjaiType.RYUKYOKU, MjaiType.END_KYOKU):
            return self._request("LiuJu")

        if event_type == MjaiType.END_GAME:
            self.reset()
            return None

        LOGGER.debug("MahjongTrioAI ignores unsupported MJAI event: %s", event)
        return None

    def _start_kyoku(self, event: dict) -> None:
        oya = int(event["oya"])
        seat_wind = [Wind.East, Wind.South, Wind.West][(self.seat - oya) % 3]
        self.feature = FeatureAgent(seat_wind, self.seat)
        self.last_observation = None
        self.pending_reach.clear()
        self.last_draw = None
        self.last_discard = None
        self.decision_source = None

        wind_number = {"E": 1, "S": 2, "W": 3, "N": 4}[event["bakaze"]]
        self._request(f"Wind {wind_number}")
        scores = list(event.get("scores", [35000, 35000, 35000]))[:3]
        round_number = max(0, int(event.get("kyoku", 1)) - 1)
        self.feature.set_match_context(
            round_number=round_number,
            honba=int(event.get("honba", 0)),
            riichi_sticks=int(event.get("kyotaku", 0)),
            scores=scores,
            dealer_seat=oya,
            round_index=(wind_number - 1) * 3 + round_number,
            total_rounds=6,
        )

        actual_dora = self._actual_dora(event["dora_marker"])
        self._request(f"Dora {self._internal_tile(actual_dora)}")

        tehais = event["tehais"]
        own_tiles = tehais[self.seat]
        if not own_tiles or any(tile == "?" for tile in own_tiles):
            raise ValueError("start_kyoku does not contain the controlled player's hand")
        internal = " ".join(self._internal_tile(tile) for tile in own_tiles)
        self._request(f"Deal {internal}")
        return None


class BotMahjongTrioAI(Bot):
    """MahjongCopilot Bot backed by this repository's PyTorch checkpoints."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        action_mode: str = "greedy",
    ) -> None:
        super().__init__("MahjongTrioAI")
        self.checkpoint = self._resolve_checkpoint(checkpoint)
        self.device = self._resolve_device(device)
        self.action_mode = action_mode.lower()
        if self.action_mode not in {"greedy", "sample"}:
            raise ValueError("trio_action_mode must be 'greedy' or 'sample'")

        self.model, self.metadata = load_checkpoint_model(
            self.checkpoint, map_location=self.device
        )
        self.model.to(self.device).eval()
        self.observation_key = str(
            self.metadata.get("observation_key", FeatureAgent.LEGACY_OBSERVATION_KEY)
        )
        self.obs_channels = int(
            self.metadata.get("obs_channels", FeatureAgent.OBS_SIZE)
        )
        expected = {
            FeatureAgent.LEGACY_OBSERVATION_KEY: FeatureAgent.OBS_SIZE,
            FeatureAgent.RICH_OBSERVATION_KEY: FeatureAgent.RICH_OBS_SIZE,
            FeatureAgent.MATCH_OBSERVATION_KEY: FeatureAgent.MATCH_OBS_SIZE,
        }
        if self.observation_key not in expected:
            raise ValueError(f"Unsupported observation_key: {self.observation_key}")
        if self.obs_channels != expected[self.observation_key]:
            raise ValueError(
                f"Checkpoint channels {self.obs_channels} do not match "
                f"{self.observation_key} ({expected[self.observation_key]})"
            )

        self.state: MjaiFeatureState | None = None
        self.runtime_operation: dict | None = None
        self.runtime_hand: list[str] = []
        self.runtime_draw: str | None = None
        self.lock = threading.RLock()
        LOGGER.info(
            "Loaded MahjongTrioAI checkpoint %s on %s (%s, %s)",
            self.checkpoint,
            self.device,
            self.metadata.get("model_name", type(self.model).__name__),
            self.observation_key,
        )

    @property
    def supported_modes(self) -> list[GameMode]:
        return [GameMode.MJ3P]

    @property
    def info_str(self) -> str:
        return (
            f"{self.name}: {self.checkpoint.name} | {self.device} | "
            f"{self.observation_key}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device.lower() == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            return torch.device("cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false")
        if resolved.type == "mps" and not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was selected but torch.backends.mps.is_available() is false")
        return resolved

    @staticmethod
    def _resolve_checkpoint(value: str) -> Path:
        raw = Path(value).expanduser()
        candidates = [raw] if raw.is_absolute() else [
            Path.cwd() / raw,
            PROJECT_ROOT / raw,
            COPILOT_ROOT / raw,
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
        rendered = ", ".join(str(path.resolve()) for path in candidates)
        raise FileNotFoundError(f"MahjongTrioAI checkpoint not found; tried: {rendered}")

    def _init_bot_impl(self, mode: GameMode = GameMode.MJ3P):
        if mode != GameMode.MJ3P:
            raise BotNotSupportingMode(mode)
        self.state = MjaiFeatureState(self.seat)
        self.runtime_operation = None
        self.runtime_hand = []
        self.runtime_draw = None

    def set_runtime_context(
        self,
        operation: dict | None,
        my_tehai: Iterable[str] | None,
        my_tsumohai: str | None,
    ) -> None:
        """Receive Mahjong Soul's authoritative legal-operation context."""
        with self.lock:
            self.runtime_operation = copy.deepcopy(operation)
            self.runtime_hand = list(my_tehai or [])
            self.runtime_draw = my_tsumohai

    def react_batch(self, input_list: list[dict]) -> dict | None:
        if not input_list:
            return None
        with self.lock:
            reaction = None
            for index, event in enumerate(input_list):
                reaction = self._react_impl(event, can_act=index == len(input_list) - 1)
            return reaction

    def react(self, input_msg: dict) -> dict | None:
        with self.lock:
            return self._react_impl(input_msg, can_act=input_msg.get("can_act", True))

    def _react_impl(self, input_msg: dict, can_act: bool) -> dict | None:
        if self.state is None:
            return None
        observation = self.state.apply(input_msg)
        if not can_act or observation is None:
            return None
        return self._choose_reaction(observation)

    def _operation_types(self) -> set[int] | None:
        if not self.runtime_operation:
            return None
        operations = self.runtime_operation.get("operationList")
        if operations is None:
            return None
        return {int(operation["type"]) for operation in operations}

    @staticmethod
    def _action_group(action: int) -> str:
        offsets = FeatureAgent.OFFSET_ACT
        if action < offsets["Peng"]:
            return "Play"
        if action < offsets["Kang"]:
            return "Peng"
        if action < offsets["AnKang"]:
            return "Kang"
        if action < offsets["BuKang"]:
            return "AnKang"
        if action < offsets["Riichi"]:
            return "BuKang"
        if action < offsets["Pei"]:
            return "Riichi"
        if action == offsets["Pei"]:
            return "Pei"
        if action == offsets["Hu"]:
            return "Hu"
        return "Pass"

    def _filter_mask(self, feature_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(feature_mask, dtype=np.float32).copy()
        operation_types = self._operation_types()
        if operation_types is None:
            return mask

        ms_type = {
            "Play": MSType.dahai,
            "Peng": MSType.pon,
            "Kang": MSType.daiminkan,
            "AnKang": MSType.ankan,
            "BuKang": MSType.kakan,
            "Riichi": MSType.reach,
            "Pei": MSType.nukidora,
        }
        for action in np.flatnonzero(mask > 0):
            group = self._action_group(int(action))
            if group in ms_type and ms_type[group] not in operation_types:
                mask[action] = 0
            elif group == "Hu" and not ({MSType.zimo, MSType.hora} & operation_types):
                mask[action] = 0
            elif group == "Pass":
                source = self.state.decision_source if self.state else None
                if source is None or source[0] == self.seat:
                    mask[action] = 0

        # Mahjong Soul is authoritative for these unambiguous single actions.
        if {MSType.zimo, MSType.hora} & operation_types:
            mask[FeatureAgent.OFFSET_ACT["Hu"]] = 1
        if MSType.nukidora in operation_types:
            mask[FeatureAgent.OFFSET_ACT["Pei"]] = 1

        if not np.any(mask):
            LOGGER.warning(
                "Mahjong Soul legality and FeatureAgent mask have no overlap; "
                "falling back to FeatureAgent mask. operation types=%s",
                sorted(operation_types),
            )
            return np.asarray(feature_mask, dtype=np.float32).copy()
        return mask

    def _choose_reaction(self, observation: dict) -> dict:
        mask = self._filter_mask(observation["action_mask"])
        legal = np.flatnonzero(mask > 0)
        if len(legal) == 0:
            return {"type": MjaiType.NONE, "actor": self.seat}

        model_input = {
            "observation": torch.as_tensor(
                observation[self.observation_key],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "action_mask": torch.as_tensor(
                mask, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
        }
        with torch.inference_mode():
            logits = self.model(model_input).squeeze(0)
            if logits.shape != (ACTION_SIZE,):
                raise ValueError(
                    f"Checkpoint output must be ({ACTION_SIZE},), got {tuple(logits.shape)}"
                )
            legal_tensor = model_input["action_mask"].squeeze(0) > 0
            logits = logits.masked_fill(~legal_tensor, torch.finfo(logits.dtype).min)
            probabilities = torch.softmax(logits, dim=0)
            if self.action_mode == "sample":
                action = int(torch.multinomial(probabilities, 1).item())
            else:
                action = int(torch.argmax(logits).item())

        reaction = self._action_to_mjai(action)
        reaction["meta_options"] = self._meta_options(probabilities, mask, action)
        # MahjongCopilot has its own optional action randomizer. The checkpoint's
        # greedy/sample setting must remain the sole policy choice.
        reaction["disable_copilot_randomize"] = True
        return reaction

    def _tile_for_action(self, action: int, group: str) -> Tile:
        offset = FeatureAgent.OFFSET_ACT[group]
        return self.state.feature._index_to_tile(action - offset)

    def _is_tsumogiri(self, pai: str) -> bool:
        draw = self.runtime_draw
        if not draw and self.state and self.state.last_draw:
            actor, last_draw = self.state.last_draw
            draw = last_draw if actor == self.seat else None
        return draw == pai

    def _operation_combinations(self, operation_type: int) -> list[list[str]]:
        if not self.runtime_operation:
            return []
        result: list[list[str]] = []
        for operation in self.runtime_operation.get("operationList", []):
            if int(operation.get("type", -1)) != operation_type:
                continue
            for combination in operation.get("combination", []):
                result.append([cvt_ms2mjai(tile) for tile in combination.split("|") if tile])
        return result

    def _matching_combination(
        self,
        operation_type: int,
        tile: Tile,
        prefer_red: bool | None = None,
    ) -> list[str] | None:
        matches = []
        for combination in self._operation_combinations(operation_type):
            try:
                if combination and all(same_tile_kind(mjai_to_tile(item), tile) for item in combination):
                    matches.append(combination)
            except ValueError:
                continue
        if not matches:
            return None
        if prefer_red is not None:
            preferred = [
                combination
                for combination in matches
                if any(item.endswith("r") for item in combination) == prefer_red
            ]
            if preferred:
                return preferred[0]
        return matches[0]

    def _hand_tiles_of_kind(self, tile: Tile) -> list[Tile]:
        hand = self.state.feature.hand
        return [candidate for candidate in hand if same_tile_kind(candidate, tile)]

    def _consume_from_hand(
        self, tile: Tile, count: int, prefer_red: bool | None = None
    ) -> list[str]:
        candidates = self._hand_tiles_of_kind(tile)
        if prefer_red is True:
            candidates.sort(key=lambda item: not item.is_red)
        elif prefer_red is False:
            candidates.sort(key=lambda item: item.is_red)
        if len(candidates) < count:
            raise ValueError(f"Not enough {tile} tiles in tracked hand for call")
        return [tile_to_mjai(item) for item in candidates[:count]]

    def _action_to_mjai(self, action: int) -> dict:
        group = self._action_group(action)
        actor = self.seat
        source = self.state.decision_source

        if group == "Play":
            tile = self._tile_for_action(action, "Play")
            pai = tile_to_mjai(tile)
            return {
                "type": MjaiType.DAHAI,
                "actor": actor,
                "pai": pai,
                "tsumogiri": self._is_tsumogiri(pai),
            }

        if group == "Riichi":
            tile = self._tile_for_action(action, "Riichi")
            pai = tile_to_mjai(tile)
            return {
                "type": MjaiType.REACH,
                "actor": actor,
                "reach_dahai": {
                    "type": MjaiType.DAHAI,
                    "actor": actor,
                    "pai": pai,
                    "tsumogiri": self._is_tsumogiri(pai),
                },
            }

        if group == "Peng":
            tile = self._tile_for_action(action, "Peng")
            consumed = self._matching_combination(
                MSType.pon, tile, prefer_red=tile.is_red
            ) or self._consume_from_hand(tile, 2, prefer_red=tile.is_red)
            return {
                "type": MjaiType.PON,
                "actor": actor,
                "target": source[0],
                "pai": source[1],
                "consumed": consumed,
            }

        if group == "Kang":
            tile = self._tile_for_action(action, "Kang")
            consumed = self._matching_combination(MSType.daiminkan, tile)
            consumed = consumed or self._consume_from_hand(tile, 3)
            return {
                "type": MjaiType.DAIMINKAN,
                "actor": actor,
                "target": source[0],
                "pai": source[1],
                "consumed": consumed,
            }

        if group == "AnKang":
            tile = self._tile_for_action(action, "AnKang")
            consumed = self._matching_combination(MSType.ankan, tile)
            consumed = consumed or self._consume_from_hand(tile, 4)
            return {
                "type": MjaiType.ANKAN,
                "actor": actor,
                "consumed": consumed,
            }

        if group == "BuKang":
            tile = self._tile_for_action(action, "BuKang")
            consumed = self._matching_combination(
                MSType.kakan, tile, prefer_red=tile.is_red
            )
            if consumed is None:
                pon = next(
                    (
                        meld
                        for meld in self.state.feature.packs[0]
                        if meld.type == MeldType.Pon
                        and same_tile_kind(meld.taken_tile, tile)
                    ),
                    None,
                )
                if pon is None:
                    raise ValueError(f"No tracked pon for kakan {tile}")
                consumed = [tile_to_mjai(item) for item in pon.tiles]
            return {
                "type": MjaiType.KAKAN,
                "actor": actor,
                "pai": tile_to_mjai(tile),
                "consumed": consumed,
            }

        if group == "Pei":
            return {"type": MjaiType.NUKIDORA, "actor": actor, "pai": "N"}

        if group == "Hu":
            target, pai = source if source is not None else (actor, "?")
            return {
                "type": MjaiType.HORA,
                "actor": actor,
                "target": target,
                "pai": pai,
            }

        return {"type": MjaiType.NONE, "actor": actor}

    def _option_code(self, action: int) -> str:
        group = self._action_group(action)
        if group == "Play":
            return tile_to_mjai(self._tile_for_action(action, "Play"))
        return {
            "Peng": MjaiType.PON,
            "Kang": "kan_select",
            "AnKang": "kan_select",
            "BuKang": "kan_select",
            "Riichi": MjaiType.REACH,
            "Pei": MjaiType.NUKIDORA,
            "Hu": MjaiType.HORA,
            "Pass": MjaiType.NONE,
        }[group]

    def _meta_options(
        self, probabilities: torch.Tensor, mask: np.ndarray, chosen_action: int
    ) -> list[tuple[str, float]]:
        values = probabilities.detach().cpu().numpy()
        grouped: dict[str, float] = defaultdict(float)
        chosen_group = self._action_group(chosen_action)
        for action in np.flatnonzero(mask > 0):
            if chosen_group == "Play" and self._action_group(int(action)) != "Play":
                continue
            grouped[self._option_code(int(action))] += float(values[action])
        total = sum(grouped.values()) or 1.0
        return sorted(
            ((code, value / total) for code, value in grouped.items()),
            key=lambda item: item[1],
            reverse=True,
        )


# Avoid noisy PyTorch internals being promoted through MahjongCopilot's logger.
logging.getLogger("torch").setLevel(logging.WARNING)
