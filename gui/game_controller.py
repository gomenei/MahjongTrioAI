from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from mahjong_env.game import ThreePlayerMahjong
from gui.replay import ReplaySession
from training.models import load_checkpoint_model


class GameController:
    """在麻将规则、玩家输入和两个 AI 之间推进牌局。"""

    HUMAN_PLAYER = 0

    ACTION_RANGES = {
        "Play": range(0, 29),
        "Peng": range(29, 58),
        "Kang": range(58, 87),
        "AnKang": range(87, 116),
        "BuKang": range(116, 145),
        "Riichi": range(145, 174),
        "Pei": range(174, 175),
        "Hu": range(175, 176),
        "Pass": range(176, 177),
    }

    def __init__(self, model_path: Path, ai_delay_ms: int = 420, replay_dir: Optional[Path] = None):
        self.model_path = Path(model_path)
        self.replay_dir = Path(replay_dir) if replay_dir is not None else Path(__file__).resolve().parents[1] / "replays"
        self.ai_delay_ms = ai_delay_ms
        self.game = ThreePlayerMahjong()
        self.model: Optional[nn.Module] = None
        self.model_observation_key = "observation"
        self.model_error = ""
        self.obs: Dict = {}
        self.last_action = ""
        self.next_ai_time = 0
        self.mode = "live"
        self.replay: Optional[ReplaySession] = None
        self.replay_error = ""
        self._load_model()
        self.reset()

    def _load_model(self):
        try:
            model, metadata = load_checkpoint_model(self.model_path, map_location="cpu")
            model.eval()
            self.model = model
            self.model_observation_key = str(
                metadata.get("observation_key", "observation")
            )
        except Exception as exc:
            # 没有模型时仍可运行：AI 会从合法动作中随机选择。
            self.model = None
            self.model_observation_key = "observation"
            self.model_error = str(exc)

    def reset(self):
        self.mode = "live"
        self.game = ThreePlayerMahjong()
        self.obs = self.game.reset(current_player=0)
        self.last_action = "牌局开始：你是玩家 0"
        self.next_ai_time = 0

    @property
    def done(self) -> bool:
        return self.mode == "live" and self.game.done

    @property
    def _human_has_observation(self) -> bool:
        return self.mode == "live" and "player_0" in self.obs and not self.done

    @property
    def human_can_act(self) -> bool:
        if not self._human_has_observation:
            return False
        valid = np.flatnonzero(self.obs["player_0"]["action_mask"])
        return not (len(valid) == 1 and int(valid[0]) == 176)

    def public_state(self):
        if self.mode == "replay" and self.replay:
            return self.replay.state
        return self.game.get_public_state()

    def _human_mask(self) -> np.ndarray:
        if not self._human_has_observation:
            return np.zeros(177, dtype=np.float32)
        return self.obs["player_0"]["action_mask"]

    def valid_human_actions(self) -> List[int]:
        return np.flatnonzero(self._human_mask()).astype(int).tolist()

    def actions_for_kind(self, kind: str) -> List[int]:
        action_range = self.ACTION_RANGES[kind]
        valid = set(self.valid_human_actions())
        return [action for action in action_range if action in valid]

    def tile_action(self, kind: str, tile) -> Optional[int]:
        if kind not in {"Play", "Riichi", "AnKang", "BuKang", "Peng", "Kang"}:
            return None
        action = self.game.agents[self.HUMAN_PLAYER].response2action(
            f"{kind} {tile}"
        )
        return action if action in self.valid_human_actions() else None

    def submit_human_action(self, action: int) -> bool:
        if action not in self.valid_human_actions():
            return False

        actions = {"player_0": action}
        for agent_name, observation in self.obs.items():
            if agent_name != "player_0":
                actions[agent_name] = self._choose_ai_action(observation)

        response = self.game.agents[0].action2response(action)
        self.last_action = f"你选择：{self._friendly_response(response)}"
        self.obs, _, _ = self.game.step(actions)
        self.next_ai_time = 0
        return True

    def update(self, now_ms: int):
        if self.mode == "replay":
            return
        if self._human_has_observation:
            valid = self.valid_human_actions()
            if valid == [176]:
                self.submit_human_action(176)
                return
        if self.done or self.human_can_act or not self.obs:
            return
        if self.next_ai_time == 0:
            self.next_ai_time = now_ms + self.ai_delay_ms
            return
        if now_ms < self.next_ai_time:
            return

        actions = {}
        descriptions = []
        for agent_name, observation in self.obs.items():
            action = self._choose_ai_action(observation)
            actions[agent_name] = action
            player = int(agent_name.rsplit("_", 1)[1])
            response = self.game.agents[player].action2response(action)
            descriptions.append(
                f"AI {player}：{self._friendly_response(response)}"
            )

        self.last_action = "　".join(descriptions)
        self.obs, _, _ = self.game.step(actions)
        self.next_ai_time = now_ms + self.ai_delay_ms

    def load_replay(self, path: Path) -> bool:
        try:
            self.replay = ReplaySession(Path(path))
            self.mode = "replay"
            self.replay_error = ""
            self.last_action = self.replay.state["event"]
            return True
        except Exception as exc:
            self.replay_error = str(exc)
            return False

    def set_live_mode(self):
        self.mode = "live"

    def set_replay_mode(self):
        if self.replay:
            self.mode = "replay"

    def replay_change_step(self, delta: int):
        if self.replay:
            self.replay.change_step(delta)
            self.last_action = self.replay.state["event"]

    def replay_seek(self, step: int):
        if self.replay:
            self.replay.seek(step)
            self.last_action = self.replay.state["event"]

    def replay_change_round(self, delta: int):
        if self.replay:
            self.replay.change_round(delta)
            self.last_action = self.replay.state["event"]

    def replay_set_view(self, player: int):
        if self.replay:
            self.replay.set_view(player)

    def _choose_ai_action(self, observation: Dict) -> int:
        mask = np.asarray(observation["action_mask"], dtype=np.float32)
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            return 176

        if self.model is None:
            return int(np.random.choice(valid_actions))

        model_input = {
            "observation": torch.as_tensor(
                observation[self.model_observation_key], dtype=torch.float32
            ).unsqueeze(0),
            "action_mask": torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0),
        }
        with torch.no_grad():
            logits = self.model(model_input)
            return int(torch.argmax(logits, dim=1).item())

    @staticmethod
    def _friendly_response(response: str) -> str:
        replacements = {
            "Play": "打",
            "Peng": "碰",
            "Kang": "明杠",
            "AnKang": "暗杠",
            "BuKang": "加杠",
            "Riichi": "立直",
            "Pei": "拔北",
            "Hu": "和牌",
            "Pass": "跳过",
        }
        parts = response.split(maxsplit=1)
        name = replacements.get(parts[0], parts[0])
        return name if len(parts) == 1 else f"{name} {parts[1]}"
