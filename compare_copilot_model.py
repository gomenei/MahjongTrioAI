"""在内部三麻模拟器中公平比较 MahjongTrioAI 与本地 Mortal 模型。

比较器把模拟器状态转换为标准 MJAI 事件，让 MahjongCopilot 使用的
libriichi3p + Mortal 原生推理链参与实际对局。两种镜像阵容都会使用相同牌山
并轮换三次座位。hand 模式比较小局，south 模式比较完整东南风场。
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait as wait_connections
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

# Keep Chinese progress/error text readable in both modern PowerShell and the
# Codex terminal, whose output decoder expects UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from battle_models import (
    MatchTask,
    Policy,
    RunningGame,
    build_lineups,
    build_wall,
    choose_actions,
    combine_rotation_results,
    lineup_ci_half,
    load_or_create_state,
    save_json_atomic,
    select_device,
    write_reports,
)
from mahjong_env.feature import FeatureAgent, parse_tile_str
from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.match import SouthMatch
from mahjong_env.tile import Meld, MeldType, Suit, Tile, Wind
from model import load_checkpoint_model
from training.parallel_env import resolve_env_workers


PROJECT_ROOT = Path(__file__).resolve().parent
COPILOT_ROOT = PROJECT_ROOT / "integrations" / "MahjongCopilot"
DEFAULT_OUR_MODEL = PROJECT_ROOT / "ppo_runs" / "ppo_rich" / "best.pt"
DEFAULT_MORTAL_MODEL = COPILOT_ROOT / "models" / "mortal_3p.pth"
FORMAT_VERSION = 2
PASS_ACTION = FeatureAgent.OFFSET_ACT["Pass"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="让 MahjongTrioAI 与 MahjongCopilot 本地 Mortal 三麻模型实际对局"
    )
    parser.add_argument("--our-model", type=Path, default=DEFAULT_OUR_MODEL)
    parser.add_argument("--mortal-model", type=Path, default=DEFAULT_MORTAL_MODEL)
    parser.add_argument("--our-name", default="MahjongTrioAI")
    parser.add_argument("--mortal-name", default="MahjongCopilot-Mortal")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--action-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--game-mode", choices=("hand", "south"), default="south",
        help="hand 比较独立小局；south 比较完整东南风场",
    )
    parser.add_argument(
        "--rounds-per-wind", type=int, choices=(3, 4), default=3,
        help="标准三麻使用3；设为4可运行东一至南四",
    )
    parser.add_argument("--min-groups", type=int, default=1000)
    parser.add_argument("--max-groups", type=int, default=5000)
    parser.add_argument(
        "--target-ci",
        type=float,
        default=100.0,
        help="两模型平均点差的95%%置信区间半宽达到该值后停止",
    )
    parser.add_argument("--batch-groups", type=int, default=10)
    parser.add_argument(
        "--env-workers",
        type=int,
        default=0,
        help="Mortal+环境持久化子进程数；0 根据CPU自动选择，1 使用串行兼容模式",
    )
    parser.add_argument(
        "--inference-wait-ms",
        type=float,
        default=2.0,
        help="主进程等待更多我方决策后再做GPU批量推理的聚合窗口",
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000,
        help="单场最大决策步数；完整南风场建议3000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "battle_results" / "copilot_vs_trio",
    )
    parser.add_argument("--fresh", action="store_true", help="忽略旧断点并重新比较")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查 Mortal 权重与 libriichi3p 是否齐全，不开始对局",
    )
    return parser.parse_args()


def tile_to_mjai(tile: Tile) -> str:
    if tile.is_honor:
        return {1: "E", 2: "S", 3: "W", 4: "N", 5: "P", 6: "F", 7: "C"}[tile.value]
    suit = {Suit.Manzu: "m", Suit.Pinzu: "p", Suit.Souzu: "s"}[tile.suit]
    return f"{tile.value}{suit}{'r' if tile.is_red else ''}"


def exact_tile_equal(left: Tile, right: Tile) -> bool:
    return (
        left.suit == right.suit
        and left.value == right.value
        and left.is_red == right.is_red
    )


def same_tile_kind(left: Tile, right: Tile) -> bool:
    return left.suit == right.suit and left.value == right.value


def remove_called_tile(tiles: Sequence[Tile], called: Tile) -> List[Tile]:
    result = list(tiles)
    for index, tile in enumerate(result):
        if exact_tile_equal(tile, called):
            result.pop(index)
            return result
    for index, tile in enumerate(result):
        if same_tile_kind(tile, called):
            result.pop(index)
            return result
    raise ValueError(f"副露中找不到被鸣牌：{called}")


def dora_marker_events(game: ThreePlayerMahjong, old_count: int) -> List[dict]:
    events = []
    for marker_index in range(old_count, game.numdora):
        marker = game.dead_wall[8 + marker_index * 2]
        events.append({"type": "dora", "dora_marker": tile_to_mjai(marker)})
    return events


@dataclass
class GameSnapshot:
    state: int
    current_player: int
    cur_tile: Tile
    numdora: int
    packs: List[List[Meld]]

    @classmethod
    def capture(cls, game: ThreePlayerMahjong) -> "GameSnapshot":
        return cls(
            state=game.state,
            current_player=game.current_player,
            cur_tile=copy.copy(game.curTile),
            numdora=game.numdora,
            packs=[list(melds) for melds in game.packs],
        )


def response_parts(
    game: ThreePlayerMahjong, observations: Dict[str, dict], actions: Dict[str, int]
) -> Dict[int, List[str]]:
    result = {}
    for agent_name in observations:
        seat = int(agent_name.rsplit("_", 1)[1])
        result[seat] = game.agents[seat].action2response(actions[agent_name]).split()
    return result


def meld_event(event_type: str, actor: int, meld: Meld, called: Tile) -> dict:
    consumed = remove_called_tile(meld.tiles, called)
    return {
        "type": event_type,
        "actor": actor,
        "target": int(meld.from_player),
        "pai": tile_to_mjai(called),
        "consumed": [tile_to_mjai(tile) for tile in consumed],
    }


def transition_events(
    before: GameSnapshot,
    game: ThreePlayerMahjong,
    responses: Dict[int, List[str]],
    pending_reach: Optional[int],
) -> tuple[List[dict], Optional[int]]:
    """Convert one internal environment transition into public MJAI events."""
    if game.done:
        return [], None

    events: List[dict] = []
    actor = before.current_player

    if before.state in (0, 1):
        parts = responses[actor]
        action = parts[0]
        if action in ("Play", "Riichi"):
            tile = parse_tile_str(parts[1])
            if action == "Riichi":
                events.append({"type": "reach", "actor": actor})
                pending_reach = actor
            events.append(
                {
                    "type": "dahai",
                    "actor": actor,
                    "pai": tile_to_mjai(tile),
                    "tsumogiri": before.state == 1 and exact_tile_equal(tile, before.cur_tile),
                }
            )
        elif action == "AnKang":
            meld = game.packs[actor][-1]
            events.extend(dora_marker_events(game, before.numdora))
            events.append(
                {
                    "type": "ankan",
                    "actor": actor,
                    "consumed": [tile_to_mjai(tile) for tile in meld.tiles],
                }
            )
        elif action == "BuKang":
            added = parse_tile_str(parts[1])
            old_pon = next(
                meld
                for meld in before.packs[actor]
                if meld.type == MeldType.Pon and same_tile_kind(meld.taken_tile, added)
            )
            events.append(
                {
                    "type": "kakan",
                    "actor": actor,
                    "pai": tile_to_mjai(added),
                    "consumed": [tile_to_mjai(tile) for tile in old_pon.tiles],
                }
            )
        elif action == "Pei":
            events.append({"type": "nukidora", "actor": actor, "pai": "N"})
        else:
            raise ValueError(f"状态 {before.state} 出现无法转换的动作：{parts}")

    elif before.state == 2:
        if pending_reach is not None:
            events.append({"type": "reach_accepted", "actor": pending_reach})
            pending_reach = None

        called_actor = next(
            (
                seat
                for seat in range(3)
                if len(game.packs[seat]) > len(before.packs[seat])
            ),
            None,
        )
        if called_actor is not None:
            meld = game.packs[called_actor][-1]
            if meld.type == MeldType.Pon:
                events.append(meld_event("pon", called_actor, meld, before.cur_tile))
            elif meld.type == MeldType.OpenKan:
                events.append(meld_event("daiminkan", called_actor, meld, before.cur_tile))
                events.extend(dora_marker_events(game, before.numdora))
                events.append(
                    {
                        "type": "tsumo",
                        "actor": game.current_player,
                        "pai": tile_to_mjai(game.curTile),
                    }
                )
            else:
                raise ValueError(f"舍牌响应产生了意外副露：{meld.type}")
        else:
            events.append(
                {
                    "type": "tsumo",
                    "actor": game.current_player,
                    "pai": tile_to_mjai(game.curTile),
                }
            )

    elif before.state == 3:
        events.extend(dora_marker_events(game, before.numdora))
        events.append(
            {
                "type": "tsumo",
                "actor": game.current_player,
                "pai": tile_to_mjai(game.curTile),
            }
        )
    else:
        raise ValueError(f"未知环境状态：{before.state}")

    return events, pending_reach


class MortalSeat:
    def __init__(self, native_module, engine, seat: int):
        self.seat = seat
        self.bot = native_module.mjai.Bot(engine, seat)
        self.pending_reaction: Optional[dict] = None

    def feed(self, events: Sequence[dict], can_act: bool) -> None:
        self.pending_reaction = None
        for index, source in enumerate(events):
            event = dict(source)
            if event["type"] == "tsumo" and int(event["actor"]) != self.seat:
                event["pai"] = "?"
            event["can_act"] = bool(can_act and index == len(events) - 1)
            raw = self.bot.react(json.dumps(event, ensure_ascii=False))
            if raw is not None:
                self.pending_reaction = json.loads(raw)


class MortalRuntime:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"找不到 MahjongCopilot Mortal 三麻权重：{model_path}\n"
                "请从 MahjongCopilot/Akagi 的完整发行包取得 mortal_3p.pth，"
                "放入 integrations/MahjongCopilot/models/。"
            )
        if str(COPILOT_ROOT) not in sys.path:
            sys.path.insert(0, str(COPILOT_ROOT))
        try:
            import libriichi3p
        except ImportError as exc:
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            expected = (
                COPILOT_ROOT
                / "libriichi3p"
                / f"libriichi3p-{version}-x86_64-pc-windows-msvc.pyd"
            )
            raise RuntimeError(
                "缺少 MahjongCopilot 的三麻原生推理库。\n"
                f"当前 Python 版本期待文件：{expected}\n"
                "请从 MahjongCopilot 的完整发行包复制对应 libriichi3p 文件。"
            ) from exc

        native = getattr(libriichi3p, "libriichi3p", libriichi3p)
        from bot.local.engine3p import get_engine

        self.native = native
        self.engine = get_engine(str(model_path.resolve()))

    def create_seats(self, seats: Iterable[int]) -> Dict[int, MortalSeat]:
        return {seat: MortalSeat(self.native, self.engine, seat) for seat in seats}


def initial_events(game: ThreePlayerMahjong, seat: int) -> List[dict]:
    tehais = [["?"] * 13 for _ in range(4)]
    tehais[seat] = [tile_to_mjai(tile) for tile in sorted(game.hands[seat])]
    bakaze = {Wind.East: "E", Wind.South: "S", Wind.West: "W", Wind.North: "N"}[
        game.prevailing_wind
    ]
    marker = tile_to_mjai(game.dead_wall[8])
    return [
        {"type": "start_game", "id": seat},
        {
            "type": "start_kyoku",
            "bakaze": bakaze,
            "dora_marker": marker,
            "honba": game.honba,
            "kyoku": game.round_number + 1,
            "kyotaku": game.riichi_sticks,
            "oya": game.current_player,
            "scores": [*game.scores, 0],
            "tehais": tehais,
        },
        {
            "type": "tsumo",
            "actor": game.current_player,
            "pai": tile_to_mjai(game.curTile),
        },
    ]


def reaction_tile(reaction: dict, action_name: str) -> Optional[Tile]:
    if action_name == "Riichi":
        reaction = reaction.get("reach_dahai") or {}
    pai = reaction.get("pai")
    if not pai or pai == "?":
        return None
    honors = {"E": 1, "S": 2, "W": 3, "N": 4, "P": 5, "F": 6, "C": 7}
    if pai in honors:
        return Tile(Suit.Honors, honors[pai])
    suit = {"m": Suit.Manzu, "p": Suit.Pinzu, "s": Suit.Souzu}[pai[1]]
    return Tile(suit, int(pai[0]), is_red=pai.endswith("r"))


def reaction_to_local_action(
    reaction: Optional[dict], observation: dict, agent: FeatureAgent
) -> tuple[int, Optional[str]]:
    legal = [int(value) for value in np.flatnonzero(observation["action_mask"] > 0)]
    if len(legal) == 1:
        return legal[0], None
    if not reaction:
        fallback = PASS_ACTION if PASS_ACTION in legal else legal[0]
        return fallback, "Mortal 在多选决策点没有返回动作"

    event_type = reaction.get("type")
    action_name = {
        "dahai": "Play",
        "reach": "Riichi",
        "pon": "Peng",
        "daiminkan": "Kang",
        "ankan": "AnKang",
        "kakan": "BuKang",
        "nukidora": "Pei",
        "hora": "Hu",
        "none": "Pass",
    }.get(event_type)
    if action_name is None:
        fallback = PASS_ACTION if PASS_ACTION in legal else legal[0]
        return fallback, f"内部环境不支持 Mortal 动作 {event_type!r}"

    desired_tile = reaction_tile(reaction, action_name)
    if action_name == "Peng":
        consumed = reaction.get("consumed", [])
        if consumed:
            # Local Peng encodes the tiles taken from our hand. A red discard
            # called with two normal fives still requires the normal Peng action.
            hand_tile = next((tile for tile in consumed if tile.endswith("r")), consumed[0])
            desired_tile = reaction_tile({"pai": hand_tile}, "Peng")

    candidates = []
    for action in legal:
        parts = agent.action2response(action).split()
        if parts[0] != action_name:
            continue
        if desired_tile is not None and len(parts) > 1:
            if not exact_tile_equal(parse_tile_str(parts[1]), desired_tile):
                continue
        candidates.append(action)
    if candidates:
        return candidates[0], None

    fallback = PASS_ACTION if PASS_ACTION in legal else legal[0]
    return fallback, f"Mortal 返回的 {reaction} 不在内部 action_mask 中"


def empty_bridge_stats() -> dict:
    return {"mortal_decisions": 0, "fallbacks": 0, "fallback_reasons": {}}


def merge_bridge_stats(target: dict, source: dict) -> None:
    target["mortal_decisions"] = target.get("mortal_decisions", 0) + source.get(
        "mortal_decisions", 0
    )
    target["fallbacks"] = target.get("fallbacks", 0) + source.get("fallbacks", 0)
    target_reasons = target.setdefault("fallback_reasons", {})
    for reason, count in source.get("fallback_reasons", {}).items():
        target_reasons[reason] = target_reasons.get(reason, 0) + count


class CopilotGameRunner:
    """在 worker 内推进一场比赛，只把我方决策送回主进程批量推理。"""

    def __init__(
        self,
        task: MatchTask,
        mortal_runtime: MortalRuntime,
        args: argparse.Namespace,
    ):
        self.task = task
        self.mortal_runtime = mortal_runtime
        self.args = args
        if args.game_mode == "south":
            game = SouthMatch(rounds_per_wind=args.rounds_per_wind)
            observations = game.reset(seed=task.seed, current_player=0)
        else:
            game = ThreePlayerMahjong()
            observations = game.reset(walls=build_wall(task.seed), current_player=0)
        self.item = RunningGame(task=task, game=game, observations=observations)
        self.mortal_seats = [
            seat
            for seat, model_index in enumerate(task.seat_models)
            if model_index == 1
        ]
        self.controllers = self._start_mortal_hand(self._hand_game())
        self.pending_reach: Optional[int] = None
        self.pending_local_actions: Optional[Dict[str, int]] = None
        self.pending_our_agents = set()
        self.bridge_stats = empty_bridge_stats()

    def _hand_game(self) -> ThreePlayerMahjong:
        game = self.item.game
        return game.game if isinstance(game, SouthMatch) else game

    def _start_mortal_hand(
        self, hand_game: ThreePlayerMahjong
    ) -> Dict[int, MortalSeat]:
        # 每个 worker 独立持有 Mortal engine；每个小局重新建立 MJAI Bot。
        controllers = self.mortal_runtime.create_seats(self.mortal_seats)
        for seat, controller in controllers.items():
            controller.feed(
                initial_events(hand_game, seat),
                can_act=seat == hand_game.current_player,
            )
        return controllers

    def _step(self, actions: Dict[str, int]) -> None:
        game = self.item.game
        hand_game = self._hand_game()
        before = GameSnapshot.capture(hand_game)
        responses = response_parts(hand_game, self.item.observations, actions)
        observations, _, _ = game.step(actions)
        self.item.steps += 1
        self.item.observations = observations
        if game.done:
            return
        if hand_game.done:
            self.controllers = self._start_mortal_hand(self._hand_game())
            self.pending_reach = None
            return
        events, self.pending_reach = transition_events(
            before, hand_game, responses, self.pending_reach
        )
        active_seats = {int(name.rsplit("_", 1)[1]) for name in observations}
        for seat, controller in self.controllers.items():
            controller.feed(events, can_act=seat in active_seats)

    def advance(self, our_actions: Optional[Dict[str, int]] = None) -> dict:
        """推进到下一次我方决策或比赛结束。"""
        try:
            if self.pending_local_actions is not None:
                supplied = set(our_actions or {})
                if supplied != self.pending_our_agents:
                    raise ValueError(
                        "主进程返回的我方动作不完整："
                        f"expected={self.pending_our_agents}, actual={supplied}"
                    )
                actions = dict(self.pending_local_actions)
                actions.update({name: int(value) for name, value in our_actions.items()})
                self.pending_local_actions = None
                self.pending_our_agents = set()
                self._step(actions)
            elif our_actions:
                raise ValueError("当前没有等待我方动作")

            while not self.item.game.done and self.item.steps < self.args.max_steps:
                hand_game = self._hand_game()
                local_actions: Dict[str, int] = {}
                requests: Dict[str, dict] = {}
                for agent_name, observation in self.item.observations.items():
                    seat = int(agent_name.rsplit("_", 1)[1])
                    if self.task.seat_models[seat] == 0:
                        requests[agent_name] = observation
                        continue
                    self.bridge_stats["mortal_decisions"] += 1
                    action, issue = reaction_to_local_action(
                        self.controllers[seat].pending_reaction,
                        observation,
                        hand_game.agents[seat],
                    )
                    local_actions[agent_name] = action
                    if issue:
                        self.bridge_stats["fallbacks"] += 1
                        reasons = self.bridge_stats["fallback_reasons"]
                        reasons[issue] = reasons.get(issue, 0) + 1

                if requests:
                    self.pending_local_actions = local_actions
                    self.pending_our_agents = set(requests)
                    return {"kind": "decision", "observations": requests}
                if not local_actions:
                    raise RuntimeError("环境没有可执行动作，也没有结束")
                self._step(local_actions)
        except Exception as exc:
            self.item.error = f"{type(exc).__name__}: {exc}"

        if not self.item.game.done and self.item.error is None:
            self.item.error = f"超过最大步数 {self.args.max_steps}"
        return {
            "kind": "done",
            "item": self.item,
            "bridge_stats": self.bridge_stats,
        }


def run_game(
    task: MatchTask,
    our_policy: Policy,
    mortal_runtime: MortalRuntime,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    bridge_stats: dict,
) -> RunningGame:
    """单进程兼容路径；也复用向量 worker 的状态机以保证规则一致。"""
    runner = CopilotGameRunner(task, mortal_runtime, args)
    event = runner.advance()
    while event["kind"] == "decision":
        names = list(event["observations"])
        actions = choose_actions(
            our_policy,
            [event["observations"][name] for name in names],
            device,
            args.action_mode,
            args.temperature,
            rng,
        )
        event = runner.advance(dict(zip(names, actions)))
    merge_bridge_stats(bridge_stats, event["bridge_stats"])
    return event["item"]


def _copilot_worker_main(
    connection: Connection,
    mortal_model: str,
    args: argparse.Namespace,
) -> None:
    """一个进程持久化一个 Mortal engine，并串行承接多场比赛。"""
    try:
        # 多个进程已经提供外层并行，限制每个引擎内部线程可避免严重超额订阅。
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        mortal_runtime = MortalRuntime(Path(mortal_model))
        connection.send(("ready", None))
        runner: Optional[CopilotGameRunner] = None
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(("closed", None))
                return
            if command == "run":
                raw_task = payload
                task = MatchTask(
                    seed=int(raw_task["seed"]),
                    rotation=int(raw_task["rotation"]),
                    seat_models=tuple(raw_task["seat_models"]),
                )
                runner = CopilotGameRunner(task, mortal_runtime, args)
                event = runner.advance()
            elif command == "actions":
                if runner is None:
                    raise RuntimeError("worker 尚未收到比赛任务")
                event = runner.advance(payload)
            else:
                raise ValueError(f"未知 worker 命令：{command}")
            connection.send((event["kind"], event))
            if event["kind"] == "done":
                runner = None
    except EOFError:
        return
    except BaseException:
        try:
            connection.send(("fatal", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass


class CopilotGamePool:
    """Mortal/环境多进程运行，我方网络在主进程 GPU 集中批量推理。"""

    def __init__(self, workers: int, mortal_model: Path, args: argparse.Namespace):
        self.context = mp.get_context("spawn")
        self.parents: List[Connection] = []
        self.processes = []
        for _ in range(workers):
            parent, child = self.context.Pipe(duplex=True)
            process = self.context.Process(
                target=_copilot_worker_main,
                args=(child, str(mortal_model.resolve()), args),
                daemon=True,
            )
            process.start()
            child.close()
            self.parents.append(parent)
            self.processes.append(process)
        self.by_fileno = {parent.fileno(): index for index, parent in enumerate(self.parents)}
        for worker_index, parent in enumerate(self.parents):
            kind, payload = parent.recv()
            if kind == "fatal":
                self.close()
                raise RuntimeError(f"Copilot worker {worker_index} 初始化失败：\n{payload}")
            if kind != "ready":
                self.close()
                raise RuntimeError(f"Copilot worker {worker_index} 未正常就绪：{kind}")

    @staticmethod
    def _task_payload(task: MatchTask) -> dict:
        return {
            "seed": task.seed,
            "rotation": task.rotation,
            "seat_models": list(task.seat_models),
        }

    def run(
        self,
        tasks: Sequence[MatchTask],
        our_policy: Policy,
        args: argparse.Namespace,
        device: torch.device,
        rng: np.random.Generator,
        bridge_stats: dict,
    ) -> List[RunningGame]:
        results: List[Optional[RunningGame]] = [None] * len(tasks)
        worker_tasks: Dict[int, int] = {}
        next_task = 0

        def assign(worker_index: int) -> None:
            nonlocal next_task
            if next_task >= len(tasks):
                return
            worker_tasks[worker_index] = next_task
            self.parents[worker_index].send(
                ("run", self._task_payload(tasks[next_task]))
            )
            next_task += 1

        for worker_index in range(min(len(self.parents), len(tasks))):
            assign(worker_index)

        while worker_tasks:
            active_connections = [self.parents[index] for index in worker_tasks]
            ready = list(wait_connections(active_connections))
            # 给其他 worker 一个很短的聚合窗口，形成更大的 GPU batch。
            deadline = time.perf_counter() + args.inference_wait_ms / 1000.0
            while len(ready) < len(active_connections):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                pending = [connection for connection in active_connections if connection not in ready]
                more = wait_connections(pending, timeout=remaining)
                if not more:
                    break
                ready.extend(more)

            decisions = []
            completed_workers = []
            for connection in ready:
                worker_index = self.by_fileno[connection.fileno()]
                kind, payload = connection.recv()
                if kind == "fatal":
                    raise RuntimeError(f"Copilot worker {worker_index} 失败：\n{payload}")
                if kind == "done":
                    task_index = worker_tasks.pop(worker_index)
                    results[task_index] = payload["item"]
                    merge_bridge_stats(bridge_stats, payload["bridge_stats"])
                    completed_workers.append(worker_index)
                elif kind == "decision":
                    decisions.append((worker_index, payload["observations"]))
                else:
                    raise RuntimeError(
                        f"Copilot worker {worker_index} 返回未知消息：{kind}"
                    )

            for worker_index in completed_workers:
                assign(worker_index)

            if decisions:
                requests = []
                owners = []
                for worker_index, observations in decisions:
                    for agent_name, observation in observations.items():
                        requests.append(observation)
                        owners.append((worker_index, agent_name))
                chosen = choose_actions(
                    our_policy,
                    requests,
                    device,
                    args.action_mode,
                    args.temperature,
                    rng,
                )
                by_worker: Dict[int, Dict[str, int]] = {}
                for (worker_index, agent_name), action in zip(owners, chosen):
                    by_worker.setdefault(worker_index, {})[agent_name] = int(action)
                for worker_index, actions in by_worker.items():
                    self.parents[worker_index].send(("actions", actions))

        if any(result is None for result in results):
            raise RuntimeError("Copilot 进程池遗漏了比赛结果")
        return list(results)

    def close(self) -> None:
        for index, process in enumerate(self.processes):
            if process.is_alive():
                try:
                    self.parents[index].send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self.processes:
            process.join(timeout=3)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for parent in self.parents:
            parent.close()
        self.parents.clear()
        self.processes.clear()


def update_report_metadata(args: argparse.Namespace, state: dict) -> None:
    report_path = args.output_dir / "battle_report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["comparison"] = {
        "format_version": FORMAT_VERSION,
        "our_model": str(args.our_model.resolve()),
        "mortal_model": str(args.mortal_model.resolve()),
        "method": "相同牌山、三次座位轮换、A+A+B 与 A+B+B 镜像阵容",
        "game_mode": args.game_mode,
        "rounds_per_wind": args.rounds_per_wind,
        "mortal_bridge": state.get("bridge_stats", {}),
    }
    limitations = report.setdefault("limitations", [])
    limitations.append("Mortal 的九种九牌等内部环境未实现动作会按合法后备动作执行并计数")
    save_json_atomic(report_path, report)


def validate_args(args: argparse.Namespace) -> None:
    if args.min_groups < 2:
        raise ValueError("--min-groups 必须至少为2")
    if args.max_groups < args.min_groups:
        raise ValueError("--max-groups 不能小于 --min-groups")
    if args.batch_groups < 1 or args.max_steps < 1:
        raise ValueError("--batch-groups 和 --max-steps 必须为正数")
    if args.env_workers < 0:
        raise ValueError("--env-workers 不能为负数；0 表示自动")
    if args.inference_wait_ms < 0:
        raise ValueError("--inference-wait-ms 不能为负数")
    if args.target_ci <= 0 or args.temperature <= 0:
        raise ValueError("--target-ci 和 --temperature 必须为正数")
    if not args.our_model.is_file():
        raise FileNotFoundError(f"找不到 MahjongTrioAI 权重：{args.our_model}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        mortal_runtime = MortalRuntime(args.mortal_model)
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"\n依赖检查失败：\n{exc}", file=sys.stderr)
        print(
            "\nMahjongCopilot 发行页：https://github.com/latorc/MahjongCopilot/releases\n"
            "Akagi 项目页：https://github.com/shinkuan/Akagi",
            file=sys.stderr,
        )
        return 2
    print(f"Mortal 运行时检查通过：{args.mortal_model}")
    if args.check_only:
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    print(
        f"比赛模式：{'完整南风场' if args.game_mode == 'south' else '独立小局'}；"
        f"每风 {args.rounds_per_wind} 局；设备：{device}"
    )
    model, metadata = load_checkpoint_model(args.our_model, map_location=device)
    model = model.to(device).eval()
    our_policy = Policy(
        name=args.our_name,
        checkpoint=args.our_model.resolve(),
        model=model,
        observation_key=str(metadata.get("observation_key", "observation")),
    )
    report_policies = [
        our_policy,
        Policy(
            name=args.mortal_name,
            checkpoint=args.mortal_model.resolve(),
            model=None,
        ),
    ]
    lineups = build_lineups(2)
    state = load_or_create_state(args, report_policies, lineups)
    state.setdefault(
        "bridge_stats",
        empty_bridge_stats(),
    )
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    capacity = args.batch_groups * 3
    worker_count = resolve_env_workers(args.env_workers, capacity)
    game_pool: Optional[CopilotGamePool] = None
    if worker_count > 1:
        # 父进程不再参与 Mortal 推理，释放依赖检查时建立的引擎。
        del mortal_runtime
        game_pool = CopilotGamePool(worker_count, args.mortal_model, args)
        print(
            f"Copilot 对局池：{worker_count} 个持久化进程；"
            "我方网络由主进程集中 GPU 批量推理"
        )
    else:
        print("Copilot 对局池：单进程兼容模式")

    try:
        for lineup_number, lineup in enumerate(lineups, 1):
            key = ",".join(map(str, lineup))
            lineup_state = state["lineups"][key]
            groups = lineup_state["groups"]
            lineup_state.setdefault("next_group_index", len(groups))
            if lineup_state["complete"]:
                continue
            print(f"\n镜像阵容 {lineup_number}/2：{lineup}")
            while lineup_state["next_group_index"] < args.max_groups:
                count = min(
                    args.batch_groups,
                    args.max_groups - lineup_state["next_group_index"],
                )
                start = lineup_state["next_group_index"]
                tasks = []
                for offset in range(count):
                    group_index = start + offset
                    seed = args.seed + lineup_number * 10_000_000 + group_index
                    for rotation in range(3):
                        seat_models = tuple(
                            lineup[(seat + rotation) % 3] for seat in range(3)
                        )
                        tasks.append(MatchTask(seed, rotation, seat_models))
                batch_bridge_stats = empty_bridge_stats()
                if game_pool is None:
                    rotations = [
                        run_game(
                            task,
                            our_policy,
                            mortal_runtime,
                            args,
                            device,
                            rng,
                            batch_bridge_stats,
                        )
                        for task in tasks
                    ]
                else:
                    rotations = game_pool.run(
                        tasks,
                        our_policy,
                        args,
                        device,
                        rng,
                        batch_bridge_stats,
                    )
                merge_bridge_stats(state["bridge_stats"], batch_bridge_stats)
                new_groups, errors = combine_rotation_results(rotations, lineup)
                groups.extend(new_groups)
                lineup_state["errors"].extend(errors)
                lineup_state["next_group_index"] += count

                half = lineup_ci_half(groups, lineup)
                fallbacks = state["bridge_stats"]["fallbacks"]
                print(
                    f"  有效牌山组 {len(groups):,} / 实战局 {len(groups) * 3:,} / "
                    f"点差95% CI ±{half:.1f}点 / 桥接后备动作 {fallbacks:,}"
                )
                save_json_atomic(args.output_dir / "progress.json", state)
                write_reports(args, state, report_policies)
                update_report_metadata(args, state)
                if len(groups) >= args.min_groups and half <= args.target_ci:
                    lineup_state["complete"] = True
                    break
                if lineup_state["errors"] and not groups:
                    raise RuntimeError("整批对局均失败，请查看 progress.json 中的 errors")

            if lineup_state["next_group_index"] >= args.max_groups:
                lineup_state["complete"] = True
            save_json_atomic(args.output_dir / "progress.json", state)

    except KeyboardInterrupt:
        save_json_atomic(args.output_dir / "progress.json", state)
        write_reports(args, state, report_policies)
        update_report_metadata(args, state)
        print("\n已停止并保存断点；重新运行时去掉 --fresh 即可继续。")
        return 130
    finally:
        if game_pool is not None:
            game_pool.close()

    write_reports(args, state, report_policies)
    update_report_metadata(args, state)
    print(f"总耗时：{time.perf_counter() - started:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
