"""持久化多进程麻将环境池，主进程集中执行模型推理。

worker 只负责保存 ThreePlayerMahjong 状态和调用 game.step。观测、动作和
action mask 通过共享内存传递，Pipe 中只发送槽位编号与完成状态，避免每一步
pickle 大型 numpy 数组。模型始终只存在于主进程，CUDA context 不会复制到 worker。
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import random
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.sharedctypes import RawArray
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from training.rules_candidate_game import CandidateThreePlayerMahjong as ThreePlayerMahjong
from training.rules_candidate_match import CandidateSouthMatch as SouthMatch
from mahjong_env.tile import Suit, Tile


MAX_PLAYERS = 3
RICH_CHANNELS = 101
MATCH_CHANNELS = 121
OBS_TILES = 30
ACTION_SIZE = 177


@dataclass(frozen=True)
class GameSpec:
    slot: int
    seed: int
    current_player: int = 0
    game_mode: str = "hand"
    rounds_per_wind: int = 3


def recommended_env_workers(
    logical_processors: Optional[int] = None,
    capacity: Optional[int] = None,
) -> int:
    """为 Python 环境步进保留主进程和系统所需的逻辑线程。

    i7-12700H 报告 20 个逻辑处理器，自动选择 12 个 worker。麻将规则代码以
    Python 为主，按物理核心附近的数量扩展通常比占满全部超线程更稳定。
    """
    logical = int(logical_processors or os.cpu_count() or 1)
    if logical >= 20:
        workers = 12
    elif logical >= 12:
        workers = 8
    elif logical >= 8:
        workers = 6
    else:
        workers = max(1, logical - 2)
    if capacity is not None:
        workers = min(workers, max(1, int(capacity)))
    return max(1, workers)


def resolve_env_workers(requested: int, capacity: int) -> int:
    if requested < 0:
        raise ValueError("env_workers 不能为负数；0 表示自动")
    if requested == 0:
        return recommended_env_workers(capacity=capacity)
    return min(int(requested), max(1, int(capacity)))


def _build_wall(seed: int) -> List[Tile]:
    """与 battle_models.build_wall 相同，但保持 worker 模块无循环依赖。"""
    wall: List[Tile] = []
    for suit in (Suit.Manzu, Suit.Pinzu, Suit.Souzu):
        for value in range(1, 10):
            if suit == Suit.Manzu and 1 < value < 9:
                continue
            if suit in (Suit.Pinzu, Suit.Souzu) and value == 5:
                wall.append(Tile(suit, value, is_red=True))
                wall.extend(Tile(suit, value) for _ in range(3))
            else:
                wall.extend(Tile(suit, value) for _ in range(4))
    for value in range(1, 8):
        wall.extend(Tile(Suit.Honors, value) for _ in range(4))
    random.Random(seed).shuffle(wall)
    return wall


def _shared_views(shared, capacity: int):
    observations = np.frombuffer(
        shared["observations"], dtype=np.float32
    ).reshape(capacity, MAX_PLAYERS, MATCH_CHANNELS, OBS_TILES)
    masks = np.frombuffer(
        shared["masks"], dtype=np.float32
    ).reshape(capacity, MAX_PLAYERS, ACTION_SIZE)
    active_agents = np.frombuffer(
        shared["active_agents"], dtype=np.uint8
    ).reshape(capacity, MAX_PLAYERS)
    actions = np.frombuffer(
        shared["actions"], dtype=np.int16
    ).reshape(capacity, MAX_PLAYERS)
    return observations, masks, active_agents, actions


def _write_observations(slot: int, game_observations: Dict, arrays) -> None:
    observations, masks, active_agents, _ = arrays
    active_agents[slot].fill(0)
    for agent_name, observation in game_observations.items():
        seat = int(agent_name.rsplit("_", 1)[1])
        match_observation = observation.get("match_observation")
        rich = observation.get("rich_observation")
        if match_observation is not None:
            observations[slot, seat] = np.asarray(
                match_observation, dtype=np.float32
            )
        elif rich is None:
            observations[slot, seat].fill(0)
            legacy = np.asarray(observation["observation"], dtype=np.float32)
            observations[slot, seat, : legacy.shape[0]] = legacy
        else:
            observations[slot, seat].fill(0)
            rich_array = np.asarray(rich, dtype=np.float32)
            observations[slot, seat, : rich_array.shape[0]] = rich_array
        masks[slot, seat] = np.asarray(
            observation["action_mask"], dtype=np.float32
        )
        active_agents[slot, seat] = 1


def forced_actions(observations: Dict) -> Optional[Dict[str, int]]:
    """Return actions only when every active player has exactly one legal move."""
    if not observations:
        return None
    actions = {}
    for name, observation in observations.items():
        legal = np.flatnonzero(np.asarray(observation["action_mask"]) > 0)
        if len(legal) != 1:
            return None
        actions[name] = int(legal[0])
    return actions


def step_to_decision(game, actions, steps, max_steps, fast_forward_forced=False):
    """Advance forced moves locally; count each real game step toward the limit."""
    while True:
        observations, _, done = game.step(actions)
        steps += 1
        if done:
            return observations, True, steps, None
        if steps >= max_steps:
            return observations, True, steps, f"超过最大步数 {max_steps}"
        actions = forced_actions(observations) if fast_forward_forced else None
        if actions is None:
            return observations, False, steps, None


def _worker_main(
    connection: Connection,
    shared,
    capacity: int,
    max_steps: int,
    fast_forward_forced: bool = False,
) -> None:
    arrays = _shared_views(shared, capacity)
    games: Dict[int, Dict] = {}
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(("closed", None))
                return
            if command == "start":
                results = []
                for raw_spec in payload:
                    spec = GameSpec(**raw_spec)
                    game = (
                        SouthMatch(rounds_per_wind=spec.rounds_per_wind)
                        if spec.game_mode == "south"
                        else ThreePlayerMahjong()
                    )
                    try:
                        if spec.game_mode == "south":
                            game_observations = game.reset(
                                seed=spec.seed,
                                current_player=spec.current_player,
                            )
                        else:
                            game_observations = game.reset(
                                walls=_build_wall(spec.seed),
                                current_player=spec.current_player,
                            )
                        games[spec.slot] = {"game": game, "steps": 0}
                        _write_observations(spec.slot, game_observations, arrays)
                        results.append(
                            {"slot": spec.slot, "done": False, "steps": 0}
                        )
                    except Exception as exc:
                        games.pop(spec.slot, None)
                        arrays[2][spec.slot].fill(0)
                        results.append(
                            {
                                "slot": spec.slot,
                                "done": True,
                                "steps": 0,
                                "error": f"{type(exc).__name__}: {exc}",
                                "game": game,
                            }
                        )
                connection.send(("started", results))
                continue
            if command == "step":
                results = []
                for slot in payload:
                    state = games[slot]
                    game = state["game"]
                    try:
                        action_dict = {
                            game.agent_names[seat]: int(arrays[3][slot, seat])
                            for seat in range(MAX_PLAYERS)
                            if arrays[2][slot, seat]
                        }
                        game_observations, done, state["steps"], error = step_to_decision(
                            game, action_dict, state["steps"], max_steps, fast_forward_forced
                        )
                        if done:
                            arrays[2][slot].fill(0)
                            games.pop(slot, None)
                            results.append(
                                {
                                    "slot": slot,
                                    "done": True,
                                    "steps": state["steps"],
                                    "error": error,
                                    "game": game,
                                }
                            )
                        else:
                            _write_observations(slot, game_observations, arrays)
                            results.append(
                                {
                                    "slot": slot,
                                    "done": False,
                                    "steps": state["steps"],
                                }
                            )
                    except Exception as exc:
                        arrays[2][slot].fill(0)
                        games.pop(slot, None)
                        results.append(
                            {
                                "slot": slot,
                                "done": True,
                                "steps": state["steps"],
                                "error": f"{type(exc).__name__}: {exc}",
                                "game": game,
                            }
                        )
                connection.send(("stepped", results))
                continue
            raise ValueError(f"未知环境池命令：{command}")
    except EOFError:
        return
    except BaseException:
        try:
            connection.send(("fatal", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass


class ParallelGamePool:
    """Shared observations with optional overlap of simulation and inference."""

    def __init__(self, capacity: int, workers: int, max_steps: int = 300,
                 fast_forward_forced: bool = False, async_min_workers: int = 0):
        if capacity <= 0 or max_steps <= 0:
            raise ValueError("capacity 和 max_steps 必须为正数")
        if async_min_workers < 0:
            raise ValueError("async_min_workers must be nonnegative")
        self.capacity = int(capacity)
        self.worker_count = resolve_env_workers(workers, self.capacity)
        self.max_steps = int(max_steps)
        self.context = mp.get_context("spawn")
        self.shared = {
            "observations": RawArray(
                ctypes.c_float,
                self.capacity * MAX_PLAYERS * MATCH_CHANNELS * OBS_TILES,
            ),
            "masks": RawArray(
                ctypes.c_float, self.capacity * MAX_PLAYERS * ACTION_SIZE
            ),
            "active_agents": RawArray(
                ctypes.c_uint8, self.capacity * MAX_PLAYERS
            ),
            "actions": RawArray(
                ctypes.c_int16, self.capacity * MAX_PLAYERS
            ),
        }
        self.arrays = _shared_views(self.shared, self.capacity)
        self.parents: List[Connection] = []
        self.processes = []
        self.slot_workers: Dict[int, int] = {}
        self.active_slots = set()
        self.ready_slots = set()
        self.pending_workers = set()
        self.async_min_workers = int(async_min_workers)
        for _ in range(self.worker_count):
            parent, child = self.context.Pipe(duplex=True)
            process = self.context.Process(
                target=_worker_main,
                args=(child, self.shared, self.capacity, self.max_steps, fast_forward_forced),
                daemon=True,
            )
            process.start()
            child.close()
            self.parents.append(parent)
            self.processes.append(process)

    def _receive(self, worker_indices: Iterable[int], expected: str) -> List[Dict]:
        results = []
        for worker_index in worker_indices:
            try:
                kind, payload = self.parents[worker_index].recv()
            except EOFError as exc:
                raise RuntimeError(
                    f"环境 worker {worker_index} 意外退出"
                ) from exc
            if kind == "fatal":
                raise RuntimeError(
                    f"环境 worker {worker_index} 失败：\n{payload}"
                )
            if kind != expected:
                raise RuntimeError(
                    f"环境 worker {worker_index} 返回 {kind!r}，预期 {expected!r}"
                )
            results.extend(payload)
        return results

    def start(self, specs: Sequence[GameSpec]) -> List[Dict]:
        if self.active_slots:
            raise RuntimeError("上一批环境尚未结束，不能开始新批次")
        self.slot_workers.clear()
        if len(specs) > self.capacity:
            raise ValueError(f"环境数 {len(specs)} 超过容量 {self.capacity}")
        slots = [spec.slot for spec in specs]
        if len(set(slots)) != len(slots):
            raise ValueError("GameSpec.slot 不能重复")
        if any(slot < 0 or slot >= self.capacity for slot in slots):
            raise ValueError("GameSpec.slot 超出共享内存范围")

        by_worker = [[] for _ in range(self.worker_count)]
        for index, spec in enumerate(specs):
            worker_index = index % self.worker_count
            self.slot_workers[spec.slot] = worker_index
            by_worker[worker_index].append(
                {
                    "slot": spec.slot,
                    "seed": spec.seed,
                    "current_player": spec.current_player,
                    "game_mode": spec.game_mode,
                    "rounds_per_wind": spec.rounds_per_wind,
                }
            )
        used = [index for index, items in enumerate(by_worker) if items]
        for worker_index in used:
            self.parents[worker_index].send(("start", by_worker[worker_index]))
        results = self._receive(used, "started")
        self.active_slots = {
            result["slot"] for result in results if not result["done"]
        }
        self.ready_slots = self.active_slots.copy()
        return results

    def observations(self, slot: int) -> Dict[str, Dict[str, np.ndarray]]:
        observations, masks, active_agents, _ = self.arrays
        result = {}
        for seat in range(MAX_PLAYERS):
            if not active_agents[slot, seat]:
                continue
            rich = observations[slot, seat]
            result[f"player_{seat}"] = {
                "observation": rich[:6],
                "rich_observation": rich[:RICH_CHANNELS],
                "match_observation": rich,
                "action_mask": masks[slot, seat],
            }
        return result

    def step(self, actions_by_slot: Dict[int, Dict[str, int]]) -> List[Dict]:
        supplied = set(actions_by_slot)
        if supplied != self.ready_slots:
            missing = sorted(self.ready_slots - supplied)
            extra = sorted(supplied - self.ready_slots)
            raise ValueError(f"环境动作槽位不完整：missing={missing}, extra={extra}")
        _, _, active_agents, actions = self.arrays
        by_worker = [[] for _ in range(self.worker_count)]
        for slot, action_dict in actions_by_slot.items():
            expected_agents = {
                f"player_{seat}"
                for seat in range(MAX_PLAYERS)
                if active_agents[slot, seat]
            }
            if set(action_dict) != expected_agents:
                raise ValueError(
                    f"slot={slot} 动作玩家不完整："
                    f"expected={expected_agents}, actual={set(action_dict)}"
                )
            for agent_name, action in action_dict.items():
                seat = int(agent_name.rsplit("_", 1)[1])
                actions[slot, seat] = int(action)
            by_worker[self.slot_workers[slot]].append(slot)

        used = [index for index, slots in enumerate(by_worker) if slots]
        if self.pending_workers.intersection(used):
            raise RuntimeError("Cannot overwrite actions while a worker is stepping")
        for worker_index in used:
            self.parents[worker_index].send(("step", by_worker[worker_index]))
        self.pending_workers.update(used)
        if self.async_min_workers:
            # Return a useful inference batch as soon as enough workers finish;
            # slower workers continue while the main process runs the models.
            target = min(self.async_min_workers, len(self.pending_workers))
            finished = set()
            while len(finished) < target:
                remaining = self.pending_workers - finished
                connections = [self.parents[index] for index in sorted(remaining)]
                ready = set(wait(connections))
                finished.update(index for index in remaining if self.parents[index] in ready)
            # Drain any additional workers already ready, without another barrier.
            finished.update(index for index in self.pending_workers - finished
                            if self.parents[index].poll())
            received = sorted(finished)
        else:
            received = used
        results = self._receive(received, "stepped")
        self.pending_workers.difference_update(received)
        for result in results:
            if result["done"]:
                self.active_slots.discard(result["slot"])
        self.ready_slots = {result["slot"] for result in results if not result["done"]}
        return results

    def close(self) -> None:
        if not self.processes:
            return
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
        self.processes.clear()
        self.parents.clear()
        self.active_slots.clear()
        self.ready_slots.clear()
        self.pending_workers.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
        return False
