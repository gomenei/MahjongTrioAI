"""让多个三麻模型实际对局，并用座位平衡和置信区间公平比较。

hand 模式比较独立小局；south 模式比较完整东南风场的最终顺位与点差。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.match import SouthMatch
from mahjong_env.tile import Suit, Tile, Wind
from training.models import load_checkpoint_model
from training.cuda_rollout import ROLLOUT_INFERENCE
from training.parallel_env import (
    GameSpec,
    ParallelGamePool,
    resolve_env_workers,
)


# v3 修正 south 模式的 wins/win_rate 语义：它们现在统计实际小局和牌，
# 一位场数继续由 firsts/first_rate 单独记录。旧进度不能与新统计混用。
FORMAT_VERSION = 3
ACTION_SIZE = 177
PASS_ACTION = 176
Z95 = 1.959963984540054


@dataclass
class Policy:
    name: str
    checkpoint: Optional[Path]
    model: Optional[torch.nn.Module]
    observation_key: str = "observation"

    @property
    def is_random(self) -> bool:
        return self.model is None


@dataclass(frozen=True)
class MatchTask:
    seed: int
    rotation: int
    seat_models: Tuple[int, int, int]


@dataclass
class RunningGame:
    task: MatchTask
    game: Optional[ThreePlayerMahjong]
    observations: Dict
    steps: int = 0
    error: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="通过大量实际三麻对局、座位轮换和95%置信区间比较模型"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        metavar="NAME=CHECKPOINT",
        help="手动指定模型；不填则发现 training_runs/*/best.pt",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("training_runs")
    )
    parser.add_argument(
        "--legacy", type=Path, default=Path("model/model.pt"),
        help="自动发现时也尝试加入此模型；与候选权重内容相同则自动去重",
    )
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--fast-forward-forced", action="store_true")
    parser.add_argument("--rollout-cuda-graphs", action="store_true")
    parser.add_argument("--async-min-workers", type=int, default=0)
    parser.add_argument(
        "--action-mode", choices=("greedy", "sample"), default="greedy"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--game-mode", choices=("hand", "south"), default="hand",
        help="hand 比较独立小局；south 比较完整南风场顺位",
    )
    parser.add_argument(
        "--rounds-per-wind", type=int, choices=(3, 4), default=3
    )
    parser.add_argument(
        "--min-groups", type=int, default=2000,
        help="每组三模型至少使用多少组牌山；每组会轮换座位并实际打3局",
    )
    parser.add_argument(
        "--max-groups", type=int, default=10000,
        help="每组三模型最多使用多少组牌山",
    )
    parser.add_argument(
        "--target-ci", type=float, default=100.0,
        help="三者两两平均点差的95%%置信区间半宽均小于该值时提前停止",
    )
    parser.add_argument(
        "--batch-groups", type=int, default=64,
        help="并行推进多少组牌山；增大可提高GPU利用率但占用更多内存",
    )
    parser.add_argument(
        "--env-workers",
        type=int,
        default=0,
        help="环境步进子进程数；0 根据 CPU 自动选择，1 使用旧单进程路径",
    )
    parser.add_argument(
        "--max-steps", type=int, default=300,
        help="单局默认300；南风场请显式设为3000",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/battle")
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="忽略并替换现有断点；否则自动续跑",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        if not name.strip():
            raise ValueError(f"模型名称为空：{spec}")
        return name.strip(), Path(raw_path)
    path = Path(spec)
    return path.parent.name or path.stem, path


def discover_model_specs(args) -> List[Tuple[str, Path]]:
    if args.models:
        specs = [parse_model_spec(spec) for spec in args.models]
    else:
        paths = sorted(args.runs_dir.glob("*/best.pt"))
        if args.legacy.exists():
            paths.append(args.legacy)
        specs = [(path.parent.name, path) for path in paths]

    unique = []
    seen_digests = set()
    seen_names = set()
    for requested_name, path in specs:
        if not path.exists():
            raise FileNotFoundError(f"模型权重不存在：{path}")
        digest = file_digest(path)
        if digest in seen_digests:
            print(f"跳过重复权重：{path}")
            continue
        name = requested_name
        suffix = 2
        while name in seen_names:
            name = f"{requested_name}_{suffix}"
            suffix += 1
        seen_digests.add(digest)
        seen_names.add(name)
        unique.append((name, path.resolve()))
    return unique


def load_policies(args, device: torch.device) -> List[Policy]:
    policies = []
    for requested_name, path in discover_model_specs(args):
        model, metadata = load_checkpoint_model(path, map_location=device)
        model = model.to(device).eval()
        stored_name = str(metadata.get("model_name", "")).strip()
        name = requested_name if args.models else (stored_name or requested_name)
        if any(policy.name == name for policy in policies):
            name = requested_name
        policies.append(
            Policy(
                name=name,
                checkpoint=path,
                model=model,
                observation_key=str(
                    metadata.get("observation_key", "observation")
                ),
            )
        )
        print(f"已加载：{name} <- {path}")
    if args.include_random:
        policies.append(Policy(name="random", checkpoint=None, model=None))
    if len(policies) < 2:
        raise ValueError("实际对局比较至少需要2个不同模型")
    if len(policies) > 8:
        raise ValueError("模型超过8个会产生过多三模型组合，请分批比较")
    return policies


def build_wall(seed: int) -> List[Tile]:
    """生成局部随机数控制的牌山，不污染全局 random 状态。"""
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


def make_running_game(task: MatchTask, args) -> RunningGame:
    if getattr(args, "game_mode", "hand") == "south":
        game = SouthMatch(rounds_per_wind=args.rounds_per_wind)
        observations = game.reset(seed=task.seed, current_player=0)
    else:
        game = ThreePlayerMahjong()
        observations = game.reset(walls=build_wall(task.seed), current_player=0)
    return RunningGame(task=task, game=game, observations=observations)


@torch.inference_mode()
def choose_actions(
    policy: Policy,
    observations: Sequence[Dict],
    device: torch.device,
    action_mode: str,
    temperature: float,
    rng: np.random.Generator,
) -> List[int]:
    masks = np.asarray(
        [observation["action_mask"] for observation in observations],
        dtype=np.float32,
    )
    if policy.is_random:
        result = []
        for mask in masks:
            legal = np.flatnonzero(mask > 0)
            result.append(int(rng.choice(legal)) if len(legal) else PASS_ACTION)
        return result

    model_input = {
        "observation": torch.as_tensor(
            np.asarray(
                [
                    observation[policy.observation_key]
                    for observation in observations
                ],
                dtype=np.float32,
            ),
            device=device,
        ),
        "action_mask": torch.as_tensor(masks, device=device),
    }
    logits = ROLLOUT_INFERENCE(policy.model, model_input)
    if logits.ndim != 2 or logits.shape[1] != ACTION_SIZE:
        raise ValueError(
            f"{policy.name} 输出形状应为 (batch, {ACTION_SIZE})，实际为 {tuple(logits.shape)}"
        )
    legal = model_input["action_mask"] > 0
    logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    if action_mode == "greedy":
        return logits.argmax(dim=1).cpu().tolist()
    probabilities = torch.softmax(logits / temperature, dim=1)
    # multinomial 不接受 numpy Generator；全局 torch 种子已由 main 固定。
    return torch.multinomial(probabilities, 1).squeeze(1).cpu().tolist()


def run_tasks_serial(
    tasks: Sequence[MatchTask],
    policies: Sequence[Policy],
    args,
    device: torch.device,
    rng: np.random.Generator,
) -> List[RunningGame]:
    running = [make_running_game(task, args) for task in tasks]
    active = running[:]
    while active:
        requests_by_model = defaultdict(list)
        for item in active:
            for agent_name, observation in item.observations.items():
                seat = int(agent_name.rsplit("_", 1)[1])
                model_index = item.task.seat_models[seat]
                requests_by_model[model_index].append(
                    (item, agent_name, observation)
                )

        selected: Dict[int, Dict[str, int]] = {
            id(item): {} for item in active
        }
        for model_index, requests in requests_by_model.items():
            actions = choose_actions(
                policies[model_index],
                [request[2] for request in requests],
                device,
                args.action_mode,
                args.temperature,
                rng,
            )
            for (item, agent_name, _), action in zip(requests, actions):
                selected[id(item)][agent_name] = int(action)

        next_active = []
        for item in active:
            try:
                item.observations, _, done = item.game.step(selected[id(item)])
                item.steps += 1
                if not done and item.steps < args.max_steps:
                    next_active.append(item)
                elif not done:
                    item.error = f"超过最大步数 {args.max_steps}"
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
        active = next_active
    return running


def run_tasks_parallel(
    tasks: Sequence[MatchTask],
    policies: Sequence[Policy],
    args,
    device: torch.device,
    rng: np.random.Generator,
    env_pool: ParallelGamePool,
) -> List[RunningGame]:
    """worker 并行推进环境，所有模型推理仍集中在主进程/GPU。"""
    running = [
        RunningGame(task=task, game=None, observations={}) for task in tasks
    ]
    started = env_pool.start(
        [
            GameSpec(
                slot=slot,
                seed=task.seed,
                current_player=0,
                game_mode=getattr(args, "game_mode", "hand"),
                rounds_per_wind=getattr(args, "rounds_per_wind", 3),
            )
            for slot, task in enumerate(tasks)
        ]
    )
    active_by_slot = {}
    for result in started:
        slot = result["slot"]
        item = running[slot]
        item.steps = result["steps"]
        if result["done"]:
            item.error = result.get("error")
            item.game = result.get("game")
        else:
            item.observations = env_pool.observations(slot)
            active_by_slot[slot] = item

    while env_pool.active_slots:
        requests_by_model = defaultdict(list)
        for slot, item in active_by_slot.items():
            for agent_name, observation in item.observations.items():
                seat = int(agent_name.rsplit("_", 1)[1])
                model_index = item.task.seat_models[seat]
                requests_by_model[model_index].append(
                    (slot, agent_name, observation)
                )

        selected: Dict[int, Dict[str, int]] = {
            slot: {} for slot in active_by_slot
        }
        for model_index, requests in requests_by_model.items():
            actions = choose_actions(
                policies[model_index],
                [request[2] for request in requests],
                device,
                args.action_mode,
                args.temperature,
                rng,
            )
            for (slot, agent_name, _), action in zip(requests, actions):
                selected[slot][agent_name] = int(action)

        next_active = {}
        for result in env_pool.step(selected):
            slot = result["slot"]
            item = running[slot]
            item.steps = result["steps"]
            if result["done"]:
                item.error = result.get("error")
                item.game = result.get("game")
                item.observations = {}
            else:
                item.observations = env_pool.observations(slot)
                next_active[slot] = item
        active_by_slot = next_active
    return running


def run_tasks(
    tasks: Sequence[MatchTask],
    policies: Sequence[Policy],
    args,
    device: torch.device,
    rng: np.random.Generator,
    env_pool: Optional[ParallelGamePool] = None,
) -> List[RunningGame]:
    if env_pool is None:
        return run_tasks_serial(tasks, policies, args, device, rng)
    return run_tasks_parallel(tasks, policies, args, device, rng, env_pool)


def empty_model_result() -> Dict[str, float]:
    return {
        "hands": 0,
        "played_hands": 0,
        "rank_sum": 0.0,
        "firsts": 0,
        "seconds": 0,
        "thirds": 0,
        "wins": 0,
        "ron_wins": 0,
        "tsumo_wins": 0,
        "deal_ins": 0,
        "draws": 0,
        "invalid_actions": 0,
        "utility_sum": 0.0,
        "point_sum": 0.0,
        "winning_points_sum": 0.0,
    }


def round_up_100(value: float) -> int:
    return int(math.ceil(value / 100.0) * 100)


def basic_points(fan: int, fu: int) -> int:
    """计算日麻基本点；正数13番以上按累计役满封顶为单倍役满。"""
    if fan < 0:
        return 8000 * (-fan)
    if fan >= 13:
        return 8000
    if fan >= 11:
        return 6000
    if fan >= 8:
        return 4000
    if fan >= 6:
        return 3000
    if fan == 5:
        return 2000
    return min(2000, int(fu) * (2 ** (int(fan) + 2)))


def score_win(
    *,
    fan: int,
    fu: int,
    winner: int,
    discarder: Optional[int],
    dealer: int,
    honba: int,
) -> Tuple[List[int], int]:
    """按雀魂段位三麻自摸损规则返回三家点差及本手打点。

    荣和由放铳者全额支付；自摸只收实际存在的两份付款，不补北家份额。
    三麻每本场荣和加200，自摸时每名支付者加100。
    """
    base = basic_points(fan, fu)
    deltas = [0, 0, 0]
    if discarder is not None:
        multiplier = 6 if winner == dealer else 4
        payment = round_up_100(base * multiplier) + 200 * honba
        deltas[winner] += payment
        deltas[discarder] -= payment
        return deltas, payment

    for seat in range(3):
        if seat == winner:
            continue
        if winner == dealer or seat == dealer:
            payment = round_up_100(base * 2)
        else:
            payment = round_up_100(base)
        payment += 100 * honba
        deltas[seat] -= payment
        deltas[winner] += payment
    return deltas, deltas[winner]


def summarize_game(item: RunningGame) -> Optional[Dict[int, Dict[str, float]]]:
    if item.error:
        return None
    game = item.game
    if isinstance(game, SouthMatch):
        return summarize_match(item, game)
    by_model = defaultdict(empty_model_result)
    for seat, model_index in enumerate(item.task.seat_models):
        by_model[model_index]["hands"] += 1

    if "非法动作" in game.result_message:
        match = re_search_player(game.result_message)
        if match is None:
            return None
        offender = match
        for seat, model_index in enumerate(item.task.seat_models):
            value = -1.0 if seat == offender else 0.5
            by_model[model_index]["utility_sum"] += value
            point_value = -8000 if seat == offender else 4000
            by_model[model_index]["point_sum"] += point_value
            if seat == offender:
                by_model[model_index]["invalid_actions"] += 1
        return by_model

    if game.winner is None:
        for model_index in item.task.seat_models:
            by_model[model_index]["draws"] += 1
        return by_model

    winner = int(game.winner)
    winner_model = item.task.seat_models[winner]
    by_model[winner_model]["wins"] += 1
    dealer = next(
        seat
        for seat, agent in enumerate(game.agents)
        if agent.seatWind == Wind.East
    )
    if game.win_by == "自摸":
        by_model[winner_model]["tsumo_wins"] += 1
        point_deltas, winning_points = score_win(
            fan=game.fans[winner],
            fu=game.fus[winner],
            winner=winner,
            discarder=None,
            dealer=dealer,
            honba=game.honba,
        )
        for seat, model_index in enumerate(item.task.seat_models):
            by_model[model_index]["utility_sum"] += 1.0 if seat == winner else -0.5
    else:
        by_model[winner_model]["ron_wins"] += 1
        discarder = int(game.current_player)
        point_deltas, winning_points = score_win(
            fan=game.fans[winner],
            fu=game.fus[winner],
            winner=winner,
            discarder=discarder,
            dealer=dealer,
            honba=game.honba,
        )
        for seat, model_index in enumerate(item.task.seat_models):
            if seat == winner:
                by_model[model_index]["utility_sum"] += 1.0
            elif seat == discarder:
                by_model[model_index]["utility_sum"] -= 1.0
                by_model[model_index]["deal_ins"] += 1

    # 独立小局无法把流局供托带到下一局；有赢家时正常结算本局立直棒。
    riichi_seats = [
        seat for seat, declared in enumerate(game.isLiZhi) if declared
    ]
    for seat in riichi_seats:
        point_deltas[seat] -= 1000
    point_deltas[winner] += 1000 * len(riichi_seats)
    for seat, model_index in enumerate(item.task.seat_models):
        by_model[model_index]["point_sum"] += point_deltas[seat]
    by_model[winner_model]["winning_points_sum"] += winning_points
    return by_model


def summarize_match(
    item: RunningGame, game: SouthMatch
) -> Dict[int, Dict[str, float]]:
    """把完整南风场按最终顺位和最终点差汇总。"""
    by_model = defaultdict(empty_model_result)
    for seat, model_index in enumerate(item.task.seat_models):
        values = by_model[model_index]
        rank = int(game.final_ranks[seat])
        values["hands"] += 1  # 兼容统计代码：南风模式下统计单位是一整场
        values["played_hands"] += game.hand_count
        values["rank_sum"] += rank
        values[("firsts", "seconds", "thirds")[rank - 1]] += 1
        values["utility_sum"] += {1: 1.0, 2: 0.0, 3: -1.0}[rank]
        values["point_sum"] += (
            game.scores[seat] - game.starting_scores[seat]
        )
        values["deal_ins"] += sum(
            result.discarder == seat for result in game.history
        )
        winning_hands = [
            result for result in game.history if result.winner == seat
        ]
        values["wins"] += len(winning_hands)
        values["tsumo_wins"] += sum(
            result.win_by == "自摸" for result in winning_hands
        )
        values["ron_wins"] += sum(
            result.win_by is not None and result.win_by != "自摸"
            for result in winning_hands
        )
        values["winning_points_sum"] += sum(
            max(0, result.deltas[seat]) for result in winning_hands
        )
        values["draws"] += sum(
            result.winner is None for result in game.history
        )
        values["invalid_actions"] += sum(
            "非法动作" in result.message for result in game.history
        )
    return by_model


def re_search_player(message: str) -> Optional[int]:
    import re

    match = re.search(r"玩家\s*(\d+)", message)
    return int(match.group(1)) if match else None


def combine_rotation_results(
    results: Sequence[RunningGame],
    lineup: Tuple[int, int, int],
) -> Tuple[List[Dict], List[str]]:
    by_seed = defaultdict(list)
    for result in results:
        by_seed[result.task.seed].append(result)
    groups = []
    errors = []
    for seed, rotations in sorted(by_seed.items()):
        if len(rotations) != 3:
            errors.append(f"seed={seed}: 座位轮换数量不是3")
            continue
        summaries = [summarize_game(rotation) for rotation in rotations]
        if any(summary is None for summary in summaries):
            detail = "; ".join(
                rotation.error or rotation.game.result_message
                for rotation in rotations
                if rotation.error or summarize_game(rotation) is None
            )
            errors.append(f"seed={seed}: {detail}")
            continue
        models = {str(index): empty_model_result() for index in lineup}
        for summary in summaries:
            for model_index, values in summary.items():
                target = models[str(model_index)]
                for key, value in values.items():
                    target[key] += value
        groups.append({"seed": seed, "models": models})
    return groups, errors


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, math.inf
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean, math.inf
    half = Z95 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half, half


def weighted_cluster_ci(
    totals_and_weights: Sequence[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """按实际出场数计算均值，并把同一牌山轮换组视为一个独立样本。

    镜像阵容中，同一模型在一个组里可能出场3次或6次。直接对各组的席位均值
    做等权平均会破坏零和性质，因此中心值必须使用 sum(total)/sum(weight)。
    区间使用 ratio estimator 的 cluster-robust 标准误。
    """
    if not totals_and_weights:
        return 0.0, 0.0, 0.0, math.inf
    total_weight = sum(weight for _, weight in totals_and_weights)
    if total_weight <= 0:
        return 0.0, 0.0, 0.0, math.inf
    mean = sum(total for total, _ in totals_and_weights) / total_weight
    if len(totals_and_weights) < 2:
        return mean, mean, mean, math.inf
    residual_square_sum = sum(
        (total - mean * weight) ** 2
        for total, weight in totals_and_weights
    )
    cluster_count = len(totals_and_weights)
    standard_error = math.sqrt(
        cluster_count / (cluster_count - 1)
        * residual_square_sum
        / (total_weight * total_weight)
    )
    half = Z95 * standard_error
    return mean, mean - half, mean + half, half


def group_mean_utility(group: Dict, model_index: int) -> float:
    values = group["models"][str(model_index)]
    return values["utility_sum"] / max(values["hands"], 1)


def group_mean_points(group: Dict, model_index: int) -> float:
    values = group["models"][str(model_index)]
    return values["point_sum"] / max(values["hands"], 1)


def wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + Z95 * Z95 / total
    centre = (p + Z95 * Z95 / (2 * total)) / denominator
    margin = (
        Z95
        * math.sqrt(p * (1 - p) / total + Z95 * Z95 / (4 * total * total))
        / denominator
    )
    return centre - margin, centre + margin


def lineup_ci_half(groups: Sequence[Dict], lineup: Tuple[int, int, int]) -> float:
    halves = []
    unique_models = tuple(dict.fromkeys(lineup))
    for left, right in itertools.combinations(unique_models, 2):
        values = [
            group_mean_points(group, left)
            - group_mean_points(group, right)
            for group in groups
        ]
        halves.append(mean_ci(values)[3])
    return max(halves, default=math.inf)


def signature(args, policies: Sequence[Policy]) -> Dict:
    game_mode = getattr(args, "game_mode", "hand")
    result = {
        "format_version": FORMAT_VERSION,
        "models": [
            {
                "name": policy.name,
                "checkpoint": (
                    str(policy.checkpoint) if policy.checkpoint else None
                ),
                "sha256": (
                    file_digest(policy.checkpoint)
                    if policy.checkpoint else "random-policy"
                ),
            }
            for policy in policies
        ],
        "seed": args.seed,
        "action_mode": args.action_mode,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
    }
    # hand 模式保持旧断点签名完全不变；只有新南风场写入新字段。
    if game_mode == "south":
        result.update(
            game_mode=game_mode,
            rounds_per_wind=args.rounds_per_wind,
        )
    runtime_flags = {name: True for name in ("fast_forward_forced", "rollout_cuda_graphs")
                     if getattr(args, name, False)}
    if getattr(args, "async_min_workers", 0):
        runtime_flags["async_min_workers"] = args.async_min_workers
    if runtime_flags:
        result["runtime_flags"] = runtime_flags
    return result


def save_json_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_or_create_state(args, policies: Sequence[Policy], lineups) -> Dict:
    progress_path = args.output_dir / "progress.json"
    expected_signature = signature(args, policies)
    if progress_path.exists() and not args.fresh:
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("signature") != expected_signature:
            raise ValueError(
                "现有断点属于不同模型或参数。请恢复原参数，或添加 --fresh 重新评测。"
            )
        print(f"从断点续跑：{progress_path}")
        return state
    state = {
        "signature": expected_signature,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lineups": {
            ",".join(map(str, lineup)): {
                "model_indices": list(lineup),
                "groups": [],
                "next_group_index": 0,
                "complete": False,
                "errors": [],
            }
            for lineup in lineups
        },
    }
    save_json_atomic(progress_path, state)
    return state


def aggregate_reports(state: Dict, policies: Sequence[Policy]) -> Tuple[List[Dict], List[Dict]]:
    totals = {index: empty_model_result() for index in range(len(policies))}
    group_utilities = defaultdict(list)
    group_points = defaultdict(list)
    pair_point_differences = defaultdict(list)
    pair_utility_differences = defaultdict(list)

    for lineup_state in state["lineups"].values():
        lineup = tuple(lineup_state["model_indices"])
        unique_models = tuple(dict.fromkeys(lineup))
        for group in lineup_state["groups"]:
            for model_index in unique_models:
                values = group["models"][str(model_index)]
                for key, value in values.items():
                    totals[model_index][key] += value
                group_utilities[model_index].append(
                    (values["utility_sum"], values["hands"])
                )
                group_points[model_index].append(
                    (values["point_sum"], values["hands"])
                )
            for left, right in itertools.combinations(unique_models, 2):
                point_difference = (
                    group_mean_points(group, left)
                    - group_mean_points(group, right)
                )
                utility_difference = (
                    group_mean_utility(group, left)
                    - group_mean_utility(group, right)
                )
                pair_point_differences[(left, right)].append(point_difference)
                pair_utility_differences[(left, right)].append(utility_difference)

    rankings = []
    for index, policy in enumerate(policies):
        total = totals[index]
        hands = int(total["hands"])
        hand_denominator = int(total["played_hands"]) or hands
        wins = int(total["wins"])
        win_low, win_high = wilson_interval(wins, hand_denominator)
        utility, utility_low, utility_high, utility_half = weighted_cluster_ci(
            group_utilities[index]
        )
        points, points_low, points_high, points_half = weighted_cluster_ci(
            group_points[index]
        )
        rankings.append(
            {
                "model_name": policy.name,
                "checkpoint": (
                    str(policy.checkpoint) if policy.checkpoint else "random"
                ),
                "hands": hands,
                "played_hands": int(total["played_hands"]),
                "seed_groups": len(group_utilities[index]),
                "wins": wins,
                "win_rate": wins / max(hand_denominator, 1),
                "win_rate_ci95_low": win_low,
                "win_rate_ci95_high": win_high,
                "average_rank": total["rank_sum"] / max(hands, 1),
                "firsts": int(total["firsts"]),
                "first_rate": total["firsts"] / max(hands, 1),
                "seconds": int(total["seconds"]),
                "second_rate": total["seconds"] / max(hands, 1),
                "thirds": int(total["thirds"]),
                "third_rate": total["thirds"] / max(hands, 1),
                "ron_wins": int(total["ron_wins"]),
                "ron_rate": total["ron_wins"] / max(hand_denominator, 1),
                "tsumo_wins": int(total["tsumo_wins"]),
                "tsumo_rate": total["tsumo_wins"] / max(hand_denominator, 1),
                "tsumo_share_of_wins": total["tsumo_wins"] / max(wins, 1),
                "deal_ins": int(total["deal_ins"]),
                "deal_in_rate": total["deal_ins"] / max(hand_denominator, 1),
                "draw_rate": total["draws"] / max(hand_denominator, 1),
                "invalid_actions": int(total["invalid_actions"]),
                "average_points": points,
                "points_ci95_low": points_low,
                "points_ci95_high": points_high,
                "points_ci95_half_width": points_half,
                "average_winning_points": (
                    total["winning_points_sum"] / max(wins, 1)
                ),
                "mean_utility": utility,
                "utility_ci95_low": utility_low,
                "utility_ci95_high": utility_high,
                "utility_ci95_half_width": utility_half,
            }
        )
    match_mode = any(row["played_hands"] > 0 for row in rankings)
    rankings.sort(
        key=(
            (lambda row: (
                -row["average_rank"],
                row["average_points"],
                row["first_rate"],
                -row["third_rate"],
            ))
            if match_mode
            else (lambda row: (
                row["average_points"],
                row["mean_utility"],
                row["win_rate"],
                -row["deal_in_rate"],
            ))
        ),
        reverse=True,
    )
    for rank, row in enumerate(rankings, 1):
        row["rank"] = rank

    pairwise = []
    for (left, right), values in sorted(pair_point_differences.items()):
        mean, low, high, half = mean_ci(values)
        utility_mean, utility_low, utility_high, _ = mean_ci(
            pair_utility_differences[(left, right)]
        )
        if low > 0:
            conclusion = f"{policies[left].name} 显著更好"
        elif high < 0:
            conclusion = f"{policies[right].name} 显著更好"
        else:
            conclusion = "差异尚不显著"
        pairwise.append(
            {
                "model_a": policies[left].name,
                "model_b": policies[right].name,
                "paired_seed_groups": len(values),
                "mean_point_difference_a_minus_b": mean,
                "point_difference_ci95_low": low,
                "point_difference_ci95_high": high,
                "point_difference_ci95_half_width": half,
                "mean_utility_difference_a_minus_b": utility_mean,
                "utility_difference_ci95_low": utility_low,
                "utility_difference_ci95_high": utility_high,
                "conclusion": conclusion,
            }
        )
    return rankings, pairwise


def write_reports(args, state: Dict, policies: Sequence[Policy]) -> None:
    game_mode = getattr(args, "game_mode", "hand")
    rankings, pairwise = aggregate_reports(state, policies)
    completed = all(item["complete"] for item in state["lineups"].values())
    report = {
        "completed": completed,
        "game_mode": game_mode,
        "ranking_metric": (
            "完整南风场平均顺位与最终点差"
            if game_mode == "south"
            else "每次实际出场的平均点数收益"
        ),
        "scoring": (
            "雀魂段位三麻自摸损：荣和按标准日麻支付；自摸只收两家付款；"
            "每本场荣和200点、自摸每家100点；累计役满封顶单倍，役满可累计"
        ),
        "secondary_metric": (
            "顺位效用：一位+1、二位0、三位-1"
            if game_mode == "south"
            else (
                "零和收益：荣和=赢家+1/放铳者-1；"
                "自摸=赢家+1/另外两家各-0.5；流局=0"
            )
        ),
        "rate_definitions": {
            "win_rate": "胡牌小局数 / 实际参与小局数",
            "ron_rate": "荣和小局数 / 实际参与小局数",
            "tsumo_rate": "自摸小局数 / 实际参与小局数",
            "tsumo_share_of_wins": "自摸小局数 / 全部胡牌小局数",
            "deal_in_rate": "点炮（放铳）小局数 / 实际参与小局数",
            "first_rate": "完整比赛一位次数 / 完整比赛参与次数",
        },
        "confidence": (
            "平均点数按实际出场数加权，区间使用以牌山轮换组为独立样本的"
            "cluster-robust 95%正态近似；小局胡牌率使用 Wilson 区间"
        ),
        "limitations": (
            [
                "当前一次只处理一个和牌者，不评估双响结果",
                "南风场支持流局罚符、连庄、本场和立直棒；不处理途中流局",
            ]
            if game_mode == "south"
            else [
                "单局模式不代表完整南风场最终顺位率",
                "当前环境一次只处理一个和牌者，不评估双响结果",
                "独立小局的流局立直棒视为退回，不会带入下一局",
            ]
        ),
        "rankings": rankings,
        "pairwise": pairwise,
    }
    save_json_atomic(args.output_dir / "battle_report.json", report)

    rank_fields = list(rankings[0].keys()) if rankings else []
    with (args.output_dir / "battle_ranking.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rank_fields)
        writer.writeheader()
        writer.writerows(rankings)
    pair_fields = list(pairwise[0].keys()) if pairwise else []
    with (args.output_dir / "battle_pairwise.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pairwise)

    print("\n实战排名")
    if not completed:
        print("注意：当前为中途结果；镜像阵容尚未全部完成，出场数暂时可能不相等。")
    if game_mode == "south":
        print(
            f"{'排名':<6}{'模型':<22}{'南风场':>8}{'小局':>9}"
            f"{'胡牌率':>10}{'自摸率':>10}{'点炮率':>10}"
            f"{'一位率':>10}{'平均顺位':>10}{'平均点数':>11}"
        )
    else:
        print(
            f"{'排名':<6}{'模型':<25}{'实战场数':>10}"
            f"{'胡牌率':>11}{'自摸率':>11}{'点炮率':>11}"
            f"{'平均点数':>12}{'点数95% CI':>24}"
        )
    for row in rankings:
        interval = f"[{row['points_ci95_low']:+.0f}, {row['points_ci95_high']:+.0f}]"
        if game_mode == "south":
            print(
                f"{row['rank']:<6}{row['model_name']:<22}{row['hands']:>8,}"
                f"{row['played_hands']:>9,}{row['win_rate']:>9.2%}"
                f"{row['tsumo_rate']:>9.2%}{row['deal_in_rate']:>9.2%}"
                f"{row['first_rate']:>9.2%}{row['average_rank']:>10.3f}"
                f"{row['average_points']:>+11.1f}"
            )
        else:
            print(
                f"{row['rank']:<6}{row['model_name']:<25}{row['hands']:>10,}"
                f"{row['win_rate']:>10.2%}{row['tsumo_rate']:>11.2%}"
                f"{row['deal_in_rate']:>11.2%}{row['average_points']:>+12.1f}"
                f"{interval:>24}"
            )
    print(f"\n详细结果：{args.output_dir / 'battle_report.json'}")
    print(f"两两比较：{args.output_dir / 'battle_pairwise.csv'}")


def build_lineups(policy_count: int) -> List[Tuple[int, int, int]]:
    if policy_count == 2:
        # 两模型时使用镜像阵容；合并统计后双方出场数完全相同。
        return [(0, 0, 1), (0, 1, 1)]
    return list(itertools.combinations(range(policy_count), 3))


def tournament(
    args,
    policies: Sequence[Policy],
    device: torch.device,
    env_pool: Optional[ParallelGamePool] = None,
) -> Dict:
    lineups = build_lineups(len(policies))
    state = load_or_create_state(args, policies, lineups)
    progress_path = args.output_dir / "progress.json"
    rng = np.random.default_rng(args.seed)

    for lineup_number, lineup in enumerate(lineups, 1):
        key = ",".join(map(str, lineup))
        lineup_state = state["lineups"][key]
        groups = lineup_state["groups"]
        lineup_state.setdefault("next_group_index", len(groups))
        if lineup_state["complete"]:
            continue
        names = " / ".join(policies[index].name for index in lineup)
        print(f"\n组合 {lineup_number}/{len(lineups)}：{names}")

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
                    tasks.append(
                        MatchTask(
                            seed=seed,
                            rotation=rotation,
                            seat_models=seat_models,
                        )
                    )
            results = run_tasks(
                tasks, policies, args, device, rng, env_pool=env_pool
            )
            new_groups, errors = combine_rotation_results(results, lineup)
            groups.extend(new_groups)
            lineup_state["next_group_index"] += count
            lineup_state["errors"].extend(errors)
            half = lineup_ci_half(groups, lineup)
            print(
                f"  有效牌山组 {len(groups):,} / 实战局 {len(groups) * 3:,} / "
                f"最宽点差95% CI ±{half:.1f}点 / 模拟异常 {len(lineup_state['errors'])}"
            )
            save_json_atomic(progress_path, state)

            if len(groups) >= args.min_groups and half <= args.target_ci:
                lineup_state["complete"] = True
                break
            if errors and not new_groups:
                raise RuntimeError(f"整个批次均模拟失败，请检查 {args.output_dir / 'progress.json'}")

        if lineup_state["next_group_index"] >= args.max_groups:
            lineup_state["complete"] = True
        save_json_atomic(progress_path, state)
        write_reports(args, state, policies)
    return state


def validate_args(args) -> None:
    if args.min_groups <= 1:
        raise ValueError("--min-groups 必须大于1")
    if args.max_groups < args.min_groups:
        raise ValueError("--max-groups 不能小于 --min-groups")
    if args.batch_groups <= 0 or args.max_steps <= 0:
        raise ValueError("--batch-groups 和 --max-steps 必须为正数")
    if args.env_workers < 0:
        raise ValueError("--env-workers 不能为负数；0 表示自动")
    if args.target_ci <= 0:
        raise ValueError("--target-ci 必须为正数")
    if args.temperature <= 0:
        raise ValueError("--temperature 必须为正数")


def main():
    args = parse_args()
    validate_args(args)
    ROLLOUT_INFERENCE.enabled = args.rollout_cuda_graphs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = load_policies(args, device)
    lineups = build_lineups(len(policies))
    print(
        f"设备：{device}；模型数：{len(policies)}；"
        f"对局阵容数：{len(lineups)}"
    )
    started = time.perf_counter()
    capacity = args.batch_groups * 3
    worker_count = resolve_env_workers(args.env_workers, capacity)
    env_pool = None
    if worker_count > 1:
        env_pool = ParallelGamePool(
            capacity=capacity,
            workers=worker_count,
            max_steps=args.max_steps,
            fast_forward_forced=args.fast_forward_forced,
            async_min_workers=args.async_min_workers,
        )
        print(
            f"环境池：{worker_count} 个持久化进程；"
            "模型推理集中在主进程"
        )
    else:
        print("环境池：单进程兼容模式")
    try:
        state = tournament(args, policies, device, env_pool=env_pool)
    except KeyboardInterrupt:
        print("\n已停止；最近一个批次前的进度已经保存，下次同命令可续跑。")
        return
    finally:
        if env_pool is not None:
            env_pool.close()
    write_reports(args, state, policies)
    print(f"总耗时：{time.perf_counter() - started:.1f} 秒")


if __name__ == "__main__":
    main()
