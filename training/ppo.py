"""用 PPO 对现有三麻监督策略做一个可复现的强化学习微调实验。

策略从已有 checkpoint 热启动，训练时只控制一个轮换座位；另外两家由冻结
baseline 或历史策略快照控制。终局奖励使用雀魂段位三麻自摸损点差。
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from evaluation.battle import (
    MatchTask,
    Policy as BattlePolicy,
    build_wall,
    combine_rotation_results,
    run_tasks,
    score_win,
)
from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.match import SouthMatch
from mahjong_env.tile import Wind
from training.models import load_checkpoint_model
from training.ppo_support import OpponentPool, masked_reference_kl
from training.cuda_rollout import ROLLOUT_INFERENCE
from training.parallel_env import (
    GameSpec,
    ParallelGamePool,
    resolve_env_workers,
)


PASS_ACTION = 176
CHECKPOINT_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="从现有三麻模型热启动并使用 PPO 做点数奖励微调"
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("training_runs/tile_transformer/best.pt"),
        help="策略和冻结对手的初始权重",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ppo_runs/ppo_trial")
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--torch-threads", type=int, default=4,
                        help="主进程CPU张量线程数，避免与环境进程争用CPU")
    parser.add_argument("--fast-forward-forced", action="store_true",
                        help="环境进程直接推进唯一合法动作，减少进程通信")
    parser.add_argument("--rollout-cuda-graphs", action="store_true",
                        help="用CUDA图重放采集推理，减少小批次GPU调用开销")
    parser.add_argument("--async-min-workers", type=int, default=0,
                        help="收到至少此数量的环境进程结果即开始推理；0为同步等待")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--game-mode", choices=("hand", "south"), default="hand",
        help="hand 训练单局；south 训练完整东南风场",
    )
    parser.add_argument(
        "--rounds-per-wind", type=int, choices=(3, 4), default=3,
        help="三麻通常为3；设为4可运行东一到南四的实验规则",
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument(
        "--hands-per-update", type=int, default=256,
        help="每次 PPO 更新前完整打完的小局数",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--reward-scale", type=float, default=8000.0,
        help="终局点差除以此数后作为 PPO 奖励",
    )
    parser.add_argument("--reward-clip", type=float, default=6.0)
    parser.add_argument(
        "--rank-reward-weight", type=float, default=1.0,
        help="南风场奖励中一位+1、二位0、三位-1的权重",
    )
    parser.add_argument(
        "--policy-temperature", type=float, default=1.0,
        help="训练策略采样温度；对手和正式评估始终使用贪心动作",
    )
    parser.add_argument(
        "--snapshot-ratio", type=float, default=0.25,
        help="使用历史策略快照而非原始 baseline 作为对手的概率",
    )
    parser.add_argument("--snapshot-interval", type=int, default=25)
    parser.add_argument("--opponent-pool-size", type=int, default=1,
                        help="保留的历史策略数量；1兼容原先的单快照训练")
    parser.add_argument("--opponent-models", nargs="*", default=[],
                        help="对手池初始权重，须与基础模型使用相同网络和观测")
    parser.add_argument("--reference-kl-coef", type=float, default=0.0,
                        help="相对冻结基础策略的KL约束强度；0关闭")
    parser.add_argument("--critic-warmup-updates", type=int, default=0,
                        help="开始时仅拟合价值网络的更新次数")
    parser.add_argument("--max-failure-rate", type=float, default=0.05,
                        help="模拟失败率超过此值时停止，防止训练有偏样本")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=0,
        help="每隔多少次更新保存一个仅含策略的时间点；0表示不额外保存",
    )
    parser.add_argument("--evaluation-interval", type=int, default=20)
    parser.add_argument(
        "--max-hours", type=float, default=0.0,
        help="累计训练时长上限；0表示只按 updates 停止",
    )
    parser.add_argument(
        "--checkpoint-minutes", type=float, default=0.0,
        help="按累计训练分钟保存轻量策略时间点；0表示关闭",
    )
    parser.add_argument(
        "--evaluation-groups", type=int, default=256,
        help="固定评估牌山组数；每组轮换座位打3局",
    )
    parser.add_argument(
        "--env-workers",
        type=int,
        default=0,
        help="环境步进子进程数；0 根据 CPU 自动选择，1 使用旧单进程路径",
    )
    parser.add_argument(
        "--max-steps", type=int, default=300,
        help="单个训练环境最多决策步数；南风场需要明显大于单局",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="忽略 output-dir/last.pt，从基础模型重新开始",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def validate_args(args) -> None:
    positive = {
        "updates": args.updates,
        "hands-per-update": args.hands_per_update,
        "ppo-epochs": args.ppo_epochs,
        "minibatch-size": args.minibatch_size,
        "actor-lr": args.actor_lr,
        "critic-lr": args.critic_lr,
        "reward-scale": args.reward_scale,
        "policy-temperature": args.policy_temperature,
        "snapshot-interval": args.snapshot_interval,
        "opponent-pool-size": args.opponent_pool_size,
        "evaluation-groups": args.evaluation_groups,
        "max-steps": args.max_steps,
        "torch-threads": args.torch_threads,
    }
    invalid = [name for name, value in positive.items() if value <= 0 or not math.isfinite(value)]
    if invalid:
        raise ValueError("以下参数必须为正数：" + "、".join(invalid))
    if args.env_workers < 0:
        raise ValueError("--env-workers 不能为负数；0 表示自动")
    if args.evaluation_interval < 0:
        raise ValueError("--evaluation-interval 不能为负数；0 表示关闭固定评估")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval 不能为负数")
    if args.max_hours < 0 or args.checkpoint_minutes < 0:
        raise ValueError("--max-hours 和 --checkpoint-minutes 不能为负数")
    if args.rank_reward_weight < 0:
        raise ValueError("--rank-reward-weight 不能为负数")
    if args.reference_kl_coef < 0 or not math.isfinite(args.reference_kl_coef):
        raise ValueError("--reference-kl-coef 必须是非负有限数")
    if args.critic_warmup_updates < 0:
        raise ValueError("--critic-warmup-updates 不能为负数")
    if len(args.opponent_models) >= args.opponent_pool_size:
        raise ValueError("对手池需要为当前策略保留一个位置，请增大 --opponent-pool-size")
    for name, value in (
        ("gamma", args.gamma),
        ("gae-lambda", args.gae_lambda),
        ("clip-ratio", args.clip_ratio),
        ("snapshot-ratio", args.snapshot_ratio),
        ("max-failure-rate", args.max_failure_rate),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"--{name} 必须在 0 到 1 之间")


class ValueNetwork(nn.Module):
    """仅在 PPO 训练中使用的 critic；不会改变最终策略的输入和参数。"""

    def __init__(self, obs_channels: int = 6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(obs_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 30, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.float()).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, actor: nn.Module, obs_channels: int = 6):
        super().__init__()
        self.actor = actor
        self.critic = ValueNetwork(obs_channels)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        return self.actor(inputs), self.critic(inputs["observation"])


@dataclass
class Transition:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    log_probability: float
    value: float


@dataclass
class TrainingEpisode:
    game: Optional[ThreePlayerMahjong]
    observations: Dict
    learner_seat: int
    opponent_kind: str
    episode_id: int
    trajectory: List[Transition] = field(default_factory=list)
    steps: int = 0
    error: Optional[str] = None


def choose_opponent(args, rng, snapshot=None):
    if rng.random() >= args.snapshot_ratio:
        return "baseline"
    return snapshot.sample(rng) if isinstance(snapshot, OpponentPool) else "snapshot"


def rollout_opponents(baseline, snapshot):
    historical = (snapshot.policies() if isinstance(snapshot, OpponentPool)
                  else {"snapshot": snapshot})
    return {"baseline": baseline, **historical}


def make_episode(episode_id: int, args, rng: np.random.Generator, snapshot=None) -> TrainingEpisode:
    learner_seat = episode_id % 3
    opponent_kind = choose_opponent(args, rng, snapshot)
    if args.game_mode == "south":
        game = SouthMatch(rounds_per_wind=args.rounds_per_wind)
        observations = game.reset(seed=args.seed + episode_id, current_player=0)
    else:
        game = ThreePlayerMahjong()
        observations = game.reset(
            walls=build_wall(args.seed + episode_id),
            current_player=0,
        )
    return TrainingEpisode(
        game=game,
        observations=observations,
        learner_seat=learner_seat,
        opponent_kind=opponent_kind,
        episode_id=episode_id,
    )


def to_model_input(
    observations: Sequence[Dict],
    device: torch.device,
    observation_key: str = "observation",
):
    return {
        "observation": torch.as_tensor(
            np.asarray(
                [item[observation_key] for item in observations],
                dtype=np.float32,
            ),
            device=device,
        ),
        "action_mask": torch.as_tensor(
            np.asarray(
                [item["action_mask"] for item in observations],
                dtype=np.float32,
            ),
            device=device,
        ),
    }


@torch.inference_mode()
def opponent_actions(
    model: nn.Module,
    observations: Sequence[Dict],
    device: torch.device,
    observation_key: str = "observation",
) -> List[int]:
    if not observations:
        return []
    inputs = to_model_input(observations, device, observation_key)
    logits = ROLLOUT_INFERENCE(model, inputs)
    legal = inputs["action_mask"] > 0
    logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    return logits.argmax(dim=1).cpu().tolist()


@torch.inference_mode()
def learner_actions(
    actor_critic: ActorCritic,
    observations: Sequence[Dict],
    device: torch.device,
    temperature: float,
    observation_key: str = "observation",
):
    inputs = to_model_input(observations, device, observation_key)
    logits, values = ROLLOUT_INFERENCE(actor_critic, inputs)
    distribution = Categorical(logits=logits / temperature)
    actions = distribution.sample()
    log_probabilities = distribution.log_prob(actions)
    return (
        actions.cpu().tolist(),
        log_probabilities.cpu().tolist(),
        values.cpu().tolist(),
    )


def terminal_point_deltas(game: ThreePlayerMahjong) -> List[int]:
    """返回一个完整小局的三家点差；流局供托视为退回。"""
    if "非法动作" in game.result_message:
        import re

        match = re.search(r"玩家\s*(\d+)", game.result_message)
        if not match:
            return [0, 0, 0]
        offender = int(match.group(1))
        return [-8000 if seat == offender else 4000 for seat in range(3)]
    if game.winner is None:
        return [0, 0, 0]

    winner = int(game.winner)
    dealer = next(
        seat
        for seat, agent in enumerate(game.agents)
        if agent.seatWind == Wind.East
    )
    discarder = None if game.win_by == "自摸" else int(game.current_player)
    deltas, _ = score_win(
        fan=game.fans[winner],
        fu=game.fus[winner],
        winner=winner,
        discarder=discarder,
        dealer=dealer,
        honba=game.honba,
    )
    riichi_seats = [
        seat for seat, declared in enumerate(game.isLiZhi) if declared
    ]
    for seat in riichi_seats:
        deltas[seat] -= 1000
    deltas[winner] += 1000 * len(riichi_seats)
    return deltas


def finish_episode(episode: TrainingEpisode, args) -> Dict:
    game = episode.game
    if game is None:
        raise RuntimeError(f"episode={episode.episode_id} 缺少终局状态")
    if isinstance(game, SouthMatch):
        learner_points = (
            game.scores[episode.learner_seat]
            - game.starting_scores[episode.learner_seat]
        )
        rank = game.final_ranks[episode.learner_seat]
        rank_bonus = {1: 1.0, 2: 0.0, 3: -1.0}[rank]
        raw_reward = (
            learner_points / args.reward_scale
            + args.rank_reward_weight * rank_bonus
        )
        winning_hands = [
            result
            for result in game.history
            if result.winner == episode.learner_seat
        ]
        win = len(winning_hands)
        tsumo_win = sum(
            result.win_by == "自摸" for result in winning_hands
        )
        ron_win = sum(
            result.win_by is not None and result.win_by != "自摸"
            for result in winning_hands
        )
        deal_ins = sum(
            result.discarder == episode.learner_seat
            for result in game.history
        )
        first = int(rank == 1)
        draw = sum(result.winner is None for result in game.history)
        invalid = int(any("非法动作" in result.message for result in game.history))
        played_hands = game.hand_count
    else:
        points = terminal_point_deltas(game)
        learner_points = points[episode.learner_seat]
        raw_reward = learner_points / args.reward_scale
        rank = None
        deal_ins = int(
            game.winner is not None
            and game.win_by != "自摸"
            and game.current_player == episode.learner_seat
        )
        win = int(game.winner == episode.learner_seat)
        tsumo_win = int(win and game.win_by == "自摸")
        ron_win = int(win and game.win_by != "自摸")
        first = None
        draw = int(game.winner is None)
        invalid = int("非法动作" in game.result_message)
        played_hands = 1
    reward = float(
        np.clip(
            raw_reward,
            -args.reward_clip,
            args.reward_clip,
        )
    )
    return {
        "episode": episode,
        "points": learner_points,
        "reward": reward,
        "win": win,
        "ron_win": ron_win,
        "tsumo_win": tsumo_win,
        "first": first,
        "rank": rank,
        "deal_in": deal_ins,
        "draw": draw,
        "invalid": invalid,
        "played_hands": played_hands,
    }


def collect_rollout_serial(
    actor_critic: ActorCritic,
    baseline: nn.Module,
    snapshot: nn.Module,
    args,
    device: torch.device,
    rng: np.random.Generator,
    first_episode_id: int,
):
    actor_critic.eval()
    baseline.eval()
    opponents = rollout_opponents(baseline, snapshot)
    for model in opponents.values():
        model.eval()
    episodes = [
        make_episode(first_episode_id + offset, args, rng, snapshot)
        for offset in range(args.hands_per_update)
    ]
    active = episodes[:]

    while active:
        learner_requests = []
        opponent_requests = {kind: [] for kind in opponents}
        selected = {id(episode): {} for episode in active}

        for episode in active:
            for agent_name, observation in episode.observations.items():
                seat = int(agent_name.rsplit("_", 1)[1])
                legal = np.flatnonzero(
                    np.asarray(observation["action_mask"]) > 0
                )
                if len(legal) <= 1:
                    selected[id(episode)][agent_name] = (
                        int(legal[0]) if len(legal) else PASS_ACTION
                    )
                elif seat == episode.learner_seat:
                    learner_requests.append((episode, agent_name, observation))
                else:
                    opponent_requests[episode.opponent_kind].append(
                        (episode, agent_name, observation)
                    )

        if learner_requests:
            actions, log_probabilities, values = learner_actions(
                actor_critic,
                [request[2] for request in learner_requests],
                device,
                args.policy_temperature,
                args.observation_key,
            )
            for request, action, log_probability, value in zip(
                learner_requests, actions, log_probabilities, values
            ):
                episode, agent_name, observation = request
                selected[id(episode)][agent_name] = int(action)
                episode.trajectory.append(
                    Transition(
                        observation=np.asarray(
                            observation[args.observation_key], dtype=np.float32
                        ).copy(),
                        action_mask=np.asarray(
                            observation["action_mask"], dtype=np.float32
                        ).copy(),
                        action=int(action),
                        log_probability=float(log_probability),
                        value=float(value),
                    )
                )

        for kind, model in opponents.items():
            requests = opponent_requests[kind]
            actions = opponent_actions(
                model,
                [request[2] for request in requests],
                device,
                args.observation_key,
            )
            for (episode, agent_name, _), action in zip(requests, actions):
                selected[id(episode)][agent_name] = int(action)

        next_active = []
        for episode in active:
            try:
                episode.observations, _, done = episode.game.step(
                    selected[id(episode)]
                )
                episode.steps += 1
                if not done and episode.steps < args.max_steps:
                    next_active.append(episode)
                elif not done:
                    episode.error = f"超过最大步数 {args.max_steps}"
            except Exception as exc:
                episode.error = f"{type(exc).__name__}: {exc}"
        active = next_active

    completed = [
        finish_episode(episode, args)
        for episode in episodes
        if episode.error is None
    ]
    failures = [f"episode={episode.episode_id} seed={args.seed + episode.episode_id} "
                f"steps={episode.steps}: {episode.error}"
                for episode in episodes if episode.error]
    return completed, failures


def collect_rollout_parallel(
    actor_critic: ActorCritic,
    baseline: nn.Module,
    snapshot: nn.Module,
    args,
    device: torch.device,
    rng: np.random.Generator,
    first_episode_id: int,
    env_pool: ParallelGamePool,
):
    """并行推进牌局，主进程一次性完成各模型的 GPU batch 推理。"""
    actor_critic.eval()
    baseline.eval()
    opponents = rollout_opponents(baseline, snapshot)
    for model in opponents.values():
        model.eval()
    episodes = []
    for offset in range(args.hands_per_update):
        episode_id = first_episode_id + offset
        episodes.append(
            TrainingEpisode(
                game=None,
                observations={},
                learner_seat=episode_id % 3,
                opponent_kind=choose_opponent(args, rng, snapshot),
                episode_id=episode_id,
            )
        )

    started = env_pool.start(
        [
            GameSpec(
                slot=slot,
                seed=args.seed + episode.episode_id,
                current_player=0,
                game_mode=args.game_mode,
                rounds_per_wind=args.rounds_per_wind,
            )
            for slot, episode in enumerate(episodes)
        ]
    )
    active_by_slot = {}
    for result in started:
        slot = result["slot"]
        episode = episodes[slot]
        episode.steps = result["steps"]
        if result["done"]:
            episode.error = result.get("error")
            episode.game = result.get("game")
        else:
            episode.observations = env_pool.observations(slot)
            active_by_slot[slot] = episode

    while env_pool.active_slots:
        learner_requests = []
        opponent_requests = {kind: [] for kind in opponents}
        selected = {slot: {} for slot in active_by_slot}

        for slot, episode in active_by_slot.items():
            for agent_name, observation in episode.observations.items():
                seat = int(agent_name.rsplit("_", 1)[1])
                legal = np.flatnonzero(
                    np.asarray(observation["action_mask"]) > 0
                )
                if len(legal) <= 1:
                    selected[slot][agent_name] = (
                        int(legal[0]) if len(legal) else PASS_ACTION
                    )
                elif seat == episode.learner_seat:
                    learner_requests.append(
                        (slot, episode, agent_name, observation)
                    )
                else:
                    opponent_requests[episode.opponent_kind].append(
                        (slot, agent_name, observation)
                    )

        if learner_requests:
            actions, log_probabilities, values = learner_actions(
                actor_critic,
                [request[3] for request in learner_requests],
                device,
                args.policy_temperature,
                args.observation_key,
            )
            for request, action, log_probability, value in zip(
                learner_requests, actions, log_probabilities, values
            ):
                slot, episode, agent_name, observation = request
                selected[slot][agent_name] = int(action)
                episode.trajectory.append(
                    Transition(
                        observation=np.asarray(
                            observation[args.observation_key],
                            dtype=np.float32,
                        ).copy(),
                        action_mask=np.asarray(
                            observation["action_mask"], dtype=np.float32
                        ).copy(),
                        action=int(action),
                        log_probability=float(log_probability),
                        value=float(value),
                    )
                )

        for kind, model in opponents.items():
            requests = opponent_requests[kind]
            actions = opponent_actions(
                model,
                [request[2] for request in requests],
                device,
                args.observation_key,
            )
            for (slot, agent_name, _), action in zip(requests, actions):
                selected[slot][agent_name] = int(action)

        next_active = {}
        for result in env_pool.step(selected):
            slot = result["slot"]
            episode = episodes[slot]
            episode.steps = result["steps"]
            if result["done"]:
                episode.error = result.get("error")
                episode.game = result.get("game")
                episode.observations = {}
            else:
                episode.observations = env_pool.observations(slot)
                next_active[slot] = episode
        active_by_slot = next_active

    completed = [
        finish_episode(episode, args)
        for episode in episodes
        if episode.error is None
    ]
    failures = [f"episode={episode.episode_id} seed={args.seed + episode.episode_id} "
                f"steps={episode.steps}: {episode.error}"
                for episode in episodes if episode.error]
    return completed, failures


def collect_rollout(
    actor_critic: ActorCritic,
    baseline: nn.Module,
    snapshot: nn.Module,
    args,
    device: torch.device,
    rng: np.random.Generator,
    first_episode_id: int,
    env_pool: Optional[ParallelGamePool] = None,
):
    if env_pool is None:
        return collect_rollout_serial(
            actor_critic,
            baseline,
            snapshot,
            args,
            device,
            rng,
            first_episode_id,
        )
    return collect_rollout_parallel(
        actor_critic,
        baseline,
        snapshot,
        args,
        device,
        rng,
        first_episode_id,
        env_pool,
    )


def build_ppo_batch(completed: Sequence[Dict], args):
    observations = []
    masks = []
    actions = []
    old_log_probabilities = []
    old_values = []
    advantages = []
    returns = []

    for result in completed:
        trajectory = result["episode"].trajectory
        if not trajectory:
            continue
        episode_advantages = [0.0] * len(trajectory)
        episode_returns = [0.0] * len(trajectory)
        next_value = 0.0
        next_advantage = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            reward = result["reward"] if index == len(trajectory) - 1 else 0.0
            value = trajectory[index].value
            delta = reward + args.gamma * next_value - value
            current_advantage = (
                delta
                + args.gamma
                * args.gae_lambda
                * next_advantage
            )
            episode_advantages[index] = current_advantage
            episode_returns[index] = current_advantage + value
            next_value = value
            next_advantage = current_advantage

        for transition, advantage, return_value in zip(
            trajectory, episode_advantages, episode_returns
        ):
            observations.append(transition.observation)
            masks.append(transition.action_mask)
            actions.append(transition.action)
            old_log_probabilities.append(transition.log_probability)
            old_values.append(transition.value)
            advantages.append(advantage)
            returns.append(return_value)

    if not observations:
        raise RuntimeError("本批次没有需要策略判断的动作")
    advantages_array = np.asarray(advantages, dtype=np.float32)
    advantages_array = (
        advantages_array - advantages_array.mean()
    ) / (advantages_array.std() + 1e-8)
    return {
        "observation": np.asarray(observations, dtype=np.float32),
        "action_mask": np.asarray(masks, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "old_log_probabilities": np.asarray(
            old_log_probabilities, dtype=np.float32
        ),
        "old_values": np.asarray(old_values, dtype=np.float32),
        "advantages": advantages_array,
        "returns": np.asarray(returns, dtype=np.float32),
    }


def ppo_update(
    actor_critic: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, np.ndarray],
    args,
    device: torch.device,
    reference: Optional[nn.Module] = None,
    update: int = 1,
):
    actor_critic.actor.eval()  # 关闭 dropout，保证新旧 log-prob 可比较
    actor_critic.critic.train()
    sample_count = len(batch["actions"])
    tensors = {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.items()
    }
    totals = defaultdict(float)
    processed = 0
    epochs_completed = 0
    reference_coef = getattr(args, "reference_kl_coef", 0.0)
    warmup = update <= getattr(args, "critic_warmup_updates", 0)
    if reference_coef > 0 and reference is None:
        raise ValueError("KL约束需要冻结的参考策略")
    if reference is not None:
        reference.eval()

    for epoch in range(args.ppo_epochs):
        permutation = torch.randperm(sample_count, device=device)
        epoch_kl = 0.0
        epoch_samples = 0
        for start in range(0, sample_count, args.minibatch_size):
            indices = permutation[start : start + args.minibatch_size]
            inputs = {
                "observation": tensors["observation"][indices],
                "action_mask": tensors["action_mask"][indices],
            }
            with torch.set_grad_enabled(not warmup):
                logits = actor_critic.actor(inputs)
            values = actor_critic.critic(inputs["observation"])
            distribution = Categorical(
                logits=logits / args.policy_temperature
            )
            new_log_probabilities = distribution.log_prob(
                tensors["actions"][indices]
            )
            log_ratio = (
                new_log_probabilities
                - tensors["old_log_probabilities"][indices]
            )
            ratio = log_ratio.exp()
            minibatch_advantages = tensors["advantages"][indices]
            unclipped = ratio * minibatch_advantages
            clipped = (
                ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio)
                * minibatch_advantages
            )
            policy_loss = -torch.min(unclipped, clipped).mean()

            old_values = tensors["old_values"][indices]
            predicted_clipped = old_values + (
                values - old_values
            ).clamp(-args.value_clip, args.value_clip)
            value_loss_unclipped = (
                values - tensors["returns"][indices]
            ).pow(2)
            value_loss_clipped = (
                predicted_clipped - tensors["returns"][indices]
            ).pow(2)
            value_loss = 0.5 * torch.max(
                value_loss_unclipped, value_loss_clipped
            ).mean()
            entropy = distribution.entropy().mean()
            reference_kl = logits.new_zeros(())
            if reference_coef > 0:
                with torch.no_grad():
                    reference_logits = reference(inputs)
                reference_kl = masked_reference_kl(
                    logits, reference_logits, inputs["action_mask"],
                    args.policy_temperature,
                )
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + reference_coef * reference_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Actor and critic have disjoint parameters. A large critic error
            # must not rescale the actor's policy gradient to almost zero.
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor_critic.actor.parameters(), args.max_grad_norm,
                error_if_nonfinite=True,
            )
            critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor_critic.critic.parameters(), args.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1).abs() > args.clip_ratio
                ).float().mean()
            size = len(indices)
            processed += size
            epoch_samples += size
            epoch_kl += approximate_kl.item() * size
            totals["policy_loss"] += policy_loss.item() * size
            totals["value_loss"] += value_loss.item() * size
            totals["entropy"] += entropy.item() * size
            totals["approximate_kl"] += approximate_kl.item() * size
            totals["clip_fraction"] += clip_fraction.item() * size
            totals["reference_kl"] += reference_kl.item() * size
            totals["actor_gradient_norm"] += float(actor_gradient_norm) * size
            totals["critic_gradient_norm"] += float(critic_gradient_norm) * size
            totals["gradient_norm"] += math.hypot(
                float(actor_gradient_norm), float(critic_gradient_norm)
            ) * size

        epochs_completed += 1
        if epoch_samples and epoch_kl / epoch_samples > args.target_kl:
            break
    return {
        key: value / max(processed, 1)
        for key, value in totals.items()
    } | {
        "transitions": sample_count,
        "ppo_epochs_completed": epochs_completed,
        "critic_warmup": warmup,
    }


@torch.inference_mode()
def evaluate_against_baseline(
    actor: nn.Module,
    baseline: nn.Module,
    args,
    device: torch.device,
    env_pool: Optional[ParallelGamePool] = None,
):
    policies = [
        BattlePolicy(
            "ppo_current", None, actor.eval(), args.observation_key
        ),
        BattlePolicy(
            "baseline", args.base_model.resolve(), baseline.eval(),
            args.observation_key,
        ),
    ]
    tasks = []
    for group in range(args.evaluation_groups):
        seed = args.seed + 900_000_000 + group
        lineup = (0, 1, 1)
        for rotation in range(3):
            tasks.append(
                MatchTask(
                    seed=seed,
                    rotation=rotation,
                    seat_models=tuple(
                        lineup[(seat + rotation) % 3]
                        for seat in range(3)
                    ),
                )
            )
    battle_args = SimpleNamespace(
        action_mode="greedy",
        temperature=1.0,
        max_steps=args.max_steps,
        game_mode=args.game_mode,
        rounds_per_wind=args.rounds_per_wind,
    )
    results = run_tasks(
        tasks,
        policies,
        battle_args,
        device,
        np.random.default_rng(args.seed),
        env_pool=env_pool,
    )
    groups, errors = combine_rotation_results(results, (0, 1, 1))
    total = defaultdict(float)
    for group in groups:
        for key, value in group["models"]["0"].items():
            total[key] += value
    hands = int(total["hands"])
    played_hands = int(total["played_hands"]) or hands
    return {
        "groups": len(groups),
        "hands": hands,
        "average_points": total["point_sum"] / max(hands, 1),
        "average_rank": total["rank_sum"] / max(hands, 1),
        "first_rate": total["firsts"] / max(hands, 1),
        "third_rate": total["thirds"] / max(hands, 1),
        "win_rate": total["wins"] / max(played_hands, 1),
        "ron_rate": total["ron_wins"] / max(played_hands, 1),
        "tsumo_rate": total["tsumo_wins"] / max(played_hands, 1),
        "deal_in_rate": total["deal_ins"] / max(played_hands, 1),
        "draw_rate": total["draws"] / max(played_hands, 1),
        "errors": len(errors),
    }


def serializable_args(args) -> Dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_torch_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_history(path: Path, history: Sequence[Dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def checkpoint_payload(
    *,
    actor_critic: ActorCritic,
    optimizer,
    snapshot,
    model_name: str,
    args,
    update: int,
    total_hands: int,
    next_episode_id: int,
    best_evaluation_points: float,
    history: Sequence[Dict],
    rng: np.random.Generator,
    baseline: Optional[nn.Module] = None,
):
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "training_method": "ppo",
        "model_name": model_name,
        "obs_channels": args.obs_channels,
        "observation_key": args.observation_key,
        # load_checkpoint_model 和实战比较器只需要以下策略字段。
        "model_state_dict": actor_critic.actor.state_dict(),
        "ppo_value_state_dict": actor_critic.critic.state_dict(),
        "ppo_optimizer_state_dict": optimizer.state_dict(),
        "ppo_snapshot_state_dict": snapshot.state_dict(),
        "ppo_update": update,
        "ppo_total_hands": total_hands,
        "ppo_next_episode_id": next_episode_id,
        "ppo_best_evaluation_points": best_evaluation_points,
        "ppo_best_evaluation_score": best_evaluation_points,
        "ppo_best_metric": (
            "-average_rank + average_points/1e6"
            if args.game_mode == "south"
            else "average_points"
        ),
        "ppo_history": list(history),
        "ppo_args": serializable_args(args),
        "numpy_rng_state": rng.bit_generator.state,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if isinstance(snapshot, OpponentPool):
        payload["ppo_opponent_pool_states"] = snapshot.checkpoint_states()
    if baseline is not None:
        payload["ppo_baseline_state_dict"] = baseline.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def policy_snapshot_payload(
    *, actor_critic: ActorCritic, model_name: str, args, update: int,
    total_hands: int, elapsed_seconds: float = 0.0,
) -> Dict:
    """供独立训练曲线测评使用的轻量策略快照，不包含优化器和历史。"""
    return {
        "format_version": CHECKPOINT_VERSION,
        "training_method": "ppo_policy_snapshot",
        "model_name": model_name,
        "obs_channels": args.obs_channels,
        "observation_key": args.observation_key,
        "model_state_dict": actor_critic.actor.state_dict(),
        "ppo_update": int(update),
        "ppo_total_hands": int(total_hands),
        "ppo_elapsed_seconds": float(elapsed_seconds),
        "source_run": str(args.output_dir.resolve()),
    }


def restore_rng(payload: Dict, rng: np.random.Generator) -> None:
    if "numpy_rng_state" in payload:
        rng.bit_generator.state = payload["numpy_rng_state"]
    if "python_rng_state" in payload:
        random.setstate(payload["python_rng_state"])
    if "torch_rng_state" in payload:
        # 断点使用 map_location=cuda 加载时，CPU RNG 状态也会
        # 被搬到 CUDA；torch.set_rng_state 只接受 CPU ByteTensor。
        torch_state = payload["torch_rng_state"].detach().to(
            device="cpu", dtype=torch.uint8
        )
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in payload["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def configure_learning_rates(optimizer, args, update: int) -> None:
    fraction = max(0.1, 1.0 - (update - 1) / max(args.updates, 1))
    optimizer.param_groups[0]["lr"] = args.actor_lr * fraction
    optimizer.param_groups[1]["lr"] = args.critic_lr * fraction


def evaluation_objective(evaluation: Dict, args) -> float:
    if args.game_mode == "south":
        return (
            -float(evaluation["average_rank"])
            + float(evaluation["average_points"]) / 1_000_000
        )
    return float(evaluation["average_points"])


def validate_resume_settings(payload, args):
    """A resumed run keeps its objective and population, even on a new host."""
    previous = payload.get("ppo_args", {})
    # Legacy checkpoints retain the existing permissive continuation behavior.
    if "ppo_opponent_pool_states" not in payload:
        return
    immutable = ("seed", "game_mode", "rounds_per_wind", "gamma", "gae_lambda",
                 "reward_scale", "reward_clip", "rank_reward_weight",
                 "policy_temperature", "reference_kl_coef", "critic_warmup_updates",
                 "opponent_pool_size", "snapshot_ratio", "snapshot_interval")
    changed = [name for name in immutable
               if name in previous and previous[name] != getattr(args, name)]
    if changed:
        raise ValueError("续训不能改变以下实验设置：" + ", ".join(changed)
                         + "；请使用新output-dir和base-model建立实验")


def main():
    args = parse_args()
    if args.game_mode == "south" and args.max_steps == 300:
        args.max_steps = 3000
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    ROLLOUT_INFERENCE.enabled = args.rollout_cuda_graphs
    if not args.base_model.exists():
        raise FileNotFoundError(f"基础模型不存在：{args.base_model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    actor, metadata = load_checkpoint_model(
        args.base_model, map_location=device
    )
    model_name = str(metadata.get("model_name", "baseline_cnn"))
    args.obs_channels = int(metadata.get("obs_channels", 6))
    args.observation_key = str(
        metadata.get("observation_key", "observation")
    )
    actor_critic = ActorCritic(
        actor.to(device), obs_channels=args.obs_channels
    ).to(device)
    if "ppo_value_state_dict" in metadata:
        actor_critic.critic.load_state_dict(metadata["ppo_value_state_dict"])
        print("已从基础 PPO checkpoint 继承价值网络")
    baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
    baseline = baseline.to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    snapshot = OpponentPool(actor_critic.actor, args.opponent_pool_size)
    for path in args.opponent_models:
        opponent, opponent_metadata = load_checkpoint_model(path, map_location=device)
        if (opponent_metadata.get("model_name", "baseline_cnn") != model_name
                or int(opponent_metadata.get("obs_channels", 6)) != args.obs_channels
                or opponent_metadata.get("observation_key", "observation") != args.observation_key):
            raise ValueError(f"对手模型结构或观测不匹配：{path}")
        snapshot.add(opponent.to(device))
        del opponent

    optimizer = torch.optim.AdamW(
        [
            {
                "params": actor_critic.actor.parameters(),
                "lr": args.actor_lr,
                "weight_decay": 1e-5,
            },
            {
                "params": actor_critic.critic.parameters(),
                "lr": args.critic_lr,
                "weight_decay": 1e-4,
            },
        ]
    )
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    history_path = args.output_dir / "history.json"
    checkpoints_dir = args.output_dir / "checkpoints"
    start_update = 1
    total_hands = 0
    next_episode_id = 0
    best_evaluation_points = -math.inf
    history: List[Dict] = []

    if last_path.exists() and not args.fresh:
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload.get("training_method") != "ppo":
            raise ValueError(f"{last_path} 不是 PPO 训练断点")
        if payload.get("model_name") != model_name:
            raise ValueError("断点模型结构与 --base-model 不一致")
        validate_resume_settings(payload, args)
        actor_critic.actor.load_state_dict(payload["model_state_dict"])
        actor_critic.critic.load_state_dict(payload["ppo_value_state_dict"])
        optimizer.load_state_dict(payload["ppo_optimizer_state_dict"])
        snapshot.restore(payload.get(
            "ppo_opponent_pool_states", [payload["ppo_snapshot_state_dict"]]
        ))
        if "ppo_baseline_state_dict" in payload:
            baseline.load_state_dict(payload["ppo_baseline_state_dict"])
        start_update = int(payload["ppo_update"]) + 1
        total_hands = int(payload["ppo_total_hands"])
        next_episode_id = int(payload["ppo_next_episode_id"])
        best_evaluation_points = float(
            payload.get(
                "ppo_best_evaluation_score",
                payload.get("ppo_best_evaluation_points", -math.inf),
            )
        )
        history = list(payload.get("ppo_history", []))
        restore_rng(payload, rng)
        print(f"从断点继续：{last_path}，下一次更新 {start_update}")

    previous_elapsed_seconds = sum(
        float(record.get("elapsed_seconds", 0.0)) for record in history
    )
    evaluation_capacity = (
        args.evaluation_groups * 3 if args.evaluation_interval > 0 else 0
    )
    pool_capacity = max(args.hands_per_update, evaluation_capacity)
    worker_count = resolve_env_workers(args.env_workers, pool_capacity)
    args.resolved_env_workers = worker_count
    (args.output_dir / "config.json").write_text(
        json.dumps(serializable_args(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"设备={device} | 策略={model_name} | 每次更新={args.hands_per_update}局 | "
        f"总更新={args.updates} | 基础模型={args.base_model}"
    )
    started = time.perf_counter()

    if start_update > 1 and (
        args.checkpoint_interval > 0 or args.checkpoint_minutes > 0
    ):
        completed_update = start_update - 1
        current_snapshot_path = (
            checkpoints_dir / f"update_{completed_update:06d}.pt"
        )
        if not current_snapshot_path.exists():
            save_torch_atomic(
                current_snapshot_path,
                policy_snapshot_payload(
                    actor_critic=actor_critic,
                    model_name=model_name,
                    args=args,
                    update=completed_update,
                    total_hands=total_hands,
                    elapsed_seconds=previous_elapsed_seconds,
                ),
            )
            print(f"已保留训练时间点：{current_snapshot_path}")

    # 当前策略和 baseline 完全相同；在三个座位轮换后，它们的
    # 期望点差基准严格为 0。先保存这个起点，可以保证 best.pt
    # 只有在 PPO 固定评估取得正点差时才会被替换。
    if start_update == 1:
        best_evaluation_points = -2.0 if args.game_mode == "south" else 0.0
        initial_payload = checkpoint_payload(
            actor_critic=actor_critic,
            optimizer=optimizer,
            snapshot=snapshot,
            model_name=model_name,
            args=args,
            update=0,
            total_hands=0,
            next_episode_id=0,
            best_evaluation_points=best_evaluation_points,
            history=history,
            rng=rng,
            baseline=baseline,
        )
        save_torch_atomic(best_path, initial_payload)
        save_torch_atomic(last_path, initial_payload)
        if args.checkpoint_interval > 0 or args.checkpoint_minutes > 0:
            save_torch_atomic(
                checkpoints_dir / "update_000000.pt",
                policy_snapshot_payload(
                    actor_critic=actor_critic,
                    model_name=model_name,
                    args=args,
                    update=0,
                    total_hands=0,
                    elapsed_seconds=0.0,
                ),
            )
        if args.game_mode == "south":
            print("固定评估基准：平均顺位 2.000（PPO 与 baseline 同权重）")
        else:
            print("固定评估基准：+0.0 点/局（PPO 与 baseline 同权重）")

    env_pool = None
    if worker_count > 1:
        env_pool = ParallelGamePool(
            capacity=pool_capacity,
            workers=worker_count,
            max_steps=args.max_steps,
            fast_forward_forced=args.fast_forward_forced,
            async_min_workers=args.async_min_workers,
        )
        atexit.register(env_pool.close)
        print(
            f"环境池={worker_count} 个持久化进程 | "
            "环境并行、GPU 集中批量推理"
        )
    else:
        print("环境池=单进程兼容模式")

    time_checkpoint_seconds = args.checkpoint_minutes * 60
    next_time_checkpoint = (
        (math.floor(previous_elapsed_seconds / time_checkpoint_seconds) + 1)
        * time_checkpoint_seconds
        if time_checkpoint_seconds > 0
        else math.inf
    )
    stopped_by_time = False
    final_update = start_update - 1
    time_budget_already_complete = (
        args.max_hours > 0
        and previous_elapsed_seconds >= args.max_hours * 3600
    )
    if time_budget_already_complete:
        stopped_by_time = True
        print("累计训练时长已经达到上限，跳过新的 PPO 更新")
    update_range = (
        range(0)
        if time_budget_already_complete
        else range(start_update, args.updates + 1)
    )
    for update in update_range:
        update_started = time.perf_counter()
        configure_learning_rates(optimizer, args, update)
        completed, failures = collect_rollout(
            actor_critic,
            baseline,
            snapshot,
            args,
            device,
            rng,
            next_episode_id,
            env_pool=env_pool,
        )
        next_episode_id += args.hands_per_update
        if len(failures) / args.hands_per_update > args.max_failure_rate:
            raise RuntimeError(f"模拟失败率超限：{len(failures)}/{args.hands_per_update}；{failures[:3]}")
        if any(result["invalid"] for result in completed):
            raise RuntimeError("模拟出现非法动作，已停止更新；请检查规则和动作掩码")
        total_hands += len(completed)
        batch = build_ppo_batch(completed, args)
        losses = ppo_update(
            actor_critic, optimizer, batch, args, device,
            reference=baseline, update=update,
        )

        points = [result["points"] for result in completed]
        rollout = {
            "hands": len(completed),
            "failed_hands": len(failures),
            "failure_details": failures,
            "average_points": float(np.mean(points)) if points else 0.0,
            "win_rate": sum(result["win"] for result in completed)
            / max(sum(result["played_hands"] for result in completed), 1),
            "ron_rate": sum(result["ron_win"] for result in completed)
            / max(sum(result["played_hands"] for result in completed), 1),
            "tsumo_rate": sum(result["tsumo_win"] for result in completed)
            / max(sum(result["played_hands"] for result in completed), 1),
            "first_rate": (
                sum(result["first"] for result in completed)
                / max(len(completed), 1)
                if args.game_mode == "south" and completed
                else None
            ),
            "deal_in_rate": sum(result["deal_in"] for result in completed)
            / max(sum(result["played_hands"] for result in completed), 1),
            "draw_rate": sum(result["draw"] for result in completed)
            / max(sum(result["played_hands"] for result in completed), 1),
            "average_rank": (
                float(np.mean([
                    result["rank"] for result in completed
                    if result["rank"] is not None
                ]))
                if args.game_mode == "south" and completed
                else None
            ),
        }
        evaluation = None
        is_evaluation_update = (
            args.evaluation_interval > 0
            and (
                update == 1
                or update % args.evaluation_interval == 0
                or update == args.updates
            )
        )
        if is_evaluation_update:
            evaluation = evaluate_against_baseline(
                actor_critic.actor,
                baseline,
                args,
                device,
                env_pool=env_pool,
            )

        if update % args.snapshot_interval == 0:
            snapshot.add(actor_critic.actor)

        record = {
            "update": update,
            "total_hands": total_hands,
            "rollout": rollout,
            "ppo": losses,
            "evaluation": evaluation,
            "actor_learning_rate": optimizer.param_groups[0]["lr"],
            "critic_learning_rate": optimizer.param_groups[1]["lr"],
            "opponent_pool_size": len(snapshot.models),
            "elapsed_seconds": time.perf_counter() - update_started,
        }
        history.append(record)
        improved = (
            evaluation is not None
            and evaluation["errors"] == 0
            and evaluation["groups"] == args.evaluation_groups
            and evaluation_objective(evaluation, args) > best_evaluation_points
        )
        if improved:
            best_evaluation_points = evaluation_objective(evaluation, args)

        payload = checkpoint_payload(
            actor_critic=actor_critic,
            optimizer=optimizer,
            snapshot=snapshot,
            model_name=model_name,
            args=args,
            update=update,
            total_hands=total_hands,
            next_episode_id=next_episode_id,
            best_evaluation_points=best_evaluation_points,
            history=history,
            rng=rng,
            baseline=baseline,
        )
        save_torch_atomic(last_path, payload)
        if improved:
            save_torch_atomic(best_path, payload)
        if args.checkpoint_interval > 0 and (
            update % args.checkpoint_interval == 0 or update == args.updates
        ):
            save_torch_atomic(
                checkpoints_dir / f"update_{update:06d}.pt",
                policy_snapshot_payload(
                    actor_critic=actor_critic,
                    model_name=model_name,
                    args=args,
                    update=update,
                    total_hands=total_hands,
                    elapsed_seconds=(
                        previous_elapsed_seconds
                        + time.perf_counter() - started
                    ),
                ),
            )
        cumulative_elapsed_seconds = (
            previous_elapsed_seconds + time.perf_counter() - started
        )
        while cumulative_elapsed_seconds >= next_time_checkpoint:
            elapsed_minutes = int(round(next_time_checkpoint / 60))
            time_snapshot_path = checkpoints_dir / (
                f"elapsed_{elapsed_minutes:04d}min_update_{update:06d}.pt"
            )
            save_torch_atomic(
                time_snapshot_path,
                policy_snapshot_payload(
                    actor_critic=actor_critic,
                    model_name=model_name,
                    args=args,
                    update=update,
                    total_hands=total_hands,
                    elapsed_seconds=cumulative_elapsed_seconds,
                ),
            )
            print(f"已保留训练时间点：{time_snapshot_path}")
            next_time_checkpoint += time_checkpoint_seconds
        save_history(history_path, history)

        if evaluation is None:
            evaluation_text = ""
        elif args.game_mode == "south":
            evaluation_text = (
                f" | 固定评估 顺位{evaluation['average_rank']:.3f} / "
                f"{evaluation['average_points']:+.1f}点/场"
            )
        else:
            evaluation_text = (
                f" | 固定评估 {evaluation['average_points']:+.1f}点/局"
            )
        completed_updates = update - start_update + 1
        mean_seconds = (
            time.perf_counter() - started
        ) / max(completed_updates, 1)
        if args.max_hours > 0:
            eta_hours = max(
                0.0,
                args.max_hours - cumulative_elapsed_seconds / 3600,
            )
        else:
            eta_hours = mean_seconds * (args.updates - update) / 3600
        print(
            f"更新 {update:04d}/{args.updates} | 总局数 {total_hands:,} | "
            f"样本 {losses['transitions']:,} | "
            f"采集 {rollout['average_points']:+.1f}点/局 | "
            f"KL {losses['approximate_kl']:.5f} | "
            f"熵 {losses['entropy']:.3f}{evaluation_text} | "
            f"{record['elapsed_seconds']:.1f}s | 预计剩余 {eta_hours:.2f}小时"
        )
        final_update = update
        if (
            args.max_hours > 0
            and cumulative_elapsed_seconds >= args.max_hours * 3600
        ):
            stopped_by_time = True
            break

    total_elapsed_seconds = (
        previous_elapsed_seconds + time.perf_counter() - started
    )
    if args.checkpoint_minutes > 0 and final_update >= 0:
        final_snapshot_path = checkpoints_dir / (
            f"final_{total_elapsed_seconds / 3600:.2f}h_"
            f"update_{final_update:06d}.pt"
        )
        save_torch_atomic(
            final_snapshot_path,
            policy_snapshot_payload(
                actor_critic=actor_critic,
                model_name=model_name,
                args=args,
                update=final_update,
                total_hands=total_hands,
                elapsed_seconds=total_elapsed_seconds,
            ),
        )
        print(f"已保留最终训练时间点：{final_snapshot_path}")
    if stopped_by_time:
        print(f"已达到累计训练时长上限 {args.max_hours:.2f} 小时")

    if env_pool is not None:
        env_pool.close()
        atexit.unregister(env_pool.close)

    print(
        f"\n训练完成，累计用时 {total_elapsed_seconds / 3600:.2f} 小时。"
        f"\n最佳策略：{best_path}"
        f"\n最新断点：{last_path}"
        f"\n训练曲线：{history_path}"
    )
    if args.evaluation_interval == 0:
        print("固定评估已关闭；best.pt 保持训练起点，请从 checkpoints 循环赛选优。")


if __name__ == "__main__":
    main()
