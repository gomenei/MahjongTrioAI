"""Check critic signal and stored behavior probabilities without training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.distributions import Categorical

import ppo_train as ppo
from model import load_checkpoint_model
from training.parallel_env import ParallelGamePool
from training.ppo_support import OpponentPool


def value_metrics(target, prediction):
    variance = float(np.var(target))
    error = target - prediction
    return {"samples": len(target), "target_mean": float(target.mean()),
            "prediction_mean": float(prediction.mean()), "target_variance": variance,
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "explained_variance": 1 - float(np.var(error)) / variance if variance > 1e-12 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grouped-walls", action="store_true")
    parser.add_argument("--temperature", type=float)
    options = parser.parse_args()
    args = SimpleNamespace(**json.loads(options.config.read_text()))
    if options.temperature is not None:
        if options.temperature <= 0:
            parser.error("temperature must be positive")
        args.policy_temperature = options.temperature
    args.hands_per_update, args.seed = 512, 291_000_000
    if options.grouped_walls:
        args.hands_per_update, args.seed, args.snapshot_ratio = 480, 301_000_000, 0.0
        original_spec = ppo.GameSpec

        def grouped_spec(**kwargs):
            episode_id = kwargs["seed"] - args.seed
            # Four repetitions share wall, learner seat and greedy baseline opponents.
            kwargs["seed"] = args.seed + (episode_id // 12) * 3 + episode_id % 3
            return original_spec(**kwargs)

        ppo.GameSpec = grouped_spec
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    actor, payload = load_checkpoint_model(options.checkpoint, map_location=device)
    ac = ppo.ActorCritic(actor, obs_channels=args.obs_channels).to(device).eval()
    ac.critic.load_state_dict(payload["ppo_value_state_dict"])
    baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
    baseline = baseline.to(device).eval()
    baseline.load_state_dict(payload["ppo_baseline_state_dict"])
    pool = OpponentPool(ac.actor, args.opponent_pool_size)
    pool.restore(payload["ppo_opponent_pool_states"])
    ppo.ROLLOUT_INFERENCE.enabled = True
    with ParallelGamePool(args.hands_per_update, 20, args.max_steps, fast_forward_forced=True,
                          async_min_workers=1) as env:
        completed, failures = ppo.collect_rollout(
            ac, baseline, pool, args, device, np.random.default_rng(args.seed), 0, env_pool=env)
    if failures or len(completed) != args.hands_per_update or any(x["invalid"] for x in completed):
        raise RuntimeError(f"Invalid diagnostic trajectories: {failures}")
    batch = ppo.build_ppo_batch(completed, args)
    target, prediction = batch["returns"], batch["old_values"]
    recomputed = []
    with torch.inference_mode():
        for start in range(0, len(target), 512):
            stop = start + 512
            inputs = {key: torch.as_tensor(batch[key][start:stop], device=device)
                      for key in ("observation", "action_mask")}
            logits = ac.actor(inputs)
            actions = torch.as_tensor(batch["actions"][start:stop], device=device)
            recomputed.extend(Categorical(logits=logits / args.policy_temperature).log_prob(actions).cpu().tolist())
    discrepancy = np.abs(np.asarray(recomputed) - batch["old_log_probabilities"])
    progress = np.concatenate([np.arange(len(x["episode"].trajectory)) /
                               len(x["episode"].trajectory) for x in completed])
    phases = {}
    for index in range(4):
        selected = (progress >= index / 4) & (progress < (index + 1) / 4)
        phases[str(index)] = value_metrics(target[selected], prediction[selected])
    report = {"purpose": "Learning-signal diagnostics only, not a strength evaluation",
              "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
              "seed": args.seed, "policy_temperature": args.policy_temperature,
              "matches": len(completed), "failures": failures,
              "critic": value_metrics(target, prediction), "critic_by_trajectory_quarter": phases,
              "behavior_log_prob_absolute_error_max": float(discrepancy.max()),
              "behavior_log_prob_absolute_error_mean": float(discrepancy.mean()),
              "rollout_average_rank": float(np.mean([x["rank"] for x in completed])),
              "rollout_average_points": float(np.mean([x["points"] for x in completed]))}
    if options.grouped_walls:
        rewards = np.asarray([x["reward"] for x in completed])
        groups = np.asarray([[rewards[group * 12 + seat + repeat * 3] for repeat in range(4)]
                             for group in range(40) for seat in range(3)])
        centered = groups - (groups.sum(axis=1, keepdims=True) - groups) / 3
        report["paired_wall_diagnostic"] = {
            "groups": len(groups), "repetitions": 4, "opponents": "greedy frozen baseline",
            "raw_reward_variance": float(rewards.var()),
            "leave_one_out_reward_variance": float(centered.var()),
            "variance_ratio": float(centered.var() / rewards.var()),
            "caution": "Variance only; no policy was updated and no strength claim is justified",
        }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
