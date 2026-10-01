"""Measure actual rollout bottlenecks without changing optimizer/model weights."""

import argparse
import cProfile
import copy
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import ppo_train as ppo
import training.parallel_env as parallel
from model import load_checkpoint_model
from training.ppo_support import OpponentPool


def cpu_usage():
    path = Path("/sys/fs/cgroup/cpu.stat")
    return int(dict(line.split() for line in path.read_text().splitlines())["usage_usec"]) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ppo_runs/league_v2_20260916/config.json")
    parser.add_argument("--hands", type=int, default=128)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fast-forward-forced", action="store_true")
    parser.add_argument("--greedy", action="store_true", help="Use identical greedy decisions for timing comparisons")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--async-min-workers", type=int, default=0)
    options = parser.parse_args()
    args = SimpleNamespace(**json.loads((ROOT / options.config).read_text()))
    args.hands_per_update = options.hands
    args.seed = 371_000_000
    torch.set_num_threads(options.threads)
    ppo.ROLLOUT_INFERENCE.enabled = options.cuda_graphs
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    actor, metadata = load_checkpoint_model(ROOT / args.base_model, map_location=device)
    ac = ppo.ActorCritic(actor.to(device), obs_channels=args.obs_channels).to(device).eval()
    baseline = copy.deepcopy(actor).eval().requires_grad_(False)
    pool = OpponentPool(actor, args.opponent_pool_size)
    for path in args.opponent_models:
        model, _ = load_checkpoint_model(ROOT / path, map_location=device)
        pool.add(model.to(device))
    counts = {"learner_seconds": 0.0, "opponent_seconds": 0.0, "env_seconds": 0.0,
              "env_calls": 0, "slot_steps": 0, "all_forced_slot_steps": 0,
              "learner_batches": [], "opponent_batches": [], "active_slots": []}
    if options.greedy:
        @torch.inference_mode()
        def greedy_learner(model, observations, device, temperature, observation_key):
            inputs = ppo.to_model_input(observations, device, observation_key)
            logits, values = ppo.ROLLOUT_INFERENCE(model, inputs)
            distribution = torch.distributions.Categorical(logits=logits / temperature)
            actions = logits.argmax(-1)
            return actions.cpu().tolist(), distribution.log_prob(actions).cpu().tolist(), values.cpu().tolist()
        ppo.learner_actions = greedy_learner
    def timed_action(name, original, observation_index):
        def call(*pos, **kw):
            observations = pos[observation_index]
            if observations:
                counts[name + "_batches"].append(len(observations))
            start = time.perf_counter()
            result = original(*pos, **kw)
            counts[name + "_seconds"] += time.perf_counter() - start
            return result
        return call
    ppo.learner_actions = timed_action("learner", ppo.learner_actions, 1)
    ppo.opponent_actions = timed_action("opponent", ppo.opponent_actions, 1)
    profiler = cProfile.Profile()
    with parallel.ParallelGamePool(options.hands, options.workers, args.max_steps,
                                   fast_forward_forced=options.fast_forward_forced,
                                   async_min_workers=options.async_min_workers) as env:
        original_step = env.step
        def timed_step(actions):
            counts["env_calls"] += 1
            counts["slot_steps"] += len(actions)
            counts["active_slots"].append(len(actions))
            for slot in actions:
                if all(np.count_nonzero(ob["action_mask"]) <= 1
                       for ob in env.observations(slot).values()):
                    counts["all_forced_slot_steps"] += 1
            start = time.perf_counter()
            result = original_step(actions)
            counts["env_seconds"] += time.perf_counter() - start
            return result
        env.step = timed_step
        started, cpu_started = time.perf_counter(), cpu_usage()
        if options.profile:
            profiler.enable()
        completed, failures = ppo.collect_rollout(ac, baseline, pool, args, device,
                                                 np.random.default_rng(args.seed), 0, env_pool=env)
        if options.profile:
            profiler.disable()
        elapsed = time.perf_counter() - started
        cpu = cpu_usage() - cpu_started
    result = {"hands": options.hands, "workers": options.workers, "threads": options.threads,
              "fast_forward_forced": options.fast_forward_forced,
              "greedy": options.greedy,
              "cuda_graphs": options.cuda_graphs,
              "async_min_workers": options.async_min_workers,
              "elapsed_seconds": elapsed, "matches_per_second": len(completed) / elapsed,
              "used_cpu_cores": cpu / elapsed, "failures": failures,
              "transitions": sum(len(row["episode"].trajectory) for row in completed)}
    result["points_by_episode"] = [row["points"] for row in completed]
    digest = hashlib.sha256()
    for row in completed:
        episode = row["episode"]
        digest.update(str(episode.episode_id).encode())
        for transition in episode.trajectory:
            digest.update(transition.observation.tobytes())
            digest.update(transition.action_mask.tobytes())
            digest.update(str(transition.action).encode())
        digest.update(str(row["points"]).encode())
    result["trajectory_sha256"] = digest.hexdigest()
    for name in ("learner_batches", "opponent_batches", "active_slots"):
        values = counts[name]
        counts[name] = {"count": len(values), "mean": float(np.mean(values)),
                        "median": float(np.median(values)), "max": max(values)} if values else {}
    result.update(counts)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2))
    if options.profile:
        with options.output.with_suffix(".profile.txt").open("w") as handle:
            pstats.Stats(profiler, stream=handle).sort_stats("cumulative").print_stats(35)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
