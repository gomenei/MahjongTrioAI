"""Compare policies on fixed baseline trajectories, separate from strength tests."""

import argparse
import copy
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
from training.ppo_support import masked_reference_kl


@torch.inference_mode()
def greedy_learner(model, observations, device, temperature, observation_key):
    inputs = ppo.to_model_input(observations, device, observation_key)
    logits, values = ppo.ROLLOUT_INFERENCE(model, inputs)
    actions = logits.argmax(-1)
    log_probs = Categorical(logits=logits / temperature).log_prob(actions)
    return actions.cpu().tolist(), log_probs.cpu().tolist(), values.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    args = SimpleNamespace(**json.loads(options.config.read_text()))
    args.hands_per_update, args.seed, args.snapshot_ratio = 128, 271_000_000, 0.0
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
    baseline = baseline.to(device).eval()
    ac = ppo.ActorCritic(copy.deepcopy(baseline), obs_channels=args.obs_channels).to(device).eval()
    ppo.ROLLOUT_INFERENCE.enabled = True
    ppo.learner_actions = greedy_learner
    with ParallelGamePool(128, 20, args.max_steps, fast_forward_forced=True,
                          async_min_workers=1) as env:
        completed, failures = ppo.collect_rollout(
            ac, baseline, baseline, args, device, np.random.default_rng(args.seed), 0, env_pool=env)
    if failures or len(completed) != 128 or any(x["invalid"] for x in completed):
        raise RuntimeError(f"Invalid diagnostic trajectories: {failures}")
    transitions = [t for result in completed for t in result["episode"].trajectory]
    results = []
    for item in options.candidates:
        name, path = item.split("=", 1)
        candidate, _ = load_checkpoint_model(path, map_location=device)
        candidate = candidate.to(device).eval()
        totals = {"action_change_fraction": 0.0, "reference_kl": 0.0,
                  "baseline_entropy": 0.0, "candidate_entropy": 0.0}
        with torch.inference_mode():
            for start in range(0, len(transitions), 512):
                chunk = transitions[start:start + 512]
                inputs = {"observation": torch.as_tensor(np.asarray([t.observation for t in chunk]), device=device),
                          "action_mask": torch.as_tensor(np.asarray([t.action_mask for t in chunk]), device=device)}
                ref_logits, logits = baseline(inputs), candidate(inputs)
                totals["action_change_fraction"] += (ref_logits.argmax(-1) != logits.argmax(-1)).sum().item()
                totals["reference_kl"] += masked_reference_kl(logits, ref_logits, inputs["action_mask"]).item() * len(chunk)
                totals["baseline_entropy"] += Categorical(logits=ref_logits).entropy().sum().item()
                totals["candidate_entropy"] += Categorical(logits=logits).entropy().sum().item()
        results.append({"name": name, "checkpoint": path,
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        **{key: value / len(transitions) for key, value in totals.items()}})
        del candidate
    report = {"purpose": "Policy-change diagnostics only; not evidence of playing strength",
              "seed": args.seed, "matches": len(completed), "decision_states": len(transitions),
              "failures": failures, "results": results}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
