"""Compare frozen-policy gradient consistency across fresh rollout batches.

This is a training-signal diagnostic, not a model-strength evaluation. No
optimizer runs and no checkpoint is changed. Lower GAE lambda adds bootstrap
bias; smaller variance alone is not grounds to claim a better training method.
"""

import argparse
import hashlib
import itertools
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


def actor_gradient(actor, batch, temperature, device):
    actor.zero_grad(set_to_none=True)
    count = len(batch["actions"])
    for start in range(0, count, 512):
        stop = start + 512
        inputs = {key: torch.as_tensor(batch[key][start:stop], device=device)
                  for key in ("observation", "action_mask")}
        actions = torch.as_tensor(batch["actions"][start:stop], device=device)
        advantages = torch.as_tensor(batch["advantages"][start:stop], device=device)
        distribution = Categorical(logits=actor(inputs) / temperature)
        loss = -(distribution.log_prob(actions) * advantages).sum() / count
        loss.backward()
    gradient = torch.cat([
        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).flatten()
        for parameter in actor.parameters()
    ]).detach().cpu()
    actor.zero_grad(set_to_none=True)
    if not torch.isfinite(gradient).all():
        raise RuntimeError("Nonfinite diagnostic gradient")
    return gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    args = SimpleNamespace(**json.loads(options.config.read_text(encoding="utf-8")))
    args.hands_per_update = 512
    if args.gamma != 1.0:
        raise ValueError("This diagnostic expects undiscounted match returns")
    torch.set_num_threads(4)
    torch.manual_seed(321_000_000)
    device = torch.device("cuda")
    actor, payload = load_checkpoint_model(options.checkpoint, map_location=device)
    ac = ppo.ActorCritic(actor, obs_channels=args.obs_channels).to(device).eval()
    ac.critic.load_state_dict(payload["ppo_value_state_dict"])
    initial_parameters = {name: value.detach().cpu().clone()
                          for name, value in ac.state_dict().items()}
    baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
    baseline = baseline.to(device).eval()
    baseline.load_state_dict(payload["ppo_baseline_state_dict"])
    opponents = OpponentPool(ac.actor, args.opponent_pool_size)
    opponents.restore(payload["ppo_opponent_pool_states"])
    ppo.ROLLOUT_INFERENCE.enabled = True
    lambdas = (1.0, 0.99, 0.95)
    gradients = {value: [] for value in lambdas}
    report = {
        "purpose": "Frozen-policy gradient consistency; not model-strength evidence",
        "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
        "policy_temperature": args.policy_temperature,
        "seeds": [321_000_000, 331_000_000, 341_000_000, 351_000_000],
        "lambdas": list(lambdas), "batches": [],
        "caution": "Smaller lambda adds bootstrap bias; variance or cosine is not a promotion criterion",
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with ParallelGamePool(512, 20, args.max_steps, fast_forward_forced=True,
                          async_min_workers=1) as env:
        for seed in report["seeds"]:
            args.seed = seed
            completed, failures = ppo.collect_rollout(
                ac, baseline, opponents, args, device, np.random.default_rng(seed), 0, env_pool=env)
            if failures or len(completed) != 512 or any(x["invalid"] for x in completed):
                raise RuntimeError(f"Invalid diagnostic rollout: {failures}")
            row = {"seed": seed, "matches": len(completed), "failures": 0, "estimators": {}}
            for value in lambdas:
                args.gae_lambda = value
                batch = ppo.build_ppo_batch(completed, args)
                raw_advantage = batch["returns"] - batch["old_values"]
                gradient = actor_gradient(ac.actor, batch, args.policy_temperature, device)
                gradients[value].append(gradient)
                row["estimators"][str(value)] = {
                    "states": len(raw_advantage), "raw_advantage_variance": float(raw_advantage.var()),
                    "gradient_norm": float(gradient.norm()),
                }
                del batch
            report["batches"].append(row)
            options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            del completed
    report["gradient_consistency"] = {}
    for value, vectors in gradients.items():
        cosines = [float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                   for a, b in itertools.combinations(vectors, 2)]
        matrix = torch.stack(vectors)
        report["gradient_consistency"][str(value)] = {
            "pairwise_cosines": cosines, "mean_cosine": float(np.mean(cosines)),
            "mean_gradient_norm": float(matrix.mean(dim=0).norm()),
            "mean_squared_gradient_deviation": float(((matrix - matrix.mean(dim=0)) ** 2).sum(dim=1).mean()),
        }
    assert all(torch.equal(value.detach().cpu(), initial_parameters[name])
               for name, value in ac.state_dict().items()), "Frozen parameters changed"
    report["parameters_unchanged"] = True
    report["completed"] = True
    options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["gradient_consistency"]), flush=True)


if __name__ == "__main__":
    main()
