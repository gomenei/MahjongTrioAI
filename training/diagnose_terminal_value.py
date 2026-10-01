"""Audit a frozen critic against actual terminal rewards on fresh trajectories.

No optimizer runs, no candidate is selected, and no strength claim is made.
Lambda-return agreement alone can overstate value accuracy because the target
contains the same critic's bootstrapped predictions.
"""
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

import ppo_train as ppo
from model import load_checkpoint_model
from training.parallel_env import ParallelGamePool
from training.ppo_support import OpponentPool


def metrics(target, prediction):
    variance = float(np.var(target))
    error = target - prediction
    return dict(samples=len(target), target_mean=float(target.mean()),
                prediction_mean=float(prediction.mean()), target_variance=variance,
                rmse=float(np.sqrt(np.mean(error ** 2))),
                explained_variance=1-float(np.var(error))/variance if variance > 1e-12 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    args = SimpleNamespace(**json.loads(options.config.read_text(encoding='utf-8')))
    if args.gamma != 1:
        raise ValueError('This audit requires undiscounted terminal rewards')
    args.hands_per_update, args.seed = 512, options.seed
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    digest = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    actor, payload = load_checkpoint_model(options.checkpoint, map_location=device)
    ac = ppo.ActorCritic(actor, obs_channels=args.obs_channels).to(device).eval()
    ac.critic.load_state_dict(payload['ppo_value_state_dict'])
    before = {k:v.detach().cpu().clone() for k,v in ac.state_dict().items()}
    baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
    baseline = baseline.to(device).eval()
    baseline.load_state_dict(payload['ppo_baseline_state_dict'])
    pool = OpponentPool(ac.actor, args.opponent_pool_size)
    pool.restore(payload['ppo_opponent_pool_states'])
    ppo.ROLLOUT_INFERENCE.enabled = True
    with ParallelGamePool(512, 20, args.max_steps, fast_forward_forced=True,
                          async_min_workers=1) as env:
        completed, failures = ppo.collect_rollout(
            ac, baseline, pool, args, device, np.random.default_rng(args.seed), 0, env_pool=env)
    if failures or len(completed) != 512 or any(x['invalid'] for x in completed):
        raise RuntimeError(f'Invalid diagnostic trajectories: {failures}')
    batch = ppo.build_ppo_batch(completed, args)
    terminal = np.concatenate([np.full(len(x['episode'].trajectory), x['reward']) for x in completed])
    phase = np.concatenate([np.arange(len(x['episode'].trajectory))/len(x['episode'].trajectory)
                            for x in completed if x['episode'].trajectory])
    distance = np.concatenate([np.arange(len(x['episode'].trajectory)-1, -1, -1) for x in completed])
    prediction = batch['old_values']
    report = dict(purpose='Frozen critic audit only; not strength evidence', seed=args.seed,
                  checkpoint=str(options.checkpoint), checkpoint_sha256=digest,
                  matches=len(completed), failures=0, gae_lambda=args.gae_lambda,
                  terminal_reward_accuracy=metrics(terminal, prediction),
                  lambda_return_agreement=metrics(batch['returns'], prediction),
                  quarters={}, mean_decisions_per_match=float(len(terminal)/len(completed)))
    for i in range(4):
        selected = (phase >= i/4) & (phase < (i+1)/4)
        report['quarters'][str(i)] = dict(
            terminal_reward_accuracy=metrics(terminal[selected], prediction[selected]),
            lambda_return_agreement=metrics(batch['returns'][selected], prediction[selected]),
            mean_direct_terminal_weight=float(np.mean(args.gae_lambda**distance[selected])))
    if not all(torch.equal(v.detach().cpu(), before[k]) for k,v in ac.state_dict().items()):
        raise RuntimeError('Frozen model parameters changed')
    if hashlib.sha256(options.checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen checkpoint changed')
    report.update(completed=True, parameters_unchanged=True, checkpoint_unchanged=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
