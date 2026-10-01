"""Controlled critic-only warmup trial; frozen champion, no policy selection."""
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

import ppo_train as ppo
from model import load_checkpoint_model
from training.parallel_env import ParallelGamePool
from training.ppo_support import OpponentPool


def predictions(critic, observations, device):
    values = []
    with torch.inference_mode():
        for start in range(0, len(observations), 512):
            x = torch.as_tensor(observations[start:start+512], device=device)
            values.extend(critic(x).cpu().tolist())
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    out = options.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(**json.loads(options.config.read_text(encoding='utf-8')))
    args.hands_per_update = 512
    args.seed = 702_260_917
    args.critic_warmup_updates = 1000
    args.reference_kl_coef = 0
    torch.set_num_threads(4)
    torch.manual_seed(702_260_917)
    device = torch.device('cuda')
    actor, _ = load_checkpoint_model(args.base_model, map_location=device)
    actor = actor.to(device).eval().requires_grad_(False)
    frozen_actor = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    td = ppo.ActorCritic(actor, obs_channels=args.obs_channels).to(device).eval()
    mc = copy.deepcopy(td)
    base = copy.deepcopy(actor)
    pool = OpponentPool(actor, args.opponent_pool_size)
    for path in args.opponent_models:
        opponent, _ = load_checkpoint_model(path, map_location=device)
        pool.add(opponent.to(device))
    optimizers = [torch.optim.AdamW(m.critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
                  for m in (td, mc)]
    ppo.ROLLOUT_INFERENCE.enabled = True
    identity = dict(purpose='Critic-target diagnostic only; no policy is trained or promoted',
                    train_seed=args.seed, test_seed=742_260_917, updates=20,
                    train_matches=20*512, test_matches=4*512, epochs=args.ppo_epochs,
                    actor_sha256=hashlib.sha256(Path(args.base_model).read_bytes()).hexdigest(),
                    arms=['lambda_return_0.95', 'terminal_return'],
                    decision='Test match-cluster mean MSE reduction must have 95% lower bound > 0; early-quarter mean MSE must not increase.',
                    parameters_identical_at_start=True, independent_test=True)
    (out/'experiment.json').write_text(json.dumps(identity, indent=2))
    history = []
    with ParallelGamePool(512, 20, args.max_steps, fast_forward_forced=True,
                          async_min_workers=1) as env:
        for update in range(1, 21):
            completed, failures = ppo.collect_rollout(
                td, base, pool, args, device, np.random.default_rng(args.seed+update),
                (update-1)*512, env_pool=env)
            if failures or len(completed) != 512 or any(x['invalid'] for x in completed):
                raise RuntimeError(f'Invalid warmup trajectories: {failures}')
            batch = ppo.build_ppo_batch(completed, args)
            mc_batch = dict(batch, old_values=predictions(mc.critic, batch['observation'], device),
                            returns=np.concatenate([np.full(len(x['episode'].trajectory), x['reward'], dtype=np.float32)
                                                    for x in completed]))
            row = dict(update=update, matches=update*512, failed=0)
            for name, model, optimizer, data in zip(identity['arms'], (td, mc), optimizers, (batch, mc_batch)):
                torch.manual_seed(712_260_917+update)
                row[name] = ppo.ppo_update(model, optimizer, data, args, device, update=update)
            history.append(row)
            (out/'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps({'update':update, 'matches':update*512}), flush=True)
            del completed, batch, mc_batch
        errors = {name:[] for name in identity['arms']}
        early = {name:[] for name in identity['arms']}
        args.seed = 742_260_917
        for part in range(4):
            completed, failures = ppo.collect_rollout(
                td, base, pool, args, device, np.random.default_rng(args.seed+part),
                part*512, env_pool=env)
            if failures or len(completed) != 512 or any(x['invalid'] for x in completed):
                raise RuntimeError(f'Invalid independent trajectories: {failures}')
            batch = ppo.build_ppo_batch(completed, args)
            for name, model in zip(identity['arms'], (td, mc)):
                predicted = predictions(model.critic, batch['observation'], device)
                offset = 0
                for result in completed:
                    count = len(result['episode'].trajectory)
                    if count:
                        squared = (predicted[offset:offset+count]-result['reward'])**2
                        errors[name].append(float(squared.mean()))
                        early[name].append(float(squared[:max(1, (count+3)//4)].mean()))
                    offset += count
            del completed, batch
    difference = np.asarray(errors[identity['arms'][0]])-np.asarray(errors[identity['arms'][1]])
    gain = float(difference.mean())
    se = float(difference.std(ddof=1)/np.sqrt(len(difference)))
    early_gain = float(np.mean(early[identity['arms'][0]])-np.mean(early[identity['arms'][1]]))
    for model in (td, mc):
        if not all(torch.equal(v.detach().cpu(), frozen_actor[k]) for k,v in model.actor.state_dict().items()):
            raise RuntimeError('Frozen actor changed')
        if not all(torch.isfinite(v).all() for v in model.critic.state_dict().values()):
            raise RuntimeError('Nonfinite critic')
    report = dict(identity, completed=True, actor_unchanged=True, failures=0,
                  match_cluster_mse={name:float(np.mean(x)) for name,x in errors.items()},
                  early_quarter_mse={name:float(np.mean(x)) for name,x in early.items()},
                  paired_mse_reduction=gain, paired_ci95=[gain-1.95996398454*se, gain+1.95996398454*se],
                  matches_with_decisions=len(difference), supports_terminal_target=gain-1.95996398454*se>0 and early_gain>=0,
                  caution='Value accuracy evidence only; a fresh preregistered policy-training trial and full archive gate remain required.')
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
