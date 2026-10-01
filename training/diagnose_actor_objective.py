"""Measure frozen PPO objective gradients on fresh v6 trajectories.

This does not optimize parameters, select a checkpoint, or measure strength.
Gradients are before Adam preconditioning and clipping; cancellation is only
evidence for a controlled follow-up experiment, not proof of a better recipe.
"""
import argparse
from datetime import datetime, timezone
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
from training.ppo_support import OpponentPool, masked_reference_kl
from training.rules_candidate_runtime import install_training, rule_identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objective_gradients(actor, reference, batch, args, device):
    parameters = tuple(actor.parameters())
    count = len(batch['actions'])
    vectors = [torch.zeros(sum(p.numel() for p in parameters), device=device)
               for _ in range(3)]
    totals = dict(policy_loss=0., weighted_reference_loss=0.,
                  weighted_entropy_loss=0., max_log_probability_error=0.)
    for start in range(0, count, 512):
        chunk = slice(start, start + 512)
        inputs = {key: torch.as_tensor(batch[key][chunk], device=device)
                  for key in ('observation', 'action_mask')}
        actions = torch.as_tensor(batch['actions'][chunk], device=device)
        advantages = torch.as_tensor(batch['advantages'][chunk], device=device)
        old_log_probs = torch.as_tensor(batch['old_log_probabilities'][chunk], device=device)
        logits = actor(inputs)
        with torch.no_grad():
            reference_logits = reference(inputs)
        distribution = Categorical(logits=logits / args.policy_temperature)
        log_probs = distribution.log_prob(actions)
        ratio = (log_probs - old_log_probs).exp()
        policy_loss = -torch.min(ratio * advantages,
                                ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio)
                                * advantages).mean()
        reference_loss = args.reference_kl_coef * masked_reference_kl(
            logits, reference_logits, inputs['action_mask'], args.policy_temperature)
        entropy_loss = -args.entropy_coef * distribution.entropy().mean()
        weight = len(actions) / count
        for i, loss in enumerate((policy_loss, reference_loss, entropy_loss)):
            gradients = torch.autograd.grad(loss * weight, parameters,
                                            retain_graph=i < 2, allow_unused=True)
            vectors[i].add_(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                      for p, g in zip(parameters, gradients)]))
        for key, loss in zip(('policy_loss', 'weighted_reference_loss',
                              'weighted_entropy_loss'),
                             (policy_loss, reference_loss, entropy_loss)):
            totals[key] += float(loss.detach()) * weight
        totals['max_log_probability_error'] = max(
            totals['max_log_probability_error'], float((log_probs - old_log_probs).abs().max()))
    if not all(torch.isfinite(x).all() for x in vectors):
        raise RuntimeError('Nonfinite objective gradient')
    policy, reference_vector, entropy_vector = [x.double().cpu() for x in vectors]
    regularizer = reference_vector + entropy_vector
    squared_norm = float(policy.dot(policy))
    if squared_norm <= 1e-20:
        raise RuntimeError('Degenerate policy gradient; cancellation undefined')
    cosine = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    return {
        'states': count, **totals,
        'policy_gradient_norm': float(policy.norm()),
        'weighted_reference_gradient_norm': float(reference_vector.norm()),
        'weighted_entropy_gradient_norm': float(entropy_vector.norm()),
        'regularizer_to_policy_norm_ratio': float(regularizer.norm() / policy.norm()),
        'policy_reference_cosine': cosine(policy, reference_vector),
        'policy_entropy_cosine': cosine(policy, entropy_vector),
        'policy_regularizer_cosine': cosine(policy, regularizer),
        'regularizer_cancellation_fraction': -float(policy.dot(regularizer)) / squared_norm,
        'combined_policy_cosine': cosine(policy, policy + regularizer),
        'combined_gradient_norm': float((policy + regularizer).norm()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    plan = json.loads(options.plan.read_text())
    install_training(ppo)
    identity = rule_identity()
    assert identity == plan['rule_identity']
    assert digest(__file__) == plan['diagnostic_source_sha256']
    torch.set_num_threads(4)
    device = torch.device('cuda')
    report = dict(purpose=__doc__, plan_sha256=digest(options.plan), rule_identity=identity,
                  started_at=datetime.now(timezone.utc).isoformat(), batches=[], completed=False)
    options.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = options.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(options.output)

    save()
    for run in plan['runs']:
        args = SimpleNamespace(**json.loads(Path(run['config']).read_text()))
        args.hands_per_update = plan['matches_per_batch']
        assert args.rules_version == identity['rules_version']
        assert args.policy_action_contract == identity['policy_action_contract']
        assert digest(run['checkpoint']) == run['checkpoint_sha256']
        assert digest(run['config']) == run['config_sha256']
        torch.manual_seed(run['seeds'][0])
        actor, payload = load_checkpoint_model(run['checkpoint'], map_location=device)
        ac = ppo.ActorCritic(actor, obs_channels=args.obs_channels).to(device).eval()
        ac.critic.load_state_dict(payload['ppo_value_state_dict'])
        before = {k: v.detach().cpu().clone() for k, v in ac.state_dict().items()}
        baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
        baseline = baseline.to(device).eval()
        baseline.load_state_dict(payload['ppo_baseline_state_dict'])
        opponents = OpponentPool(ac.actor, args.opponent_pool_size)
        opponents.restore(payload['ppo_opponent_pool_states'])
        ppo.ROLLOUT_INFERENCE.enabled = True
        with ppo.ParallelGamePool(args.hands_per_update, 20, args.max_steps,
                                 fast_forward_forced=True, async_min_workers=1) as env:
            for seed in run['seeds']:
                args.seed = seed
                torch.manual_seed(seed)
                completed, failures = ppo.collect_rollout(
                    ac, baseline, opponents, args, device, np.random.default_rng(seed),
                    0, env_pool=env)
                if failures or len(completed) != args.hands_per_update or any(x['invalid'] for x in completed):
                    raise RuntimeError(f'Invalid diagnostic trajectories: {failures}')
                batch = ppo.build_ppo_batch(completed, args)
                row = dict(checkpoint=run['checkpoint'], seed=seed, matches=len(completed),
                           failures=0, **objective_gradients(ac.actor, baseline, batch, args, device))
                report['batches'].append(row)
                save()
                print(json.dumps(row), flush=True)
                del completed, batch
        if not all(torch.equal(v.detach().cpu(), before[k]) for k, v in ac.state_dict().items()):
            raise RuntimeError('Frozen parameters changed')
        assert digest(run['checkpoint']) == run['checkpoint_sha256']
        del ac, actor, baseline, opponents, before, payload
    summaries = {}
    for run in plan['runs']:
        rows = [r for r in report['batches'] if r['checkpoint'] == run['checkpoint']]
        summaries[run['checkpoint']] = {
            'mean_cancellation_fraction': float(np.mean([r['regularizer_cancellation_fraction'] for r in rows])),
            'all_batches_opposed': all(r['policy_regularizer_cosine'] < 0 for r in rows),
        }
    report.update(
        completed=True, parameters_unchanged=True, checkpoints_unchanged=True,
        summaries=summaries,
        supports_regularizer_cancellation_hypothesis=all(
            r['mean_cancellation_fraction'] >= plan['minimum_mean_cancellation_fraction']
            and r['all_batches_opposed'] for r in summaries.values()),
        finished_at=datetime.now(timezone.utc).isoformat(),
        caution='Raw gradients before Adam/clipping. Not strength evidence or permission to install.')
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'batches'}), flush=True)


if __name__ == '__main__':
    main()
