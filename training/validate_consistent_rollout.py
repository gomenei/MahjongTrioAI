"""Bounded, frozen comparison of legacy and consistent rollout numerics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.distributions import Categorical

import ppo_train as ppo
from model import load_checkpoint_model
from training.ppo_support import OpponentPool
from training.rules_candidate_runtime import install_training as install_rules, rule_identity
from training.policy_numerics import configure, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    install_rules(ppo)
    args = SimpleNamespace(**json.loads(options.config.read_text()))
    args.hands_per_update, args.seed = 512, 760260918
    torch.set_num_threads(4)
    device = torch.device('cuda')
    digest = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    actor, payload = load_checkpoint_model(options.checkpoint, map_location=device)
    ac = ppo.ActorCritic(actor, obs_channels=args.obs_channels).to(device).eval()
    ac.critic.load_state_dict(payload['ppo_value_state_dict'])
    before = {k: v.detach().cpu().clone() for k, v in ac.state_dict().items()}
    baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
    baseline = baseline.to(device).eval()
    baseline.load_state_dict(payload['ppo_baseline_state_dict'])
    opponents = OpponentPool(ac.actor, args.opponent_pool_size)
    opponents.restore(payload['ppo_opponent_pool_states'])
    report = dict(seed=args.seed, matches_per_mode=512, checkpoint_sha256=digest,
                  purpose=__doc__, rule_identity=rule_identity(), numerics=identity(),
                  started_at=datetime.now(timezone.utc).isoformat(), modes={}, completed=False)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in ('legacy', 'consistent'):
        if mode == 'consistent':
            configure()
        else:
            ppo.ROLLOUT_INFERENCE.models.clear()
            torch.backends.mha.set_fastpath_enabled(True)
        assert not ppo.ROLLOUT_INFERENCE.models
        ppo.ROLLOUT_INFERENCE.enabled = True
        torch.manual_seed(args.seed)
        with ppo.ParallelGamePool(512, 20, args.max_steps, fast_forward_forced=True,
                                 async_min_workers=1) as env:
            started = time.perf_counter()
            completed, failures = ppo.collect_rollout(
                ac, baseline, opponents, args, device, np.random.default_rng(args.seed),
                0, env_pool=env)
            elapsed = time.perf_counter() - started
        if failures or len(completed) != 512 or any(x['invalid'] for x in completed):
            raise RuntimeError(f'Invalid trajectories: {failures}')
        batch = ppo.build_ppo_batch(completed, args)
        replay = []
        for start in range(0, len(batch['actions']), 512):
            chunk = slice(start, start+512)
            inputs = {k: torch.as_tensor(batch[k][chunk], device=device)
                      for k in ('observation', 'action_mask')}
            actions = torch.as_tensor(batch['actions'][chunk], device=device)
            # Gradients must be enabled to exercise the actual optimizer path.
            with torch.enable_grad():
                logits = ac.actor(inputs)
                logp = Categorical(logits=logits / args.policy_temperature).log_prob(actions)
                replay.extend(logp.detach().cpu().tolist())
            del logits, logp
        discrepancy = np.abs(np.asarray(replay) - batch['old_log_probabilities'])
        report['modes'][mode] = dict(matches=512, failures=0, states=len(discrepancy),
            rollout_seconds=elapsed, matches_per_second=512/elapsed,
            mean_log_probability_error=float(discrepancy.mean()),
            max_log_probability_error=float(discrepancy.max()),
            mha_fastpath=torch.backends.mha.get_fastpath_enabled())
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['modes'][mode]), flush=True)
        del completed, batch
    assert all(torch.equal(v.detach().cpu(), before[k]) for k,v in ac.state_dict().items())
    assert hashlib.sha256(options.checkpoint.read_bytes()).hexdigest() == digest
    corrected = report['modes']['consistent']
    report.update(completed=True, parameters_unchanged=True, checkpoint_unchanged=True,
        pass_integrity=(corrected['max_log_probability_error'] < 1e-4
                        and corrected['mean_log_probability_error'] < 1e-6),
        rollout_time_ratio=corrected['rollout_seconds']/report['modes']['legacy']['rollout_seconds'],
        finished_at=datetime.now(timezone.utc).isoformat(),
        caution='Numerical correctness only, not strength; paired timing is descriptive, not a benchmark.')
    options.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
