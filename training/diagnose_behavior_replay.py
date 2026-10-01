"""Compare recorded behavior probabilities with frozen forward paths.

No optimizer, no strength measurements, and no candidate selection. Saved fresh
observations can be reused for numerical debugging, never for promotion tests.
"""
import argparse
from contextlib import nullcontext
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
from torch.nn.attention import SDPBackend, sdpa_kernel

import ppo_train as ppo
from model import load_checkpoint_model
from training.ppo_support import OpponentPool
from training.rules_candidate_runtime import install_training, rule_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    install_training(ppo)
    args = SimpleNamespace(**json.loads(options.config.read_text()))
    args.hands_per_update, args.seed = 128, options.seed
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    checkpoint_hash = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    actor, payload = load_checkpoint_model(options.checkpoint, map_location=device)
    ac = ppo.ActorCritic(actor, obs_channels=args.obs_channels).to(device).eval()
    ac.critic.load_state_dict(payload['ppo_value_state_dict'])
    before = {k: v.detach().cpu().clone() for k, v in ac.state_dict().items()}
    baseline, _ = load_checkpoint_model(args.base_model, map_location=device)
    baseline = baseline.to(device).eval()
    baseline.load_state_dict(payload['ppo_baseline_state_dict'])
    opponents = OpponentPool(ac.actor, args.opponent_pool_size)
    opponents.restore(payload['ppo_opponent_pool_states'])
    ppo.ROLLOUT_INFERENCE.enabled = True
    with ppo.ParallelGamePool(128, 20, args.max_steps, fast_forward_forced=True,
                             async_min_workers=1) as env:
        completed, failures = ppo.collect_rollout(
            ac, baseline, opponents, args, device, np.random.default_rng(args.seed),
            0, env_pool=env)
    if failures or len(completed) != 128 or any(x['invalid'] for x in completed):
        raise RuntimeError(f'Invalid trajectories: {failures}')
    batch = ppo.build_ppo_batch(completed, args)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(options.output.with_suffix('.npz'), **{
        k: batch[k] for k in ('observation', 'action_mask', 'actions', 'old_log_probabilities')})
    modes = ('graph_inference', 'eager_inference', 'gradient_forward',
             'unfused_inference', 'unfused_gradient', 'math_inference', 'math_gradient')
    values = {mode: [] for mode in modes}
    greedy = {mode: [] for mode in modes}
    original_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        for start in range(0, len(batch['actions']), 512):
            chunk = slice(start, start + 512)
            inputs = {k: torch.as_tensor(batch[k][chunk], device=device)
                      for k in ('observation', 'action_mask')}
            actions = torch.as_tensor(batch['actions'][chunk], device=device)
            for mode in modes:
                torch.backends.mha.set_fastpath_enabled(
                    False if mode.startswith(('unfused', 'math')) else original_fastpath)
                gradient = mode.endswith('gradient') or mode == 'gradient_forward'
                context = torch.enable_grad() if gradient else torch.inference_mode()
                attention = sdpa_kernel(SDPBackend.MATH) if mode.startswith('math') else nullcontext()
                with context, attention:
                    logits = (ppo.ROLLOUT_INFERENCE(ac, inputs)[0]
                              if mode == 'graph_inference' else ac.actor(inputs))
                    logp = Categorical(logits=logits / args.policy_temperature).log_prob(actions)
                    values[mode].extend(logp.detach().cpu().tolist())
                    greedy[mode].extend(logits.detach().argmax(-1).cpu().tolist())
                del logits, logp
    finally:
        torch.backends.mha.set_fastpath_enabled(original_fastpath)
    values = {k: np.asarray(v) for k, v in values.items()}
    recorded = batch['old_log_probabilities']

    def comparison(left, right):
        error = np.abs(left - right)
        return dict(mean=float(error.mean()), maximum=float(error.max()),
                    p99=float(np.quantile(error, .99)),
                    fraction_above_1e_3=float(np.mean(error > .001)))

    report = dict(purpose=__doc__, completed=True, matches=128, failures=0,
                  decision_states=len(recorded), seed=args.seed, checkpoint_sha256=checkpoint_hash,
                  rule_identity=rule_identity(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  matmul_precision=torch.get_float32_matmul_precision(),
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  original_mha_fastpath=original_fastpath,
                  versus_recorded={k: comparison(v, recorded) for k, v in values.items()},
                  versus_eager_inference={k: comparison(v, values['eager_inference']) for k, v in values.items()},
                  versus_math_inference={k: comparison(v, values['math_inference']) for k, v in values.items()},
                  greedy_change_vs_eager={k: float(np.mean(np.asarray(v) != greedy['eager_inference']))
                                          for k, v in greedy.items()},
                  finished_at=datetime.now(timezone.utc).isoformat())
    assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in ac.state_dict().items())
    assert hashlib.sha256(options.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    report.update(parameters_unchanged=True, checkpoint_unchanged=True)
    options.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
