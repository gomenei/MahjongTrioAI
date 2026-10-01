"""Run a frozen policy through candidate match flow; never measure improvement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from battle_models import Policy, choose_actions
from model import load_checkpoint_model
from training.rules_candidate_match import CandidateSouthMatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--matches', type=int, default=64)
    parser.add_argument('--seed-start', type=int, default=985260917)
    args = parser.parse_args()
    if args.matches < 1:
        parser.error('--matches must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / 'status.json'
    if status_path.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = ROOT / 'model/autodl_verified_20260916.pt'
    model, metadata = load_checkpoint_model(checkpoint, map_location=device)
    policy = Policy('frozen_champion', checkpoint, model.to(device).eval(), metadata['observation_key'])
    status = dict(purpose='Rule flow and checkpoint compatibility only; no strength claim',
                  pid=os.getpid(), status='running', target_matches=args.matches,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  candidate_source_sha256=hashlib.sha256(Path(CandidateSouthMatch.__module__.replace('.', '/')+'.py').read_bytes()).hexdigest(),
                  rules_version=CandidateSouthMatch.rules_version,
                  candidate_sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                     for p in ('training/rules_candidate_match.py',
                                               'training/rules_candidate_game.py',
                                               'training/rules_candidate_feature.py',
                                               'training/rules_candidate_kan.py',
                                               'training/rules_candidate_scoring.py',
                                               'training/rules_candidate_metrics.py')},
                  seed_start=args.seed_start,
                  completed=0, errors=[], matches=[])

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
        temporary.replace(status_path)

    save()
    for index in range(args.matches):
        seed = args.seed_start + index
        start_round = 0 if index % 2 == 0 else 6
        game = CandidateSouthMatch()
        observation = game.reset(seed)
        if start_round:
            game.round_index, game.dealer = start_round, 0
            observation = game._start_hand()
        steps, west_seen = 0, bool(start_round)
        try:
            while not game.done:
                names = list(observation)
                values = [observation[name] for name in names]
                if not all(np.isfinite(v[policy.observation_key]).all() for v in values):
                    raise ValueError('Non-finite observation')
                actions = choose_actions(policy, values, device, 'greedy', 1., np.random.default_rng(seed))
                if not all(v['action_mask'][a] > 0 for v, a in zip(values, actions)):
                    raise ValueError('Policy selected an illegal action')
                observation, _, _ = game.step(dict(zip(names, actions)))
                steps += 1
                west_seen |= game.round_index >= 6
                if '非法动作' in game.game.result_message or steps > 20000:
                    raise ValueError('Invalid action or step safety limit')
            if sum(game.scores) != 105000 or sorted(game.final_ranks) != [1, 2, 3]:
                raise ValueError('Invalid final score total or ranks')
            if any(bool(h.winners) != (h.winner is not None)
                   or len(h.winners) != len(set(h.winners)) for h in game.history):
                raise ValueError('Inconsistent multi-winner history')
            status['completed'] += 1
            status['matches'].append(dict(seed=seed, start_round=start_round,
                                          hands=game.hand_count, steps=steps,
                                          west_seen=west_seen, final_round=game.round_index,
                                          multi_ron_hands=sum(len(h.winners)>1 for h in game.history)))
        except Exception as error:
            status['errors'].append(dict(seed=seed, start_round=start_round, error=str(error)))
            break
        save()
    status['status'] = 'failed' if status['errors'] else 'completed'
    save()
    print(json.dumps({k:v for k,v in status.items() if k!='matches'}), flush=True)


if __name__ == '__main__':
    main()
