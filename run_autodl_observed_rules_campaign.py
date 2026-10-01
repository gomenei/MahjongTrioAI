"""Generation v15: correct drawn-tile observations and independently re-evaluate all six archives."""
from pathlib import Path

import run_autodl_long_generation_campaign as generation
import run_autodl_fixed_archive_campaign as archive
from run_autodl_long_rules_campaign import average_rules, run_rules
from run_autodl_training import ROOT, sha256
from training.rules_candidate_runtime import RULE_SOURCES, rule_identity
from training.policy_numerics import identity as numerics_identity

RECIPE = 'generation_v15'
CHAMPION = Path('model/autodl_verified_20260918.pt')
CHAMPION_SHA256 = 'd4dcf0d7ba2aeeba562056ed586f97f95b5bc67bb478bfe8f9fae97db7ae675e'
TRAIN_SEEDS = (600000000, 1000000000)
SMOKE_SEEDS = {600000000: 580260919, 1000000000: 590260919}
CONFIRM_SEEDS = (1230000000, 1340000000)
EXPORT_SMOKE_SEED = 1670000000
ARCHIVE = (Path('model/autodl_verified_20260916.pt'), *archive.ARCHIVE)
ARCHIVE_SHA256 = ('160c8c3bba2cf5a808be56eebd46b91828e4d193bcacd7a266a50e2efcf48895', *archive.ARCHIVE_SHA256)


def training_command(seed, workers, device, smoke=False):
    command = archive.training_command(seed, workers, device, smoke)
    command[1] = 'ppo_train_consistent_rules.py'
    changes = {
        '--base-model': CHAMPION.as_posix(),
        '--output-dir': f'ppo_runs/{RECIPE}_{seed}' + ('_smoke' if smoke else ''),
        '--checkpoint-minutes': '0' if smoke else '40',
        '--max-hours': '0' if smoke else '8',
        '--seed': str(SMOKE_SEEDS[seed] if smoke else seed),
        '--snapshot-ratio': '.72',
        '--opponent-pool-size': '6',
    }
    for flag, value in changes.items():
        command[command.index(flag) + 1] = value
    start = command.index('--opponent-models') + 1
    end = next((i for i in range(start, len(command)) if command[i].startswith('--')),len(command))
    command[start:end] = [p.as_posix() for p in ARCHIVE]
    return command


def main():
    for path, digest in zip((CHAMPION,*ARCHIVE),(CHAMPION_SHA256,*ARCHIVE_SHA256)):
        if sha256(ROOT/path) != digest:
            raise ValueError(f'Frozen opponent changed: {path}')
    generation.CHAMPION = CHAMPION
    generation.CHAMPION_SHA256 = CHAMPION_SHA256
    generation.run = run_rules
    generation.average_checkpoints = average_rules
    generation.main(recipe=RECIPE, training_seeds=TRAIN_SEEDS, confirm_seeds=CONFIRM_SEEDS,
        export_smoke_seed=EXPORT_SMOKE_SEED, command_builder=training_command,
        extra_sources=(*RULE_SOURCES, 'training/policy_numerics.py',
            'ppo_train_consistent_rules.py', 'battle_models_rules.py',
            'run_autodl_rules_archive_gate.py', 'run_autodl_fixed_archive_campaign.py',
            'run_autodl_generation_campaign.py', 'run_autodl_long_rules_campaign.py', Path(__file__).name),
        extra_identity={**rule_identity(), **numerics_identity(),
            'hypothesis': 'Repair missing drawn-tile and concealed-kan hand observations; fresh optimizer, unchanged eight-hour recipe, all opponents use the corrected input',
            'candidate_minutes_per_seed': [360,400,440,480],
            'smoke_seeds': {str(k):v for k,v in SMOKE_SEEDS.items()},
            'archive': [{'checkpoint':p.as_posix(),'sha256':h} for p,h in zip(ARCHIVE,ARCHIVE_SHA256)],
            'training_opponent_probabilities': {'current_champion':.4,'each_of_five_archived_models':.12},
            'snapshot_pool_frozen_for_entire_run': True,
            'promotion_requires_archive_gate': 'archive_gate_v6',
            'statistical_attempt': 6,
            'remaining_rank_evidence': 'Internal v6 rules only; no strong external or Soulten evidence'})


if __name__ == '__main__':
    main()
