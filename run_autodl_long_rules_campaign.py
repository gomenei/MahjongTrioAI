"""Generation v13: eight-hour branches and a frozen late-checkpoint average."""
from pathlib import Path

import run_autodl_long_generation_campaign as generation
import run_autodl_fixed_archive_campaign as archive
from run_autodl_training import ROOT, sha256
from training.rules_candidate_runtime import RULE_SOURCES, rule_identity
from training.policy_numerics import identity as numerics_identity

RECIPE = 'generation_v13'
TRAIN_SEEDS = (4100000000, 4180000000)
SMOKE_SEEDS = {4100000000: 380260918, 4180000000: 390260918}
CONFIRM_SEEDS = (2700000000, 2810000000)
EXPORT_SMOKE_SEED = 2590000000
ORIGINAL_RUN = generation.run
ORIGINAL_AVERAGE = generation.average_checkpoints


def training_command(seed, workers, device, smoke=False):
    command = archive.training_command(seed, workers, device, smoke)
    command[1] = 'ppo_train_consistent_rules.py'
    changes = {
        '--output-dir': f'ppo_runs/{RECIPE}_{seed}' + ('_smoke' if smoke else ''),
        '--checkpoint-minutes': '0' if smoke else '40',
        '--max-hours': '0' if smoke else '8',
        '--seed': str(SMOKE_SEEDS[seed] if smoke else seed),
    }
    for flag, value in changes.items():
        command[command.index(flag)+1] = value
    return command


def run_rules(command):
    command = list(command)
    if command[1] == 'battle_models.py':
        # Promotion uses the same standard inference path as the installed app.
        command[1] = 'battle_models_rules.py'
    elif command[1] != 'ppo_train_consistent_rules.py':
        raise ValueError('Unexpected generation subprocess')
    return ORIGINAL_RUN(command)


def average_rules(sources, destination, **kwargs):
    import torch
    rules, numerics = rule_identity(), numerics_identity()
    for path in sources:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('rules_version', 'policy_action_contract',
                    'observation_contract', 'observation_source_sha256'):
            if payload.get(key) != rules[key]:
                raise ValueError('Averaging source has a different rules/action contract')
        if any(payload.get(k) != v for k,v in numerics.items()):
            raise ValueError('Averaging source used different training numerics')
    validation = ORIGINAL_AVERAGE(sources, destination, **kwargs)
    payload = torch.load(destination, map_location='cpu', weights_only=True)
    payload.update(rules)
    payload.update(numerics)
    torch.save(payload, destination)
    return {**validation, **rules, **numerics}


def main():
    for path,digest in zip(archive.ARCHIVE, archive.ARCHIVE_SHA256):
        if sha256(ROOT/path) != digest:
            raise ValueError(f'Archive changed: {path}')
    generation.run = run_rules
    generation.average_checkpoints = average_rules
    generation.main(recipe=RECIPE, training_seeds=TRAIN_SEEDS, confirm_seeds=CONFIRM_SEEDS,
        export_smoke_seed=EXPORT_SMOKE_SEED, command_builder=training_command,
        extra_sources=(*RULE_SOURCES, 'training/policy_numerics.py',
            'ppo_train_consistent_rules.py', 'battle_models_rules.py',
            'run_autodl_rules_archive_gate.py', 'run_autodl_fixed_archive_campaign.py',
            'run_autodl_generation_campaign.py', Path(__file__).name),
        extra_identity={**rule_identity(), **numerics_identity(),
            'hypothesis': 'Two-hour branches may be undertrained; increase to8h with late fixed averaging',
            'attribution_limit': 'Duration, averaging window and training numerics change together; no isolated causal claim',
            'candidate_minutes_per_seed': [360,400,440,480],
            'smoke_seeds': {str(k):v for k,v in SMOKE_SEEDS.items()},
            'archive': [{'checkpoint':p.as_posix(),'sha256':h}
                        for p,h in zip(archive.ARCHIVE,archive.ARCHIVE_SHA256)],
            'training_opponent_probabilities': {'current_champion':.4,'each_archived_model':.15},
            'snapshot_pool_frozen_for_entire_run': True,
            'promotion_requires_archive_gate': 'archive_gate_v4',
            'statistical_attempt': 4,
            'remaining_rank_evidence': 'Targeted v6 rules only; no strong external or Soulten evidence'})


if __name__ == '__main__':
    main()
