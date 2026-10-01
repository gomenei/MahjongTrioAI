"""Generation v12: same fixed archive recipe in the repaired v6 environment."""
from pathlib import Path

import run_autodl_generation_campaign as generation
import run_autodl_fixed_archive_campaign as archive
from run_autodl_training import ROOT, sha256
from training.rules_candidate_runtime import RULE_SOURCES, rule_identity

RECIPE = 'generation_v12'
TRAIN_SEEDS = (250260917,270260917)
CONFIRM_SEEDS = (1280000000,1390000000)
EXPORT_SMOKE_SEED = 1170000000
ORIGINAL_RUN = generation.run
ORIGINAL_AVERAGE = generation.average_checkpoints


def training_command(seed, workers, device, smoke=False):
    command = archive.training_command(seed, workers, device, smoke)
    command[1] = 'ppo_train_rules.py'
    command[command.index('--output-dir')+1] = f'ppo_runs/{RECIPE}_{seed}' + ('_smoke' if smoke else '')
    # Smoke uses a separate training seed block; formal starts at its declared seed.
    if smoke:
        command[command.index('--seed')+1] = str(seed+1000000)
    return command


def run_rules(command):
    command = list(command)
    if command[1] == 'battle_models.py':
        command[1] = 'battle_models_rules.py'
    elif command[1] != 'ppo_train_rules.py':
        raise ValueError('Unexpected generation subprocess')
    return ORIGINAL_RUN(command)


def average_rules(sources, destination, **kwargs):
    import torch
    identity = rule_identity()
    for path in sources:
        payload = torch.load(path,map_location='cpu',weights_only=True)
        if any(payload.get(k) != identity[k] for k in ('rules_version','policy_action_contract')):
            raise ValueError('Averaging source has a different rules/action contract')
    validation = ORIGINAL_AVERAGE(sources,destination,**kwargs)
    payload = torch.load(destination,map_location='cpu',weights_only=True)
    payload.update(identity)
    torch.save(payload,destination)
    return {**validation,**identity}


def main():
    for p,h in zip(archive.ARCHIVE,archive.ARCHIVE_SHA256):
        if sha256(ROOT/p) != h: raise ValueError(f'Archive changed: {p}')
    generation.run = run_rules
    generation.average_checkpoints = average_rules
    generation.main(recipe=RECIPE,training_seeds=TRAIN_SEEDS,confirm_seeds=CONFIRM_SEEDS,
                    export_smoke_seed=EXPORT_SMOKE_SEED,command_builder=training_command,
                    extra_sources=(*RULE_SOURCES,'ppo_train_rules.py','battle_models_rules.py',
                                   'run_autodl_rules_archive_gate.py','run_autodl_fixed_archive_campaign.py',
                                   Path(__file__).name),
                    extra_identity={**rule_identity(),
                        'hypothesis':'Corrected scoring, West extension and multi-ron reduce reward/model mismatch',
                        'archive':[{'checkpoint':p.as_posix(),'sha256':h}
                                   for p,h in zip(archive.ARCHIVE,archive.ARCHIVE_SHA256)],
                        'training_opponent_probabilities':{'current_champion':.4,'each_archived_model':.15},
                        'snapshot_pool_frozen_for_entire_run':True,
                        'promotion_requires_archive_gate':'archive_gate_v3','statistical_attempt':3,
                        'rules_validation_scope':'Targeted cross-implementation cases, not complete platform certification',
                        'remaining_rank_evidence':'No independently verified strong opponent or Soulten calibration'})


if __name__ == '__main__': main()
