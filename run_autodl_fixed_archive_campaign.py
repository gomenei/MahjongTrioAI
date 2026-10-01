"""Generation v11: train against the entire frozen archive throughout each run."""
from pathlib import Path

import run_autodl_generation_campaign as generation
from run_autodl_training import BASE, HISTORY, ROOT, sha256


RECIPE = 'generation_v11'
TRAIN_SEEDS = (210_260_917, 230_260_917)
CONFIRM_SEEDS = (3_841_000_000, 3_951_000_000)
EXPORT_SMOKE_SEED = 4_061_000_000
ARCHIVE = (BASE, HISTORY, Path('model/archive/rank_v5.pt'), Path('model/archive/rank_v7.pt'))
ARCHIVE_SHA256 = (
    'dc1e6a850ce76e89ce5cd3820d36f4f8e2ac7ff5145e1eb3474290d4f58dafe5',
    '7c1d812c6a6e495664c7cb339e29c023570b804595d46088f8425eb9a1632192',
    '36367a5f7a8f025f3b711224c1068c04a3afd673d216f49e7d98a35002e74d4b',
    'f7d488d098d9c8484d2d44b1d065e38ef229819fbe0912e52e05a7bbc7d7b513',
)


def training_command(seed, workers, device, smoke=False):
    command = generation.training_command(seed, workers, device, smoke)
    changes = {'--output-dir': f'ppo_runs/{RECIPE}_{seed}' + ('_smoke' if smoke else ''),
               '--opponent-pool-size': str(len(ARCHIVE) + 1), '--snapshot-interval': '100001'}
    for flag, value in changes.items():
        command[command.index(flag) + 1] = value
    start = command.index('--opponent-models') + 1
    command[start:start + 1] = [x.as_posix() for x in ARCHIVE]
    # Initial pool = champion + four archived policies, fixed for the whole run.
    assert int(command[command.index('--snapshot-interval') + 1]) > int(command[command.index('--updates') + 1])
    return command


def main():
    for path, digest in zip(ARCHIVE, ARCHIVE_SHA256):
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Frozen archive changed: {path}')
    generation.main(recipe=RECIPE, training_seeds=TRAIN_SEEDS, confirm_seeds=CONFIRM_SEEDS,
                    export_smoke_seed=EXPORT_SMOKE_SEED, command_builder=training_command,
                    extra_sources=(Path(__file__).name,), extra_identity={
                        'hypothesis': 'Stable archive coverage reduces specialization to recent snapshots',
                        'archive': [{'checkpoint': p.as_posix(), 'sha256': h} for p, h in zip(ARCHIVE, ARCHIVE_SHA256)],
                        'training_opponent_probabilities': {'current_champion': .4, 'each_archived_model': .15},
                        'snapshot_pool_frozen_for_entire_run': True,
                        'promotion_requires_archive_gate': 'archive_gate_v2', 'statistical_attempt': 2})


if __name__ == '__main__':
    main()
