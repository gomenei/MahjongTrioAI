import sys

import pytest

from ppo_train import parse_args, validate_args
from run_autodl_fixed_archive_campaign import ARCHIVE, TRAIN_SEEDS, training_command


@pytest.mark.parametrize('smoke', [True, False])
def test_all_archived_opponents_remain_present_for_entire_run(monkeypatch, smoke):
    command = training_command(TRAIN_SEEDS[0], 20, 'cpu', smoke)
    monkeypatch.setattr(sys, 'argv', command[1:])
    args = parse_args()
    validate_args(args)
    assert args.opponent_models == [p.as_posix() for p in ARCHIVE]
    assert args.opponent_pool_size == len(ARCHIVE) + 1
    assert args.snapshot_interval > args.updates
    assert args.snapshot_ratio == .75
    assert str(args.base_model).replace('\\', '/') == 'model/autodl_verified_20260916.pt'
    assert ('_smoke' in str(args.output_dir)) == smoke
