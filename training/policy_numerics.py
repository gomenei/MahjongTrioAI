"""Opt-in agreement between rollout probabilities and gradient forward passes.

Production evaluation keeps its usual forward path. Only training processes
enable this mode, before building CUDA graphs or collecting any trajectories.
"""
import hashlib
from pathlib import Path

import torch

from training.cuda_rollout import ROLLOUT_INFERENCE

NUMERICS_VERSION = 'unfused_transformer_rollout_v1'


def identity():
    return {
        'training_numerics': NUMERICS_VERSION,
        'training_mha_fastpath': False,
        'evaluation_mha_fastpath': True,
        'training_numerics_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def configure():
    # Captured CUDA graphs retain the old execution path after the global switch.
    # Drop them before any rollout; the model weights and optimizer are unchanged.
    ROLLOUT_INFERENCE.models.clear()
    torch.backends.mha.set_fastpath_enabled(False)


def install_training(module):
    configure()
    parse = module.parse_args
    def parse_consistent_args():
        args = parse()
        args.training_numerics = NUMERICS_VERSION
        return args
    module.parse_args = parse_consistent_args
    validate = module.validate_resume_settings
    def validate_consistent_resume(payload, args):
        if payload.get('ppo_args', {}).get('training_numerics') != NUMERICS_VERSION:
            raise ValueError('Cannot resume a PPO run with different training numerics')
        validate(payload, args)
    module.validate_resume_settings = validate_consistent_resume
    for name in ('checkpoint_payload', 'policy_snapshot_payload'):
        original = getattr(module, name)
        def consistent_payload(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            result.update(identity())
            return result
        setattr(module, name, consistent_payload)
