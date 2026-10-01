from types import SimpleNamespace

import pytest
import torch

from training.cuda_rollout import ROLLOUT_INFERENCE
from training.policy_numerics import configure, identity, install_training


@pytest.fixture(autouse=True)
def restore_global_state():
    original = torch.backends.mha.get_fastpath_enabled()
    yield
    ROLLOUT_INFERENCE.models.clear()
    torch.backends.mha.set_fastpath_enabled(original)


def test_switch_discards_previously_captured_forward_paths():
    model = torch.nn.Linear(1, 1)
    ROLLOUT_INFERENCE.models[model] = {'old_captured_path': object()}
    torch.backends.mha.set_fastpath_enabled(True)
    configure()
    assert not torch.backends.mha.get_fastpath_enabled()
    assert len(ROLLOUT_INFERENCE.models) == 0


def test_resume_rejects_old_numerics_and_records_full_snapshot_identity():
    validated = []
    module = SimpleNamespace(
        parse_args=lambda: SimpleNamespace(),
        validate_resume_settings=lambda payload,args: validated.append(True),
        checkpoint_payload=lambda: {'keep':'checkpoint'},
        policy_snapshot_payload=lambda: {'keep':'policy'})
    install_training(module)
    args = module.parse_args()
    with pytest.raises(ValueError, match='different training numerics'):
        module.validate_resume_settings({'ppo_args':{}},args)
    assert not validated
    module.validate_resume_settings({'ppo_args':vars(args)},args)
    assert validated == [True]
    for method in ('checkpoint_payload','policy_snapshot_payload'):
        payload = getattr(module,method)()
        assert all(payload[k] == v for k,v in identity().items())
        assert 'keep' in payload
