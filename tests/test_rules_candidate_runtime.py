from types import SimpleNamespace

import numpy as np
import pytest

from training.rules_candidate_match import CandidateSouthMatch
from training.rules_candidate_parallel import GameSpec, ParallelGamePool
from training.rules_candidate_runtime import install_args


def test_explicit_rules_contract_and_hand_mode_rejection():
    module = SimpleNamespace(parse_args=lambda: SimpleNamespace(game_mode='south'))
    install_args(module)
    args = module.parse_args()
    assert args.rules_version == CandidateSouthMatch.rules_version
    assert args.policy_action_contract.endswith('always_declined')
    assert args.observation_contract == 'current_hand_after_draw_and_kan_v1'
    assert len(args.observation_source_sha256) == 64
    module = SimpleNamespace(parse_args=lambda: SimpleNamespace(game_mode='hand'))
    install_args(module)
    with pytest.raises(ValueError): module.parse_args()


@pytest.mark.parametrize('field', ['rules_version','policy_action_contract','rules_source_sha256',
                                 'observation_contract','observation_source_sha256'])
def test_archive_gate_rejects_mixed_rules_before_scoring(field):
    from run_autodl_rules_archive_gate import assess_rules_pair
    from training.rules_candidate_runtime import rule_identity
    signature = rule_identity()
    signature.pop(field)
    with pytest.raises(ValueError,match='different rules'):
        assess_rules_pair({'signature':signature},'new','old',123,6000,.001)


def actions(observations):
    result = {}
    for name, obs in observations.items():
        legal = np.flatnonzero(obs['action_mask'])
        result[name] = 175 if 175 in legal else int(legal[0])
    return result


def test_spawned_workers_run_candidate_and_match_serial_results():
    expected = {}
    for slot in range(2):
        g = CandidateSouthMatch()
        obs = g.reset(997260917+slot)
        while not g.done:
            obs, _, _ = g.step(actions(obs))
        expected[slot] = (g.scores,g.final_ranks,g.hand_count)
    with ParallelGamePool(capacity=2,workers=2,max_steps=5000,async_min_workers=1) as pool:
        results = pool.start([GameSpec(slot,997260917+slot,game_mode='south') for slot in range(2)])
        completed = {}
        while results:
            pending = {}
            for row in results:
                assert not row.get('error')
                slot = row['slot']
                if row['done']:
                    g = row['game']
                    assert isinstance(g,CandidateSouthMatch)
                    assert all(hasattr(h,'winners') for h in g.history)
                    completed[slot] = (g.scores,g.final_ranks,g.hand_count)
                else:
                    pending[slot] = actions(pool.observations(slot))
            if not pool.active_slots: break
            results = pool.step(pending)
        assert completed == expected
