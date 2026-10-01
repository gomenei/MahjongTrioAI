"""Keep the permanent archive gate and additionally require identical v6 rules."""
from types import SimpleNamespace

import run_autodl_archive_gate as gate
from training.rules_candidate_runtime import rule_identity


def assess_rules_pair(progress, *args, **kwargs):
    expected = rule_identity()
    if any(progress['signature'].get(k) != v for k,v in expected.items()):
        raise ValueError('Archive comparison used different rules, sources or action contract')
    return ORIGINAL_ASSESS(progress, *args, **kwargs)


ORIGINAL_ASSESS = gate.assess_pair
ORIGINAL_RUN = gate.subprocess.run


def run_rules(command, **kwargs):
    command = list(command)
    if command[1] != 'battle_models.py':
        raise ValueError('Unexpected archive subprocess')
    command[1] = 'battle_models_rules.py'
    return ORIGINAL_RUN(command, **kwargs)


if __name__ == '__main__':
    gate.assess_pair = assess_rules_pair
    gate.subprocess = SimpleNamespace(run=run_rules)
    gate.main()
