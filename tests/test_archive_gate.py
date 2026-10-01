import copy

import pytest

from run_autodl_archive_gate import assess_pair, comparison_alpha, promotion_decision, validate_plan


def progress_fixture(groups=10):
    result = {'signature': {'models': [{'name': 'candidate', 'sha256': 'new'},
                                      {'name': 'baseline', 'sha256': 'old'}],
                            'seed': 123, 'game_mode': 'south', 'rounds_per_wind': 3,
                            'action_mode': 'greedy'}, 'lineups': {}}
    for index, lineup in enumerate(((0, 0, 1), (0, 1, 1)), 1):
        rows = []
        for i in range(groups):
            models = {}
            for model in (0, 1):
                hands = lineup.count(model) * 3
                sign = 1 if model == 0 else -1
                models[str(model)] = {'hands': hands, 'played_hands': hands * 8,
                                     'rank_sum': (2 - sign * .1) * hands,
                                     'point_sum': sign * (1000 + i * 10) * hands,
                                     'utility_sum': sign * (.1 + i * .001) * hands,
                                     'deal_ins': hands, 'invalid_actions': 0}
            rows.append({'seed': 123 + index * 10_000_000 + i, 'models': models})
        result['lineups'][','.join(map(str, lineup))] = {
            'model_indices': list(lineup), 'complete': True, 'errors': [], 'groups': rows}
    return result


def test_winning_every_archived_opponent_is_required():
    passed = assess_pair(progress_fixture(), 'new', 'old', 123, 10, .001, .02)
    assert passed['pass']
    assert passed['matches'] == 60
    assert not promotion_decision({'B': [passed, passed]}, ['B', 'C'])
    # A beats B, B historically beats C, but C beats A: reject A, even if pooled
    # scores against B would overwhelm this loss. No Elo averaging bypass.
    lost = copy.deepcopy(progress_fixture())
    for lineup in lost['lineups'].values():
        for group in lineup['groups']:
            for row in group['models'].values():
                row['point_sum'] *= -1
                row['utility_sum'] *= -1
                row['rank_sum'] = 4 * row['hands'] - row['rank_sum']
    failed = assess_pair(lost, 'new', 'old', 123, 10, .001, .02)
    assert not failed['pass']
    assert not promotion_decision({'B': [passed, passed], 'C': [failed, failed]}, ['B', 'C'])
    assert promotion_decision({'B': [passed, passed], 'C': [passed, passed]}, ['B', 'C'])


@pytest.mark.parametrize('defect', ['hash', 'seed', 'rotation', 'incomplete', 'lineup', 'nan'])
def test_invalid_evidence_is_rejected(defect):
    progress = progress_fixture()
    first = next(iter(progress['lineups'].values()))
    if defect == 'hash':
        progress['signature']['models'][1]['sha256'] = 'unexpected'
    elif defect == 'seed':
        first['groups'][1]['seed'] = first['groups'][0]['seed']
    elif defect == 'rotation':
        first['groups'][0]['models']['0']['hands'] -= 1
    elif defect == 'incomplete':
        first['complete'] = False
    elif defect == 'lineup':
        del progress['lineups']['0,1,1']
    else:
        first['groups'][0]['models']['0']['point_sum'] = float('nan')
    with pytest.raises(ValueError):
        assess_pair(progress, 'new', 'old', 123, 10, .001)


@pytest.mark.parametrize('defect', ['simulation_error', 'invalid_action', 'fallback'])
def test_errors_and_bridge_fallbacks_cannot_count_as_strength(defect):
    progress = progress_fixture()
    first = next(iter(progress['lineups'].values()))
    if defect == 'simulation_error':
        first['errors'].append('failed match')
    elif defect == 'invalid_action':
        first['groups'][0]['models']['0']['invalid_actions'] = 1
    else:
        progress['bridge_stats'] = {'fallbacks': 1}
    assert not assess_pair(progress, 'new', 'old', 123, 10, .001)['pass']


def test_reversed_model_order_preserves_candidate_orientation():
    progress = progress_fixture()
    original = assess_pair(progress, 'new', 'old', 123, 10, .001)
    progress['signature']['models'].reverse()
    for item in progress['lineups'].values():
        for group in item['groups']:
            for model, row in group['models'].items():
                row['point_sum'] *= -1
                row['utility_sum'] *= -1
                row['rank_sum'] = 4 * row['hands'] - row['rank_sum']
    reversed_result = assess_pair(progress, 'new', 'old', 123, 10, .001)
    for key in ('rank_improvement', 'candidate_average_rank', 'opponent_average_rank'):
        assert reversed_result.pop(key) == pytest.approx(original.pop(key))
    assert reversed_result == original


def test_repeated_attempts_and_many_opponents_tighten_confidence():
    assert comparison_alpha(2, 5) < comparison_alpha(1, 5) < comparison_alpha(1, 2)
    total = sum(comparison_alpha(i, 5) * 5 * 2 * 2 for i in range(1, 1000))
    assert total < .05


def test_plan_prevents_one_holdout_or_duplicate_archive_entries():
    plan = {'groups': 6000, 'attempt': 1, 'opponents': [
        {'name': 'current', 'role': 'current_champion', 'sha256': 'a', 'seeds': [1, 2]},
        {'name': 'history', 'role': 'archive', 'sha256': 'b', 'seeds': [3, 4]}]}
    validate_plan(plan)
    for field, value in [('sha256', 'a'), ('seeds', [3]), ('seeds', [1, 4])]:
        changed = copy.deepcopy(plan)
        changed['opponents'][1][field] = value
        with pytest.raises(ValueError):
            validate_plan(changed)
