"""Require a frozen candidate to beat every archived opponent before promotion.

This is a separate post-training job. It never changes a running experiment,
installs a model, or controls the server's power state.
"""
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from run_autodl_training import read_json, sha256, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def comparison_alpha(attempt, opponents):
    """Spend at most .05 over all attempts, then split across all comparisons.

    Each opponent has two seed sets and two one-sided metrics. The underlying
    intervals use the existing large-sample normal approximation, not an exact
    finite-sample guarantee. Failed attempts still consume their index.
    """
    if attempt < 1 or opponents < 1:
        raise ValueError("Positive attempt and opponent count required")
    return .05 / (attempt * (attempt + 1)) / (opponents * 2 * 2)


def interval(values, alpha):
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "standard_error": se,
            "ci95_low": mean - statistics.NormalDist().inv_cdf(.975) * se,
            "ci95_high": mean + statistics.NormalDist().inv_cdf(.975) * se,
            "adjusted_one_sided_lower": mean - statistics.NormalDist().inv_cdf(1 - alpha) * se}


def assess_pair(progress, candidate_sha, opponent_sha, seed, groups, alpha, min_rank_gain=0):
    """Recompute from seat-rotation groups, rejecting incomplete or mixed runs."""
    if groups < 2 or not 0 < alpha < .5:
        raise ValueError("Invalid sample size or confidence level")
    signature = progress['signature']
    models = signature['models']
    expected = {'candidate': candidate_sha, 'baseline': opponent_sha}
    if len(models) != 2 or {x['name']: x['sha256'] for x in models} != expected:
        raise ValueError('Wrong candidate or opponent identity')
    if candidate_sha == opponent_sha:
        raise ValueError('Candidate must differ from opponent')
    if (signature['seed'] != seed or signature.get('game_mode') != 'south'
            or signature.get('rounds_per_wind') != 3 or signature['action_mode'] != 'greedy'):
        raise ValueError('Wrong evaluation protocol')
    candidate_index = next(i for i, x in enumerate(models) if x['name'] == 'candidate')
    opponent_index = 1 - candidate_index
    expected_lineups = ((0, 0, 1), (0, 1, 1))
    if set(progress['lineups']) != {','.join(map(str, x)) for x in expected_lineups}:
        raise ValueError('Missing or extra mirror lineup')
    totals = {i: dict(hands=0, played_hands=0, rank_sum=0., deal_ins=0) for i in (0, 1)}
    point_differences, utility_differences = [], []
    errors = invalid = 0
    for number, lineup in enumerate(expected_lineups, 1):
        item = progress['lineups'][','.join(map(str, lineup))]
        if (not item['complete'] or item['model_indices'] != list(lineup)
                or len(item['groups']) != groups):
            raise ValueError('Incomplete mirror lineup')
        errors += len(item.get('errors', []))
        seeds = [x['seed'] for x in item['groups']]
        if len(set(seeds)) != groups or set(seeds) != set(range(seed + number * 10_000_000,
                                                              seed + number * 10_000_000 + groups)):
            raise ValueError('Duplicate or unexpected evaluation seeds')
        for group in item['groups']:
            rows = group['models']
            if set(rows) != {'0', '1'}:
                raise ValueError('Wrong model group')
            for i in (0, 1):
                row = rows[str(i)]
                if row['hands'] != lineup.count(i) * 3 or row['played_hands'] <= 0:
                    raise ValueError('Incomplete seat rotations')
                if not all(math.isfinite(float(row[k])) for k in
                           ('rank_sum', 'point_sum', 'utility_sum', 'deal_ins', 'invalid_actions')):
                    raise ValueError('Nonfinite evaluation values')
                invalid += row['invalid_actions']
                for k in totals[i]:
                    totals[i][k] += row[k]
            a, b = rows[str(candidate_index)], rows[str(opponent_index)]
            point_differences.append(a['point_sum'] / a['hands'] - b['point_sum'] / b['hands'])
            utility_differences.append(a['utility_sum'] / a['hands'] - b['utility_sum'] / b['hands'])
    a, b = totals[candidate_index], totals[opponent_index]
    rank_gain = b['rank_sum'] / b['hands'] - a['rank_sum'] / a['hands']
    deal_change = a['deal_ins'] / a['played_hands'] - b['deal_ins'] / b['played_hands']
    points, utility = interval(point_differences, alpha), interval(utility_differences, alpha)
    fallbacks = progress.get('bridge_stats', {}).get('fallbacks', 0)
    valid = errors == invalid == fallbacks == 0
    passed = (valid and rank_gain >= min_rank_gain and deal_change <= .01
              and points['adjusted_one_sided_lower'] > 0 and utility['adjusted_one_sided_lower'] > 0)
    return {'pass': passed, 'valid': valid, 'matches': groups * 6,
            'paired_seed_groups': groups * 2, 'rank_improvement': rank_gain,
            'candidate_average_rank': a['rank_sum'] / a['hands'],
            'opponent_average_rank': b['rank_sum'] / b['hands'],
            'deal_in_rate_change': deal_change, 'points': points, 'utility': utility,
            'errors': errors, 'invalid_actions': invalid, 'fallbacks': fallbacks,
            'one_sided_alpha': alpha}


def promotion_decision(results, expected_names):
    # A high pooled average cannot compensate for losing to an older champion.
    if set(results) != set(expected_names):
        return False
    return all(len(rows) == 2 and all(x['pass'] for x in rows) for rows in results.values())


def validate_plan(plan):
    opponents = plan['opponents']
    if (not opponents or opponents[0]['role'] != 'current_champion'
            or sum(x['role'] == 'current_champion' for x in opponents) != 1):
        raise ValueError('Exactly one current champion must be first')
    if (len({x['name'] for x in opponents}) != len(opponents)
            or len({x['sha256'] for x in opponents}) != len(opponents)):
        raise ValueError('Archive contains duplicate names or weights')
    if plan['groups'] < 6000 or plan['attempt'] < 1:
        raise ValueError('Insufficient samples or invalid attempt index')
    for opponent in opponents:
        if len(opponent['seeds']) != 2 or len(set(opponent['seeds'])) != 2:
            raise ValueError('Each opponent needs two distinct fixed holdouts')
    current_seeds = set(opponents[0]['seeds'])
    if any(current_seeds.intersection(x['seeds']) for x in opponents[1:]):
        raise ValueError('Archive holdouts must be separate from current-champion holdouts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = read_json(args.plan)
    validate_plan(plan)
    output = root / plan['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / 'gate.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'plan': plan, 'source_sha256': {name: sha256(root / name) for name in
                 (*plan['sources'], Path(__file__).name)}}
    manifest = output / 'experiment.json'
    if manifest.exists() and read_json(manifest)['identity'] != identity:
        raise ValueError('Gate identity changed; use a new predeclared attempt')
    if not manifest.exists():
        write_json(manifest, {'identity': identity, 'declared_at': now()})
    state_path = output / 'status.json'

    def status(stage, **fields):
        write_json(state_path, {'stage': stage, 'pid': os.getpid(), 'updated_at': now(), **fields})

    def verify():
        for name, digest in identity['source_sha256'].items():
            if sha256(root / name) != digest:
                raise ValueError(f'Source changed: {name}')
        for opponent in plan['opponents']:
            if sha256(root / opponent['checkpoint']) != opponent['sha256']:
                raise ValueError(f'Archived opponent changed: {opponent["name"]}')

    try:
        verify()
        upstream = Path(plan['upstream_project'])
        upstream_summary = upstream / plan['upstream_summary']
        status('waiting_for_training', status='waiting')
        while not upstream_summary.exists():
            upstream_state = read_json(upstream / plan['upstream_status'])
            if upstream_state.get('status') == 'failed':
                raise RuntimeError('Upstream experiment failed; no evaluation launched')
            if not args.wait:
                return
            time.sleep(30)  # Detached server process; no model calls or tokens.
        summary = read_json(upstream_summary)
        if not summary.get('strength_improvement_proven'):
            result = {'status': 'not_proven', 'reason': 'current_champion_gate_failed',
                      'install_allowed': False, 'soul_ten_proven': False, 'finished_at': now()}
            write_json(output / 'summary.json', result)
            status('completed', status='completed', archive_gate_pass=False,
                   reason='current_champion_gate_failed')
            return
        champion = next(x for x in plan['opponents'] if x['role'] == 'current_champion')
        if summary['baseline']['sha256'] != champion['sha256']:
            raise ValueError('Upstream champion differs from archive')
        selected = summary['candidate']
        source = upstream / selected['checkpoint']
        digest = selected['sha256']
        if sha256(source) != digest:
            raise ValueError('Upstream frozen candidate changed')
        candidate = root / 'model' / 'candidate_frozen.pt'
        candidate.parent.mkdir(exist_ok=True)
        if candidate.exists() and sha256(candidate) != digest:
            raise ValueError('Different candidate already frozen')
        if not candidate.exists():
            shutil.copyfile(source, candidate)
        selection = {'checkpoint': str(candidate.relative_to(root)), 'sha256': digest,
                     'upstream': str(source)}
        write_json(output / 'candidate.json', selection)
        alpha = comparison_alpha(plan['attempt'], len(plan['opponents']))
        results = {}
        for opponent in plan['opponents']:
            rows = []
            for seed in opponent['seeds']:
                verify()
                if sha256(candidate) != digest:
                    raise ValueError('Frozen candidate changed')
                status(f'{opponent["name"]}:{seed}', status='running')
                if opponent['role'] == 'current_champion':
                    progress_path = upstream / plan['upstream_confirm_dir'] / str(seed) / 'progress.json'
                else:
                    eval_dir = output / 'matches' / opponent['name'] / str(seed)
                    command = [sys.executable, 'battle_models.py', '--fast-forward-forced',
                               '--rollout-cuda-graphs', '--async-min-workers', '1', '--models',
                               f'candidate={candidate}', f'baseline={root / opponent["checkpoint"]}',
                               '--output-dir', str(eval_dir), '--seed', str(seed), '--game-mode', 'south',
                               '--rounds-per-wind', '3', '--max-steps', '3000', '--min-groups', str(plan['groups']),
                               '--max-groups', str(plan['groups']), '--batch-groups', '16',
                               '--env-workers', str(plan['workers']), '--device', 'cuda']
                    subprocess.run(command, cwd=root, check=True)
                    progress_path = eval_dir / 'progress.json'
                rows.append(assess_pair(read_json(progress_path), digest, opponent['sha256'], seed,
                                       plan['groups'], alpha, .02 if opponent['role'] == 'current_champion' else 0))
            results[opponent['name']] = rows
            write_json(output / 'results.json', results)
            if opponent['role'] == 'current_champion' and not all(row['pass'] for row in rows):
                # Predeclared futility stop only: never stop early to claim success.
                break
        verify()
        passed = promotion_decision(results, [x['name'] for x in plan['opponents']])
        result = {'status': 'pass' if passed else 'not_proven', 'candidate': selection,
                  'results': results, 'archive_gate_pass': passed, 'install_allowed': passed,
                  'application_compatibility_required': True, 'soul_ten_proven': False,
                  'scope': 'Predeclared archived opponents in the internal south-match simulator only',
                  'attempt': plan['attempt'], 'alpha_per_comparison': alpha, 'finished_at': now()}
        write_json(output / 'summary.json', result)
        status('completed', status='completed', archive_gate_pass=passed)
    except BaseException as error:
        status('failed', status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
