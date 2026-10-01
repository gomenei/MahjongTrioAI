"""Independent exhaustive-draw settlement and continuation boundary fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mahjong_env.tile import Suit, Tile
from training.rules_candidate_match import CandidateSouthMatch


TENPAI = [
    [9,10,11,12,13,14,15,16,19,19,31,31,31],
    [0,0,0,8,8,8,18,19,20,21,22,23,27],
    [27,27,27,28,28,28,29,29,29,32,32,33,33],
]
NOTEN = [
    [9,10,11,12,13,14,15,16,19,19,31,31,33],
    [0,0,0,8,8,8,18,19,20,21,22,24,27],
    [27,27,27,28,28,28,29,29,32,32,33,33,30],
]


def tile(kind):
    return Tile((Suit.Manzu, Suit.Pinzu, Suit.Souzu, Suit.Honors)[kind // 9], kind % 9 + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new comparison output')
    sys.path.insert(0, str(args.reference_package.resolve()))
    from riichienv import GameRule, RiichiEnv

    cases = [
        ('east_draw_dealer_noten', 0, [35000]*3, [1], 2),
        ('east_draw_dealer_tenpai', 0, [35000]*3, [0], 2),
        ('south3_all_noten_west_entry', 5, [35000]*3, [], 0),
        ('south3_all_tenpai_repeat', 5, [35000]*3, [0,1,2], 1),
        ('south3_dealer_below_target', 5, [34000,34000,37000], [2], 0),
        ('south3_dealer_reaches_target', 5, [33000,34000,38000], [2], 0),
        ('south3_tied_dealer_behind', 5, [41000,26000,38000], [2], 0),
        ('west1_dealer_not_first_repeat', 6, [30000,44000,31000], [0], 2),
        ('west1_below_target', 6, [34000,37000,34000], [1], 2),
        ('west1_target_reached', 6, [34000,38000,33000], [1], 2),
        ('west3_all_noten_end', 8, [35000]*3, [], 0),
        ('east_two_tenpai_dealer_noten', 0, [35000]*3, [1,2], 1),
        ('east_two_tenpai_dealer_tenpai', 0, [35000]*3, [0,1], 1),
    ]
    results = []
    for name, round_index, scores, tenpai, honba in cases:
        dealer = round_index % 3
        ref = RiichiEnv(game_mode='3p-red-half', seed=989260917,
                       rule=GameRule.default_mjsoul())
        ref.reset(oya=dealer, honba=honba)
        ref.round_wind = round_index // 3
        # Reach the actual last draw using only legal discards and passes.
        for _ in range(200):
            draws = sum(e['type'] == 'tsumo' for e in ref.mjai_log)
            if draws == 55:
                break
            if ref.done() or draws > 55:
                raise RuntimeError('Unexpected wall progression')
            choices = {}
            for seat, obs in ref.get_observations(ref.active_players).items():
                aa = obs.legal_actions()
                pp = [a for a in aa if json.loads(a.to_mjai())['type'] == 'none']
                dd = [a for a in aa if json.loads(a.to_mjai())['type'] == 'dahai']
                choices[seat] = (pp or dd)[0]
            ref.step(choices)
        else:
            raise RuntimeError('Draw fixture safety limit')

        # Replace the closed hands to isolate final tenpai/score boundaries.
        # These are synthetic fixtures, not platform game records.
        kinds = [list(TENPAI[s] if s in tenpai else NOTEN[s]) for s in range(3)]
        physical, counts = [], {}
        for hand in kinds:
            ids = []
            for kind in hand:
                copy = counts.get(kind, 0)
                assert copy < 4
                ids.append(kind*4+copy)
                counts[kind] = copy+1
            physical.append(ids)
        drawn = 30*4 + counts.get(30, 0)
        physical[ref.current_player].append(drawn)
        ref.hands, ref.drawn_tile = physical, drawn
        ref.set_scores(scores)
        ref.riichi_declared = [False]*3
        ref.is_first_turn = False
        # Simple discards exclude nagashi mangan from these noten-payment cases.
        ref.discards = [[36], [40], [44]]
        last = next(a for a in ref.get_observations(ref.active_players)[ref.current_player].legal_actions()
                    if json.loads(a.to_mjai()).get('pai') == 'N'
                    and json.loads(a.to_mjai())['type'] == 'dahai')
        ref.step({ref.current_player: last})
        # Other players may get a response decision before draw settlement.
        if not any(e['type'] == 'ryukyoku' for e in ref.mjai_log):
            passes = {s: next(a for a in o.legal_actions()
                              if json.loads(a.to_mjai())['type'] == 'none')
                      for s, o in ref.get_observations(ref.active_players).items()}
            ref.step(passes)
        if not any(e['type'] == 'ryukyoku' for e in ref.mjai_log):
            raise RuntimeError('Reference did not settle an exhaustive draw')

        candidate = CandidateSouthMatch()
        candidate.reset(989260917)
        candidate.round_index, candidate.dealer = round_index, dealer
        candidate.scores, candidate.honba = scores.copy(), honba
        candidate._start_hand()
        candidate.game.hands = [[tile(k) for k in hand] for hand in kinds]
        candidate.game.packs = [[], [], []]
        candidate.game.winner = None
        candidate.game.result_message = '流局'
        candidate._settle_hand()
        draw_event = next(e for e in ref.mjai_log if e['type'] == 'ryukyoku')
        reference = dict(done=ref.done(), scores=ref.scores(),
                         round_index=ref.round_wind*3+ref.kyoku_idx,
                         dealer=ref.oya, honba=ref.honba)
        actual = dict(done=candidate.done, scores=candidate.scores,
                      round_index=candidate.round_index, dealer=candidate.dealer,
                      honba=candidate.honba)
        equal = all(reference[k] == actual[k] for k in ('done', 'scores'))
        if not ref.done():
            equal &= all(reference[k] == actual[k] for k in ('round_index', 'dealer', 'honba'))
        equal &= candidate.history[-1].tenpai == tenpai
        results.append(dict(name=name, expected_tenpai=tenpai,
                            candidate_tenpai=candidate.history[-1].tenpai,
                            reference_draw_event=draw_event,
                            reference=reference, candidate=actual, equal=equal))
    binary = next((args.reference_package/'riichienv').glob('_riichienv*.pyd'))
    result = dict(purpose='Independent exhaustive-draw comparison, not platform certification',
                  reference_package='riichienv==0.4.10',
                  reference_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  rules_version=CandidateSouthMatch.rules_version,
                  candidate_sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                     for p in ('training/rules_candidate_match.py',
                                               'training/rules_candidate_game.py',
                                               'training/rules_candidate_feature.py')},
                  cases=results, all_equal=all(r['equal'] for r in results))
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cases=len(results), all_equal=result['all_equal'])))
    if not result['all_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
