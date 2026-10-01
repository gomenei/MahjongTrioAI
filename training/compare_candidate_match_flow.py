"""Cross-check terminal win fixtures against an isolated RiichiEnv installation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.rules_candidate_match import CandidateSouthMatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_package.resolve()))
    from riichienv import GameRule, RiichiEnv

    cases = [
        ('south3_below_target', 5, [33000, 39000, 33000], 0, 1, 0),
        ('south3_39900', 5, [34700, 37000, 33300], 0, 1, 0),
        ('south3_40000', 5, [34800, 37000, 33200], 0, 1, 0),
        ('south3_dealer_below_target', 5, [36000, 37000, 32000], 2, 0, 0),
        ('south3_dealer_at_target', 5, [36000, 36700, 32300], 2, 0, 0),
        ('south3_dealer_tied_behind', 5, [40000, 32700, 32300], 2, 1, 0),
        ('west1_non_dealer_at_target', 6, [34000, 34800, 36200], 1, 2, 0),
        ('west1_dealer_not_first', 6, [29000, 45000, 31000], 0, 2, 0),
        ('west3_below_target', 8, [33000, 39000, 33000], 0, 1, 0),
        ('east1_non_dealer_resets_honba', 0, [35000, 35000, 35000], 1, 2, 2),
    ]
    # Closed white dragon + pure straight: three han, forty fu, no dora.
    winning_hand = ['1p','2p','3p','4p','5p','6p','7p','8p','2s','2s','P','P','P']
    other_hands = [
        ['1m','1m','9m','9m','1s','3s','4s','5s','6s','7s','8s','9s','N'],
        ['E','E','E','S','S','S','W','W','W','F','F','C','C'],
    ]
    results = []
    for name, round_index, scores, winner, discarder, honba in cases:
        dealer = round_index % 3
        reference = RiichiEnv(game_mode='3p-red-half', rule=GameRule.default_mjsoul())
        hands = [None]*3
        hands[winner] = winning_hand
        for seat, hand in zip([s for s in range(3) if s != winner], other_hands):
            hands[seat] = hand
        reference.apply_event({'type':'start_game'})
        reference.apply_event(dict(type='start_kyoku', bakaze=['E','S','W'][round_index//3],
                                   kyoku=round_index%3+1, honba=honba, kyotaku=0,
                                   oya=dealer, scores=scores, tehais=hands, dora_marker='E'))
        reference.apply_event(dict(type='tsumo', actor=discarder, pai='9p'))
        observation = reference.observe_event(dict(type='dahai', actor=discarder,
                                                   pai='9p', tsumogiri=True), winner)
        action = next(a for a in observation.legal_actions()
                      if json.loads(a.to_mjai())['type']=='hora')
        reference.step({winner:action})

        candidate = CandidateSouthMatch()
        candidate.reset(986260917)
        candidate.round_index, candidate.dealer = round_index, dealer
        candidate.scores, candidate.honba = scores.copy(), honba
        candidate._start_hand()
        game = candidate.game
        game.winner, game.current_player, game.win_by = winner, discarder, '荣和'
        game.fans[winner], game.fus[winner] = 3, 40
        game.result_message = 'fixed terminal win fixture'
        candidate._settle_hand()
        ref = dict(done=reference.done(), scores=reference.scores(),
                   round_index=reference.round_wind*3+reference.kyoku_idx,
                   dealer=reference.oya, honba=reference.honba,
                   pending_next_round=reference.needs_initialize_next_round)
        actual = dict(done=candidate.done, scores=candidate.scores,
                      round_index=candidate.round_index, dealer=candidate.dealer,
                      honba=candidate.honba)
        # If the reference defers initializing the next hand, preserve raw state
        # and compare only finality/scores until an explicit step resolves it.
        equal = ref['done']==actual['done'] and ref['scores']==actual['scores']
        if not ref['done'] and not ref['pending_next_round']:
            equal &= all(ref[key]==actual[key] for key in ('round_index','dealer','honba'))
        results.append(dict(name=name, reference=ref, candidate=actual, equal=equal))
    binary = next((args.reference_package/'riichienv').glob('_riichienv*.pyd'))
    result = dict(purpose='Independent implementation comparison; not platform certification',
                  reference_package='riichienv==0.4.10',
                  reference_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256((ROOT/'training/rules_candidate_match.py').read_bytes()).hexdigest(),
                  cases=results, all_equal=all(r['equal'] for r in results))
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
