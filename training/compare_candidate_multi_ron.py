"""Execute real multi-ron claims and compare settlement against an isolated engine."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mahjong_env.feature import parse_tile_str
from compare_copilot_model import tile_to_mjai
from training.rules_candidate_ron_fixtures import discard, position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh comparison output')
    sys.path.insert(0, str(args.reference_package.resolve()))
    from riichienv import GameRule, RiichiEnv
    cases = []
    for dealer in range(3):
        for discarder in range(3):
            cases.append(dict(name=f'rotation_{dealer}_{discarder}', round_index=dealer,
                              discarder=discarder, honba=2, sticks=2))
    for claims in ([1], [2], [1,2]):
        cases.append(dict(name='declaration_ron_' + ''.join(map(str,claims)),
                          sticks=2, riichi=True, claims=claims))
    cases += [
        dict(name='only_later_claim', honba=2, sticks=2, claims=[2]),
        dict(name='south3_dealer_second_continues',round_index=5,discarder=0,scores=[55000,28000,22000]),
        dict(name='south3_both_children_finish',round_index=5,discarder=2),
        dict(name='west1_dealer_second_continues',round_index=6,discarder=1,scores=[22000,55000,28000]),
    ]
    results = []
    for spec in cases:
        options = dict(spec)
        name, riichi, claims = options.pop('name'), options.pop('riichi',False), options.pop('claims',None)
        candidate, hands, winners = position(**options)
        claims = winners if claims is None else claims
        ref = RiichiEnv(game_mode='3p-red-half',rule=GameRule.default_mjsoul())
        ref.apply_event(dict(type='start_game'))
        ref.apply_event(dict(type='start_kyoku',bakaze=['E','S','W'][candidate.round_index//3],
                             kyoku=candidate.round_number+1,honba=candidate.honba,
                             kyotaku=candidate.riichi_sticks,oya=candidate.dealer,
                             scores=candidate.scores,
                             tehais=[[tile_to_mjai(parse_tile_str(t)) for t in raw.split()] for raw in hands],
                             dora_marker='E'))
        thrower = candidate.game.current_player
        ref.apply_event(dict(type='tsumo',actor=thrower,pai='9p'))
        if riichi:
            ref.apply_event(dict(type='reach',actor=thrower))
        ref.apply_event(dict(type='dahai',actor=thrower,pai='9p',tsumogiri=True))
        actions = {}
        for seat, obs in ref.get_observations(winners).items():
            kind = 'hora' if seat in claims else 'none'
            actions[seat] = next(a for a in obs.legal_actions() if json.loads(a.to_mjai())['type'] == kind)
        ref.step(actions)

        discard(candidate,riichi=riichi)
        game = candidate.game
        game.step({game.agent_names[w]:175 if w in claims else 176 for w in winners})
        if '非法动作' in game.result_message:
            raise RuntimeError(game.result_message)
        candidate._settle_hand()
        expected = dict(done=ref.done(), scores=ref.scores(), dealer=ref.oya,
                        round_index=ref.round_wind*3+ref.kyoku_idx,honba=ref.honba,
                        sticks=ref.riichi_sticks)
        actual = dict(done=candidate.done,scores=candidate.scores,dealer=candidate.dealer,
                      round_index=candidate.round_index,honba=candidate.honba,
                      sticks=candidate.riichi_sticks)
        keys = ('done','scores','sticks') if ref.done() else tuple(expected)
        equal = all(expected[k] == actual[k] for k in keys)
        results.append(dict(name=name,reference=expected,candidate=actual,equal=equal,
                            candidate_winners=game.winners,candidate_fans=game.fans,candidate_fus=game.fus,
                            reference_wins=[e for e in ref.mjai_log if e['type']=='hora']))
    binary = next((args.reference_package/'riichienv').glob('_riichienv*.pyd'))
    result = dict(purpose='Independent multi-ron rule comparison; not strength evidence',
                  reference_package='riichienv==0.4.10',
                  reference_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  candidate_sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                      'training/rules_candidate_match.py','training/rules_candidate_game.py',
                      'training/rules_candidate_scoring.py','training/rules_candidate_metrics.py')},
                  cases=results,all_equal=all(r['equal'] for r in results))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(results),all_equal=result['all_equal'])))
    if not result['all_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
