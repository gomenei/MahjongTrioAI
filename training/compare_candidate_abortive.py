"""Independent actual-action comparisons for abortive-draw settlement."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_copilot_model import tile_to_mjai
from mahjong_env.feature import parse_tile_str
from mahjong_env.tile import Meld, MeldType
from training.rules_candidate_match import CandidateSouthMatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    sys.path.insert(0, str(args.reference_package.resolve()))
    from riichienv import GameRule, RiichiEnv
    cases = []
    for reason in ('kyushu_kyuhai','suukansansen'):
        for round_index in (0,2,5,6,8):
            m = CandidateSouthMatch()
            m.reset(999260917)
            m.round_index, m.dealer = round_index, round_index % 3
            m.scores = [25000]*3
            m.scores[m.dealer] = 53000
            m.riichi_sticks, m.honba = 2, 2
            m._start_hand()
            g = m.game
            r = RiichiEnv(game_mode='3p-red-half',rule=GameRule.default_mjsoul())
            r.apply_event(dict(type='start_game'))
            if reason == 'kyushu_kyuhai':
                raw = ['1万 9万 1筒 9筒 1条 9条 东 南 西 白 白 发 中',
                       '2筒 2筒 2筒 3筒 3筒 3筒 4筒 4筒 4筒 5筒 5筒 5筒 6筒',
                       '2条 2条 2条 3条 3条 3条 4条 4条 4条 5条 5条 5条 6条']
                actor, drawn = 0, '北'
            else:
                raw = ['1万 1万 1万 1万 9万 9万 9万 9万 2筒 3筒 4筒 7条 8条',
                       '东 东 东 东 南 南 南 南 2条 3条 4条 5筒 6筒',
                       '1筒 1筒 2筒 2筒 3筒 3筒 4筒 4筒 5筒 6筒 6筒 7筒 8筒']
                actor, drawn = 1, '9条'
            hands = [[parse_tile_str(t) for t in h.split()] for h in raw]
            r.apply_event(dict(type='start_kyoku',bakaze=['E','S','W'][round_index//3],
                               kyoku=round_index%3+1,honba=2,kyotaku=2,oya=m.dealer,
                               scores=m.scores,tehais=[[tile_to_mjai(t) for t in h] for h in hands],
                               dora_marker='C'))
            g.hands = hands
            g.current_player, g.state, g.curTile = actor, 1, parse_tile_str(drawn)
            if reason == 'suukansansen':
                for seat, name in ((0,'1万'),(0,'9万'),(1,'东'),(1,'南')):
                    t = parse_tile_str(name)
                    r.apply_event(dict(type='ankan',actor=seat,consumed=[tile_to_mjai(t)]*4))
                    for _ in range(4): g.hands[seat].remove(t)
                    g.packs[seat].append(Meld(MeldType.ClosedKan,[t]*4,t))
            r.apply_event(dict(type='tsumo',actor=actor,pai=tile_to_mjai(g.curTile)))
            kind = 'ryukyoku' if reason == 'kyushu_kyuhai' else 'dahai'
            action = next(a for a in r.get_observations([actor])[actor].legal_actions()
                          if json.loads(a.to_mjai())['type']==kind
                          and (kind=='ryukyoku' or json.loads(a.to_mjai()).get('pai')=='9s'))
            r.step({actor:action})
            if reason == 'kyushu_kyuhai':
                g.abort_nine_terminals(actor)
            else:
                # Core response phase after the replacement tile has been discarded.
                # Reference above executes the discard and automatic no-claim check.
                g.state = 2
                g.step({g.agent_names[s]:176 for s in range(3) if s != actor})
            assert g.done and g.abortive_draw == reason
            m._settle_hand()
            expected = dict(scores=r.scores(),done=r.done(),dealer=r.oya,
                            round_index=r.round_wind*3+r.kyoku_idx,honba=r.honba,sticks=r.riichi_sticks)
            actual = dict(scores=m.scores,done=m.done,dealer=m.dealer,
                          round_index=m.round_index,honba=m.honba,sticks=m.riichi_sticks)
            cases.append(dict(reason=reason,round_index=round_index,reference=expected,candidate=actual,
                              equal=expected==actual,reference_draws=[e for e in r.mjai_log if e['type']=='ryukyoku']))
    result = dict(reference_package='riichienv==0.4.10',rules_version=CandidateSouthMatch.rules_version,
                  policy_action_contract=CandidateSouthMatch.policy_action_contract,
                  purpose='Rule comparison, not strength evidence',
                  candidate_sources={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (ROOT/'training').glob('rules_candidate_*.py')},
                  cases=cases,all_equal=all(c['equal'] for c in cases))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(cases),all_equal=result['all_equal'])))
    if not result['all_equal']: raise SystemExit(1)

if __name__ == '__main__': main()
