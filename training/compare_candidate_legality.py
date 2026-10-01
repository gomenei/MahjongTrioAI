"""Boundary comparisons with an isolated reference; not a playing-strength test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mahjong_env.tile import Wind
from training.rules_candidate_feature import CandidateFeatureAgent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh diagnostic output')
    sys.path.insert(0, str(args.reference_package.resolve()))
    from riichienv import GameRule, Phase, RiichiEnv

    results = []
    for preset in ('mjsoul', 'tenhou'):
        for left in range(5):
            reference = RiichiEnv(game_mode='3p-red-half', seed=987260917,
                                 rule=getattr(GameRule, 'default_' + preset)())
            reference.reset()
            # Advance the actual engine with legal discards/passes to establish
            # the wall counter. Assigning .wall changes tiles, NOT drawable_count.
            target_draws = 55 - left
            for _ in range(200):
                draws = sum(e['type'] == 'tsumo' for e in reference.mjai_log)
                if draws == target_draws:
                    break
                if reference.done() or draws > target_draws:
                    raise RuntimeError('Unexpected fixture progression')
                actions = {}
                for seat, obs in reference.get_observations(reference.active_players).items():
                    legals = obs.legal_actions()
                    passes = [a for a in legals if json.loads(a.to_mjai())['type'] == 'none']
                    discards = [a for a in legals if json.loads(a.to_mjai())['type'] == 'dahai']
                    actions[seat] = (passes or discards)[0]
                reference.step(actions)
            else:
                raise RuntimeError('Fixture progression safety limit')

            conditions = [(s, False, '东') for s in (900, 1000, 35000)]
            if left in (0, 1, 4):
                conditions += [(35000, r, d) for r in (False, True) for d in ('东', '北')]
            for score, riichi, draw in conditions:
                ref = reference.clone()
                # Synthetic legality boundary: hand/points/flags are explicit;
                # no claim that this substituted position is a recorded game.
                drawn = 108 if draw == '东' else 122
                hands = ref.hands
                hands[0] = [36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 120, 121, drawn]
                ref.hands = hands
                ref.current_player, ref.active_players = 0, [0]
                ref.drawn_tile, ref.phase = drawn, Phase.WaitAct
                ref.is_first_turn = False
                ref.riichi_declared = [riichi, False, False]
                ref.set_scores([score, 35000, 35000])
                types = {json.loads(a.to_mjai())['type']
                         for a in ref.get_observations([0])[0].legal_actions()}
                expected = dict(riichi='reach' in types, north='kita' in types)

                agent = CandidateFeatureAgent(Wind.East, 0)
                agent.request2obs('Wind 1')
                agent.request2obs('Deal 1筒 2筒 3筒 4筒 5筒 6筒 7筒 8筒 9筒 1条 2条 北 北')
                agent.scores[0], agent.isLiZhi[0] = score, riichi
                agent.tileWall = left + 1
                agent.turn_count[0], agent.discard_called = 1, True
                mask = agent.request2obs('Draw ' + draw)['action_mask']
                actual = dict(riichi=bool(mask[145:174].any()), north=bool(mask[174]))
                results.append(dict(preset=preset, remaining=left, score=score,
                                    declared=riichi, drawn=draw, reference=expected,
                                    candidate=actual, equal=actual == expected))

    binary = next((args.reference_package / 'riichienv').glob('_riichienv*.pyd'))
    result = dict(purpose='Independent legality comparison; not platform certification',
                  reference_package='riichienv==0.4.10', reference_binary_sha256=sha(binary),
                  fixture='Real legal wall progression followed by explicit synthetic hand/flags',
                  candidate_sources={p: sha(ROOT / p) for p in (
                      'training/rules_candidate_feature.py', 'training/rules_candidate_game.py')},
                  cases=results, all_equal=all(r['equal'] for r in results))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(cases=len(results), all_equal=result['all_equal'])))
    if not result['all_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
