"""Compare post-riichi concealed-kan boundary positions against a separate engine."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mahjong_env.feature import parse_tile_str
from mahjong_env.tile import Wind
from mahjong_env.util import _tile_to_34_index
from training.rules_candidate_feature import CandidateFeatureAgent
from training.rules_candidate_match import tile_kinds
from mahjong_env.tile import Tile, Suit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--random-cases', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output')
    sys.path.insert(0, str(args.reference_package.resolve()))
    from riichienv import GameRule, RiichiEnv
    cases = [
        ('valid', '1筒 1筒 1筒 2筒 3筒 4筒 5条 6条 7条 8条 9条 东 东', '1筒'),
        ('changed_waits', '4筒 4筒 4筒 5筒 6筒 7筒 2条 3条 4条 3筒 3筒 1筒 2筒', '4筒'),
        ('stored_quad', '1筒 1筒 1筒 1筒 2筒 3筒 4筒 5筒 6筒 7筒 8筒 9筒 东', '北'),
        ('red_draw', '5筒 5筒 5筒 1筒 2筒 3筒 1条 2条 3条 4条 5条 东 东', '红5筒'),
    ]
    if not 0 <= args.random_cases <= 1000:
        parser.error('--random-cases must be between 0 and 1000')
    rng, kinds, generated = random.Random(994260917), tile_kinds(), 0
    for attempt in range(20000):
        if generated == args.random_cases:
            break
        drawn = rng.choice(kinds)
        hand = [drawn]*3
        for _ in range(2):
            suit, start = rng.choice((Suit.Pinzu, Suit.Souzu)), rng.randrange(1, 8)
            hand += [Tile(suit, start+i) for i in range(3)]
        pair = rng.choice(kinds)
        hand += [pair]*2
        suit, start = rng.choice((Suit.Pinzu, Suit.Souzu)), rng.randrange(1, 9)
        hand += [Tile(suit, start), Tile(suit, start+1)]
        if hand.count(drawn) != 3 or any(hand.count(t) > 4 for t in hand):
            continue
        # Allocate the red five when all four copies are present. Otherwise
        # this synthetic hand uses ordinary copies; no physical tile repeats.
        red_kinds = set()
        strings = []
        all_tiles = hand + [drawn]
        for t in all_tiles:
            kind = _tile_to_34_index(t)
            if kind in (13,22) and all_tiles.count(t) == 4 and kind not in red_kinds:
                strings.append(str(Tile(t.suit,t.value,is_red=True)))
                red_kinds.add(kind)
            else:
                strings.append(str(t))
        cases.append((f'generated_{attempt}', ' '.join(strings[:-1]), strings[-1]))
        generated += 1
    if generated != args.random_cases:
        raise RuntimeError('Could not construct requested boundary cases')
    results = []
    for preset in ('mjsoul', 'tenhou'):
        for name, raw, draw in cases:
            used, physical = set(), []
            for value in raw.split() + [draw]:
                t = parse_tile_str(value)
                kind = _tile_to_34_index(t)
                copies = [0] if t.is_red else ([1,2,3] if kind in (13,22) else range(4))
                tid = next(kind*4+c for c in copies if kind*4+c not in used)
                used.add(tid)
                physical.append(tid)
            ref = RiichiEnv(game_mode='3p-red-half', seed=992260917,
                           rule=getattr(GameRule, 'default_' + preset)())
            ref.reset()
            hands = ref.hands
            hands[0] = physical
            ref.hands, ref.drawn_tile = hands, physical[-1]
            ref.riichi_declared, ref.is_first_turn = [True, False, False], False
            legals = [json.loads(a.to_mjai()) for a in ref.get_observations([0])[0].legal_actions()]
            reference = any(a['type'] == 'ankan' for a in legals)

            agent = CandidateFeatureAgent(Wind.East, 0)
            agent.request2obs('Wind 1')
            agent.request2obs('Deal ' + raw)
            agent.turn_count[0], agent.discard_called = 1, True
            agent.isLiZhi[0] = True
            mask = agent.request2obs('Draw ' + draw)['action_mask']
            actual = bool(mask[87:116].any())
            results.append(dict(preset=preset, case=name, reference_ankan=reference,
                                candidate_ankan=actual, equal=reference == actual,
                                hand=raw, drawn=draw,
                                reference_legal_actions=legals))
    binary = next((args.reference_package/'riichienv').glob('_riichienv*.pyd'))
    result = dict(purpose='Synthetic independent legality comparison, not platform certification',
                  reference_package='riichienv==0.4.10',
                  fixture_seed=994260917, generated_per_preset=generated,
                  reference_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  candidate_sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                      'training/rules_candidate_feature.py','training/rules_candidate_kan.py')},
                  cases=results, all_equal=all(r['equal'] for r in results))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(results),all_equal=result['all_equal'])))
    if not result['all_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
