"""Capture reproducible bridge mismatches; never assess playing strength."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import compare_copilot_model as bridge
from battle_models import MatchTask, Policy, choose_actions
from model import load_checkpoint_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--matches', type=int, default=96)
    parser.add_argument('--seed', type=int, default=982_260_917)
    options = parser.parse_args()
    out = options.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'status.json').exists():
        raise ValueError('Use a new diagnostic output directory')
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = ROOT/'model/autodl_verified_20260916.pt'
    model, metadata = load_checkpoint_model(checkpoint, map_location=device)
    policy = Policy('champion', checkpoint, model.to(device).eval(), metadata['observation_key'])
    runtime = bridge.MortalRuntime(bridge.DEFAULT_MORTAL_MODEL)
    args = SimpleNamespace(game_mode='south', rounds_per_wind=3, max_steps=3000,
                           action_mode='greedy', temperature=1.)
    captured = []
    current = {}
    stats = bridge.empty_bridge_stats()
    original_feed = bridge.MortalSeat.feed
    original_map = bridge.reaction_to_local_action

    def traced_feed(seat, events, can_act):
        if not hasattr(seat, 'diagnostic_events'):
            seat.diagnostic_events = []
        for index, source in enumerate(events):
            event = dict(source)
            if event['type'] == 'tsumo' and event['actor'] != seat.seat:
                event['pai'] = '?'
            event['can_act'] = bool(can_act and index == len(events)-1)
            seat.diagnostic_events.append(event)
        original_feed(seat, events, can_act)

    def traced_map(reaction, observation, agent):
        action, issue = original_map(reaction, observation, agent)
        if issue and len(captured) < 24:
            runner = current['runner']
            game = runner._hand_game()
            seat = next(i for i,a in enumerate(game.agents) if a is agent)
            legal = np.flatnonzero(observation['action_mask'] > 0).tolist()
            captured.append(dict(seed=runner.task.seed, step=runner.item.steps, seat=seat,
                                 seat_models=list(runner.task.seat_models), issue=issue,
                                 reaction=reaction, legal_responses=[agent.action2response(a) for a in legal],
                                 hand=[bridge.tile_to_mjai(t) for t in agent.hand],
                                 agent_tile_wall=agent.tileWall, wall_last=game.WallLast,
                                 current_player=game.current_player, game_state=game.state,
                                 riichi=list(agent.isLiZhi),
                                 events=list(runner.controllers[seat].diagnostic_events)))
            bridge.save_json_atomic(out/'cases.json', captured)
        return action, issue

    bridge.MortalSeat.feed = traced_feed
    bridge.reaction_to_local_action = traced_map
    identity = dict(purpose='Bridge correctness only; fallback games are not strength evidence',
                    pid=os.getpid(), max_matches=options.matches, seed=options.seed,
                    stop_rule='Stop at max matches or after three nukidora mismatches have been captured',
                    champion_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    external_sha256=hashlib.sha256(bridge.DEFAULT_MORTAL_MODEL.read_bytes()).hexdigest(),
                    bridge_sha256=hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest())
    bridge.save_json_atomic(out/'experiment.json', identity)
    completed = 0
    errors = []
    try:
        for index in range(options.matches):
            lineup = (0,0,1) if index%2 == 0 else (0,1,1)
            shift = index%3
            seats = lineup[shift:]+lineup[:shift]
            task = MatchTask(options.seed+index, shift, seats)
            runner = bridge.CopilotGameRunner(task, runtime, args)
            current['runner'] = runner
            event = runner.advance()
            while event['kind'] == 'decision':
                names = list(event['observations'])
                actions = choose_actions(policy, [event['observations'][n] for n in names],
                                         device, 'greedy', 1., np.random.default_rng(task.seed))
                event = runner.advance(dict(zip(names, actions)))
            bridge.merge_bridge_stats(stats, event['bridge_stats'])
            if event['item'].error:
                errors.append(dict(seed=task.seed, error=event['item'].error))
            else:
                completed += 1
            status = dict(identity, status='running', attempted=index+1, completed_matches=completed,
                          errors=errors, captured_cases=len(captured),
                          captured_types=sorted({(c['reaction'] or {}).get('type','missing') for c in captured}),
                          fallbacks=stats['fallbacks'])
            bridge.save_json_atomic(out/'status.json',status)
            print(json.dumps({k:status[k] for k in ('attempted','completed_matches','captured_types','fallbacks')}),flush=True)
            if errors or sum((c['reaction'] or {}).get('type') == 'nukidora' for c in captured) >= 3:
                break
        status['status'] = 'completed'
        bridge.save_json_atomic(out/'status.json',status)
    finally:
        bridge.MortalSeat.feed = original_feed
        bridge.reaction_to_local_action = original_map


if __name__ == '__main__':
    main()
