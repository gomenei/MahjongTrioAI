"""Replay previously failing evaluation cases for simulator regression only."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from battle_models import MatchTask, Policy, run_tasks_serial
from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.tile import MeldType
from model import load_checkpoint_model
from run_autodl_training import BASE
from training.cuda_rollout import ROLLOUT_INFERENCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    ROLLOUT_INFERENCE.enabled = True
    policies = []
    for name, path in (("candidate", args.candidate), ("baseline", BASE)):
        model, metadata = load_checkpoint_model(path, map_location=device)
        policies.append(Policy(name, path, model.to(device).eval(), metadata["observation_key"]))
    cases = [(851000032, (0, 0, 1)), (851000888, (0, 0, 1)),
             (861000600, (0, 1, 1)), (961000167, (0, 0, 1))]
    tasks = [MatchTask(seed, rotation, tuple(lineup[(seat + rotation) % 3] for seat in range(3)))
             for seed, lineup in cases for rotation in range(3)]
    overflows = []
    original = ThreePlayerMahjong._plusdora

    def checked_plusdora(game):
        if game.numdora >= 5:
            overflows.append({"numdora": game.numdora, "current_player": game.current_player,
                              "state": game.state,
                              "kan_counts": [sum(m.type in (MeldType.OpenKan, MeldType.ClosedKan)
                                                 for m in packs) for packs in game.packs],
                              "packs": [[(m.type.name, str(m.taken_tile)) for m in packs]
                                        for packs in game.packs]})
        return original(game)

    ThreePlayerMahjong._plusdora = checked_plusdora
    options = SimpleNamespace(game_mode="south", rounds_per_wind=3, action_mode="greedy",
                              temperature=1.0, max_steps=3000)
    results = run_tasks_serial(tasks, policies, options, device, np.random.default_rng(2))
    report = {"purpose": "Regression cases only; previously inspected holdout data, not new strength evidence",
              "overflows": overflows,
              "results": [{"seed": item.task.seed, "rotation": item.task.rotation,
                           "error": item.error, "done": item.game.done,
                           "steps": item.steps, "scores": item.game.scores,
                           "invalid": "非法动作" in item.game.result_message}
                          for item in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
