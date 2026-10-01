"""Bounded rank-focused experiment with fresh, predeclared holdouts."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from run_autodl_training import (
    BASE, HISTORY, ROOT, TRAIN_SOURCES, battle, candidates, read_json, run,
    sha256, train_command, write_json,
)


RECIPES = {
    "rank_v3": {"seed": 70_260_916, "actor_lr": "1e-6", "screen_seed": 1_131_000_000,
                "confirm_seeds": [1_241_000_000, 1_351_000_000]},
    "rank_v4": {"seed": 90_260_916, "actor_lr": "5e-6", "screen_seed": 1_431_000_000,
                "confirm_seeds": [1_541_000_000, 1_651_000_000]},
    "rank_v5": {"seed": 110_260_916, "actor_lr": "5e-6", "screen_seed": 1_731_000_000,
                "confirm_seeds": [1_841_000_000, 1_951_000_000], "policy_temperature": "0.5"},
    "rank_v6": {"seed": 130_260_916, "actor_lr": "5e-6", "screen_seed": 2_031_000_000,
                "confirm_seeds": [2_141_000_000, 2_251_000_000], "gae_lambda": "0.95"},
    "rank_v7": {"seed": 150_260_916, "actor_lr": "5e-6", "screen_seed": 2_331_000_000,
                "confirm_seeds": [2_441_000_000, 2_551_000_000], "gae_lambda": "0.95"},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def command_for(args, smoke=False):
    _, command = train_command(args, smoke=smoke)
    changes = {
        "--output-dir": str(args.run_dir) + ("_smoke" if smoke else ""),
        "--actor-lr": RECIPES[args.recipe]["actor_lr"],
        "--hands-per-update": "512",
        "--gae-lambda": RECIPES[args.recipe].get("gae_lambda", "1.0"),
        "--snapshot-interval": "1" if smoke else "12",
        "--checkpoint-minutes": "0" if smoke else "30",
        "--max-failure-rate": "0",
        "--ppo-epochs": "4",
    }
    for flag, value in changes.items():
        command[command.index(flag) + 1] = value
    command += ["--reward-scale", "32000", "--rank-reward-weight", "2"]
    if "policy_temperature" in RECIPES[args.recipe]:
        command += ["--policy-temperature", RECIPES[args.recipe]["policy_temperature"]]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--env-workers", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="rank_v3")
    args = parser.parse_args()
    if not 0 < args.hours <= 4:
        parser.error("hours must be in (0,4]")
    recipe = RECIPES[args.recipe]
    args.seed, args.hands_per_update = recipe["seed"], 512
    args.run_dir = Path("ppo_runs") / f"{args.recipe}_{args.seed}"
    eval_dir = Path("battle_results") / f"{args.recipe}_{args.seed}"
    output = ROOT / "campaigns" / args.recipe
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = command_for(args)
    identity = {
        "recipe": args.recipe, "command": command,
        "base_sha256": sha256(ROOT / BASE), "history_sha256": sha256(ROOT / HISTORY),
        "source_sha256": {name: sha256(ROOT / name) for name in
                          (*TRAIN_SOURCES, "run_autodl_training.py", Path(__file__).name)},
        "screen_seed": recipe["screen_seed"], "screen_groups_per_lineup": 256,
        "confirm_seeds": recipe["confirm_seeds"], "confirm_groups_per_lineup": 1000,
        "keep_server_running": True,
    }
    manifest_path = output / "experiment.json"
    if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
        raise ValueError("Experiment identity changed; use a new isolated experiment")
    if not manifest_path.exists():
        write_json(manifest_path, {"identity": identity, "declared_at": now()})
    state_path = output / "status.json"
    state = read_json(state_path) if state_path.exists() else {"completed_stages": []}
    if state.get("status") == "completed":
        return

    def status(stage, **extra):
        state.update(status="running", stage=stage, pid=os.getpid(), updated_at=now(), **extra)
        write_json(state_path, state)

    def complete(stage):
        state["completed_stages"].append(stage)
        status(stage)

    try:
        for stage, train_cmd in [("smoke", command_for(args, smoke=True)), ("train", command)]:
            if stage not in state["completed_stages"]:
                status(stage)
                run(train_cmd)
                complete(stage)
        selection_path = ROOT / eval_dir / "selection.json"
        if not selection_path.exists():
            status("screen")
            rows = []
            for candidate in candidates(ROOT / args.run_dir):
                relative = candidate.relative_to(ROOT)
                metrics = battle(args, relative, eval_dir / "screen" / candidate.stem,
                                 recipe["screen_seed"], 256)
                rows.append({"checkpoint": relative.as_posix(), "sha256": sha256(candidate), **metrics})
                write_json(ROOT / eval_dir / "screening.json", rows)
            eligible = [row for row in rows if row["valid"] and row["rank_improvement"] > 0
                        and row["deal_in_rate_change"] <= 0.01]
            selected = min(eligible, key=lambda row: row["average_rank"]) if eligible else None
            write_json(selection_path, {"selected": selected, "frozen_at": now(),
                                       "base_sha256": identity["base_sha256"],
                                       "confirm_seeds": recipe["confirm_seeds"]})
        selection = read_json(selection_path)
        selected = selection["selected"]
        rows = []
        if selected is not None:
            candidate = Path(selected["checkpoint"])
            if sha256(ROOT / candidate) != selected["sha256"]:
                raise ValueError("Frozen candidate changed")
            if sha256(ROOT / BASE) != selection["base_sha256"]:
                raise ValueError("Frozen baseline changed")
            for seed in selection["confirm_seeds"]:
                status(f"confirm:{seed}")
                rows.append(battle(args, candidate, eval_dir / "confirm" / str(seed), seed, 1000))
        proven = bool(rows) and all(row["pass"] for row in rows)
        result = {"status": "pass" if proven else "not_proven", "candidate": selected,
                  "results": rows, "strength_improvement_proven": proven,
                  "automatically_installed": False, "finished_at": now()}
        write_json(ROOT / eval_dir / "promotion.json", result)
        write_json(output / "summary.json", result)
        state.update(status="completed", stage=None, updated_at=now(),
                     strength_improvement_proven=proven)
        write_json(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at=now())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
