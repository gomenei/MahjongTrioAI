"""Train the next generation from the installed champion, then test against it."""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from run_autodl_average_campaign import average_checkpoints
from run_autodl_rank_campaign import command_for
from run_autodl_training import BASE, ROOT, TRAIN_SOURCES, assess_report, read_json, run, sha256, write_json


RECIPE = "generation_v10"
CHAMPION = Path("model/autodl_verified_20260916.pt")
CHAMPION_SHA256 = "160c8c3bba2cf5a808be56eebd46b91828e4d193bcacd7a266a50e2efcf48895"
TRAIN_SEEDS = (170_260_917, 190_260_917)
CONFIRM_SEEDS = (3_181_000_000, 3_291_000_000)
CONFIRM_GROUPS = 6000
EXPORT_SMOKE_SEED = 3_401_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def training_command(seed, workers, device, smoke=False):
    args = SimpleNamespace(recipe="rank_v7", seed=seed, hours=2.0, hands_per_update=512,
                           env_workers=workers, device=device,
                           run_dir=Path("ppo_runs") / f"{RECIPE}_{seed}")
    command = command_for(args, smoke=smoke)
    command[command.index("--base-model") + 1] = CHAMPION.as_posix()
    # The fixed opponent and KL reference are the new champion. Keep the old
    # six-hour model in the initial history pool for opponent diversity.
    command[command.index("--opponent-models") + 1] = BASE.as_posix()
    return command


def time_checkpoints(run_dir):
    """Take the first saved update at each fixed time, ignoring resume snapshots."""
    result = []
    for minute in (30, 60, 90, 120):
        matches = sorted((run_dir / "checkpoints").glob(f"elapsed_{minute:04d}min_update_*.pt"))
        if not matches:
            raise ValueError(f"Missing predeclared {minute}-minute snapshot: {run_dir}")
        result.append(matches[0])
    return result


def validate_history(path, smoke=False):
    history = read_json(path)
    if not history or (smoke and len(history) != 3):
        raise ValueError("Missing or incomplete training history")
    failures = sum(row["rollout"]["failed_hands"] for row in history)
    finite = all(math.isfinite(v) for row in history for v in row["ppo"].values()
                 if isinstance(v, (int, float)))
    if failures or not finite:
        raise ValueError("Training integrity check failed")
    return {"updates": len(history), "matches": history[-1]["total_hands"],
            "failed": failures, "finite_ppo": finite}


def evaluate(args, candidate, output, seed, groups):
    run([sys.executable, "battle_models.py", "--fast-forward-forced", "--rollout-cuda-graphs",
         "--async-min-workers", "1", "--models", f"candidate={candidate}",
         f"baseline={CHAMPION.as_posix()}", "--output-dir", str(output), "--seed", str(seed),
         "--game-mode", "south", "--rounds-per-wind", "3", "--max-steps", "3000",
         "--min-groups", str(groups), "--max-groups", str(groups), "--batch-groups", "16",
         "--env-workers", str(args.env_workers), "--device", args.device])
    report = read_json(ROOT / output / "battle_report.json")
    progress = read_json(ROOT / output / "progress.json")
    rows = {row["model_name"]: row for row in report["rankings"]}
    if Path(rows["baseline"]["checkpoint"]).resolve() != (ROOT / CHAMPION).resolve():
        raise ValueError("Evaluation used a different baseline")
    if Path(rows["candidate"]["checkpoint"]).resolve() != (ROOT / candidate).resolve():
        raise ValueError("Evaluation used a different candidate")
    errors = sum(len(x.get("errors", [])) for x in progress["lineups"].values())
    return assess_report(report, groups, errors)


def main(*, recipe=RECIPE, training_seeds=TRAIN_SEEDS, confirm_seeds=CONFIRM_SEEDS,
         export_smoke_seed=EXPORT_SMOKE_SEED, command_builder=training_command,
         extra_identity=None, extra_sources=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-workers", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import fcntl
    import torch
    torch.set_num_threads(4)
    output = ROOT / "campaigns" / recipe
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha256(ROOT / CHAMPION) != CHAMPION_SHA256:
        raise ValueError("Installed champion copy changed")
    identity = {
        "recipe": recipe, "champion": CHAMPION.as_posix(), "champion_sha256": CHAMPION_SHA256,
        "historical_opponent_sha256": sha256(ROOT / BASE), "train_seeds": list(training_seeds),
        "train_commands": [command_builder(s, args.env_workers, args.device) for s in training_seeds],
        "hours_per_seed": 2, "candidate_rule": "uniform mean of all eight fixed 30/60/90/120-minute snapshots",
        "candidate_count": 1, "confirm_seeds": list(confirm_seeds), "confirm_groups_per_lineup": CONFIRM_GROUPS,
        "confirm_matches_total": CONFIRM_GROUPS * 12,
        "promotion_gate": "both holdouts must pass unchanged criteria AGAINST INSTALLED CHAMPION",
        "export_smoke_seed": export_smoke_seed, "export_smoke_groups_per_lineup": 64,
        "export_smoke_gate": "validity only; no strength-based selection",
        "source_sha256": {name: sha256(ROOT / name) for name in
                          (*TRAIN_SOURCES, "run_autodl_training.py", "run_autodl_rank_campaign.py",
                           "run_autodl_average_campaign.py", Path(__file__).name, *extra_sources)},
        "no_optional_stopping_or_holdout_reselection": True, "keep_server_running": True,
    }
    if extra_identity is not None:
        identity['additional_protocol'] = extra_identity
    manifest = output / "experiment.json"
    if manifest.exists() and read_json(manifest)["identity"] != identity:
        raise ValueError("Experiment identity changed; use a new experiment")
    if not manifest.exists():
        write_json(manifest, {"identity": identity, "declared_at": now()})
    state_path = output / "status.json"
    state = read_json(state_path) if state_path.exists() else {"completed_stages": []}
    if state.get("status") == "completed":
        return

    def status(stage):
        state.update(status="running", stage=stage, pid=os.getpid(), updated_at=now())
        state.pop("error", None)
        write_json(state_path, state)

    eval_dir = Path("battle_results") / recipe
    candidate = Path("ppo_runs") / recipe / "uniform_eight_frozen.pt"
    selection_path = ROOT / eval_dir / "selection.json"
    try:
        for seed in training_seeds:
            for smoke in (True, False):
                stage = f"{seed}:" + ("smoke" if smoke else "train")
                if stage in state["completed_stages"]:
                    continue
                status(stage)
                command = command_builder(seed, args.env_workers, args.device, smoke)
                run(command)  # ppo_train resumes last.pt with optimizer and RNG state.
                run_dir = ROOT / command[command.index("--output-dir") + 1]
                metrics = validate_history(run_dir / "history.json", smoke)
                write_json(output / f"{stage.replace(':', '_')}.json", metrics)
                state["completed_stages"].append(stage)
                status(stage)
        if not selection_path.exists():
            status("average")
            sources = [p for seed in training_seeds
                       for p in time_checkpoints(ROOT / "ppo_runs" / f"{recipe}_{seed}")]
            ingredients = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "weight": 1 / 8}
                           for p in sources]
            validation = average_checkpoints(sources, ROOT / candidate, recipe=recipe)
            write_json(output / "validation.json", validation)
            write_json(selection_path, {"selected": {"checkpoint": candidate.as_posix(),
                       "sha256": sha256(ROOT / candidate)}, "frozen_at": now(),
                       "baseline_sha256": CHAMPION_SHA256, "ingredients": ingredients,
                       "confirm_seeds": list(confirm_seeds)})
        selection = read_json(selection_path)

        def verify_frozen():
            if sha256(ROOT / CHAMPION) != CHAMPION_SHA256:
                raise ValueError("Frozen champion changed")
            if sha256(ROOT / candidate) != selection["selected"]["sha256"]:
                raise ValueError("Frozen candidate changed")
            for item in selection["ingredients"]:
                if sha256(ROOT / item["path"]) != item["sha256"]:
                    raise ValueError("Averaging source changed")

        verify_frozen()
        if "export_smoke" not in state["completed_stages"]:
            status("export_smoke")
            metrics = evaluate(args, candidate, eval_dir / "smoke", export_smoke_seed, 64)
            write_json(output / "export_smoke.json", metrics)
            if not metrics["valid"]:
                raise ValueError("Export smoke integrity failed")
            state["completed_stages"].append("export_smoke")
            status("export_smoke")
        results = []
        for seed in confirm_seeds:
            verify_frozen()
            status(f"confirm:{seed}")
            results.append(evaluate(args, candidate, eval_dir / "confirm" / str(seed), seed, CONFIRM_GROUPS))
        verify_frozen()
        proven = len(results) == 2 and all(row["pass"] for row in results)
        summary = {"status": "pass" if proven else "not_proven", "candidate": selection["selected"],
                   "baseline": {"checkpoint": CHAMPION.as_posix(), "sha256": CHAMPION_SHA256},
                   "results": results, "strength_improvement_proven": proven,
                   "automatically_installed": False, "finished_at": now()}
        write_json(output / "summary.json", summary)
        write_json(ROOT / eval_dir / "promotion.json", summary)
        state.update(status="completed", stage=None, updated_at=now(), strength_improvement_proven=proven)
        write_json(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at=now())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
