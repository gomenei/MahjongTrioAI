"""Run two training seeds, screening and held-out confirmation without SSH."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_autodl_training import BASE, ROOT, read_json, sha256, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260916, 50260916])
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--env-workers", type=int, default=20)
    parser.add_argument("--hands-per-update", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("campaigns/league_v2"))
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    # Prevent a repeated launch from consuming a second GPU job concurrently.
    import fcntl
    lock = (output / "campaign.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Campaign is already running")
    state_path = output / "status.json"
    identity = {"seeds": args.seeds, "hours_per_seed": args.hours,
                "hands_per_update": args.hands_per_update,
                "base_sha256": sha256(ROOT / BASE)}
    state = read_json(state_path) if state_path.exists() else {
        "identity": identity, "started_at": utc_now(), "completed_stages": [],
    }
    if state["identity"] != identity:
        raise SystemExit("Existing campaign settings differ; use a new campaign directory")
    if state.get("status") == "completed":
        print("Campaign is already complete")
        return
    previous_child = state.get("child_pid")
    if previous_child:
        command_path = Path(f"/proc/{previous_child}/cmdline")
        if command_path.exists() and b"run_autodl_training.py" in command_path.read_bytes():
            raise SystemExit("A previous stage is still running; refusing duplicate training")
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    state.update(pid=os.getpid(), status="running", updated_at=utc_now())
    write_json(state_path, state)
    try:
        for seed in args.seeds:
            for stage in ("train", "screen", "confirm"):
                key = f"{seed}:{stage}"
                if key in state["completed_stages"]:
                    continue
                selection = ROOT / "battle_results" / f"league_v2_{seed}" / "selection.json"
                # Recover a screen that finished immediately before interruption.
                if stage == "screen" and selection.exists():
                    state["completed_stages"].append(key)
                    write_json(state_path, state)
                    continue
                if stage == "confirm" and selection.exists() and read_json(selection)["selected"] is None:
                    state["completed_stages"].append(key)
                    write_json(state_path, state)
                    continue
                command = [sys.executable, "run_autodl_training.py", stage,
                           "--seed", str(seed), "--hours", str(args.hours),
                           "--env-workers", str(args.env_workers),
                           "--hands-per-update", str(args.hands_per_update)]
                state.update(current_stage=key, stage_started_at=utc_now(), updated_at=utc_now())
                write_json(state_path, state)
                print(f"{utc_now()} starting {key}", flush=True)
                with (output / f"{seed}_{stage}.log").open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT)
                    state["child_pid"] = process.pid
                    write_json(state_path, state)
                    while process.poll() is None:
                        time.sleep(10)
                        state["updated_at"] = utc_now()
                        write_json(state_path, state)
                    if process.returncode:
                        raise RuntimeError(f"{key} exited with code {process.returncode}; see stage log")
                state["completed_stages"].append(key)
                write_json(state_path, state)
        outcomes = []
        for seed in args.seeds:
            promotion = ROOT / "battle_results" / f"league_v2_{seed}" / "promotion.json"
            outcomes.append({"seed": seed, **read_json(promotion)} if promotion.exists()
                            else {"seed": seed, "status": "no_candidate"})
        passed = [row for row in outcomes if row["status"] == "pass"]
        summary = {"outcomes": outcomes, "strength_improvement_proven": bool(passed),
                   "baseline": BASE.as_posix(), "finished_at": utc_now()}
        if passed:
            strongest = max(passed, key=lambda row: min(
                result["rank_improvement"] for result in row["results"]))
            champion = ROOT / strongest["candidate"]["checkpoint"]
            destination = output / "verified_candidate.pt"
            shutil.copy2(champion, destination)
            summary.update(candidate=str(destination.relative_to(ROOT)),
                           candidate_sha256=sha256(destination), training_seed=strongest["seed"])
        write_json(output / "summary.json", summary)
        state.update(status="completed", current_stage=None, updated_at=utc_now(),
                     strength_improvement_proven=bool(passed))
        write_json(state_path, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at=utc_now())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
