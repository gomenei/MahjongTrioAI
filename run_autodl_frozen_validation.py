"""One fixed-size, independent validation of an already frozen averaged policy."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from run_autodl_training import BASE, ROOT, TRAIN_SOURCES, battle, read_json, sha256, write_json


RECIPE = "frozen_validation_v9"
CANDIDATE = Path("ppo_runs/average_v8_fix1/uniform_eight_frozen.pt")
CANDIDATE_SHA256 = "160c8c3bba2cf5a808be56eebd46b91828e4d193bcacd7a266a50e2efcf48895"
BASE_SHA256 = "dc1e6a850ce76e89ce5cd3820d36f4f8e2ac7ff5145e1eb3474290d4f58dafe5"
CONFIRM_SEEDS = (2_961_000_000, 3_071_000_000)
GROUPS = 6000


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_frozen():
    if sha256(ROOT / CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("Frozen candidate changed")
    if sha256(ROOT / BASE) != BASE_SHA256:
        raise ValueError("Frozen baseline changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-workers", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import fcntl
    output = ROOT / "campaigns" / RECIPE
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_frozen()
    identity = {
        "recipe": RECIPE, "candidate": CANDIDATE.as_posix(),
        "candidate_sha256": CANDIDATE_SHA256, "base_sha256": BASE_SHA256,
        "source_sha256": {name: sha256(ROOT / name) for name in
                          (*TRAIN_SOURCES, "run_autodl_training.py", Path(__file__).name)},
        "confirm_seeds": list(CONFIRM_SEEDS), "groups_per_lineup": GROUPS,
        "matches_per_set": GROUPS * 6, "total_matches": GROUPS * 12,
        "promotion_gate": "both independent sets must pass unchanged assess_report criteria",
        "protocol": "one fixed candidate, fixed sample size, no optional stopping or reselection",
        "prior_result": "average_v8_fix1 remains NOT_PROVEN on its original 12000 matches",
        "purpose": "increase precision using entirely new independent walls; do not pool old tests",
        "after_failure": "close this validation; do not repeat new seeds until it passes",
        "env_workers": args.env_workers, "device": args.device, "keep_server_running": True,
    }
    manifest = output / "experiment.json"
    if manifest.exists() and read_json(manifest)["identity"] != identity:
        raise ValueError("Experiment identity changed")
    if not manifest.exists():
        write_json(manifest, {"identity": identity, "declared_at": now()})
    state_path = output / "status.json"
    state = read_json(state_path) if state_path.exists() else {}
    if state.get("status") == "completed":
        return
    eval_dir = Path("battle_results") / RECIPE
    try:
        results = []
        for seed in CONFIRM_SEEDS:
            verify_frozen()
            state.update(status="running", stage=f"confirm:{seed}", pid=os.getpid(), updated_at=now())
            write_json(state_path, state)
            results.append(battle(args, CANDIDATE, eval_dir / "confirm" / str(seed), seed, GROUPS))
        verify_frozen()
        proven = len(results) == 2 and all(row["pass"] for row in results)
        summary = {"status": "pass" if proven else "not_proven",
                   "candidate": {"checkpoint": CANDIDATE.as_posix(), "sha256": CANDIDATE_SHA256},
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
