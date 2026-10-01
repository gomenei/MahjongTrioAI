"""Portable, resumable south-match PPO experiment and held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
BASE = Path("ppo_runs/south_8h_from_u300/checkpoints/elapsed_0360min_update_001561.pt")
HISTORY = Path("ppo_runs/south_8h_from_u300/checkpoints/elapsed_0240min_update_001055.pt")
SCREEN_SEED = 731_000_000
CONFIRM_SEEDS = (841_000_000, 951_000_000)
TRAIN_SOURCES = ("ppo_train.py", "model.py", "training/parallel_env.py", "training/ppo_support.py",
                 "training/cuda_rollout.py", "battle_models.py",
                 "mahjong_env/game.py", "mahjong_env/feature.py", "mahjong_env/match.py",
                 "mahjong_env/tile.py", "mahjong_env/util.py", "mahjong_env/scoring.py")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    print("运行：", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def train_command(args, smoke=False):
    run_dir = Path("ppo_runs") / (f"league_v2_smoke_{args.seed}" if smoke else f"league_v2_{args.seed}")
    command = [sys.executable, "ppo_train.py", "--base-model", str(BASE),
               "--output-dir", str(run_dir), "--game-mode", "south", "--rounds-per-wind", "3",
               "--seed", str(args.seed), "--device", args.device,
               "--fast-forward-forced", "--rollout-cuda-graphs",
               "--async-min-workers", "1",
               "--updates", "3" if smoke else "100000",
               "--hands-per-update", "8" if smoke else str(args.hands_per_update),
               "--ppo-epochs", "2" if smoke else "4", "--minibatch-size", "512",
               "--actor-lr", "1e-6", "--critic-lr", "6e-5",
               "--gamma", "1.0", "--gae-lambda", "0.99", "--entropy-coef", "0.003",
               "--reference-kl-coef", "0.05", "--critic-warmup-updates", "1" if smoke else "10",
               "--snapshot-ratio", "0.75", "--snapshot-interval", "1" if smoke else "50",
               "--opponent-pool-size", "4", "--opponent-models", str(HISTORY),
               "--evaluation-interval", "0", "--max-steps", "3000",
               "--env-workers", str(args.env_workers), "--max-failure-rate", "0.01",
               "--checkpoint-interval", "1" if smoke else "0",
               "--checkpoint-minutes", "0" if smoke else "60",
               "--max-hours", "0" if smoke else str(args.hours)]
    return run_dir, command


def train(args, smoke=False):
    run_dir, command = train_command(args, smoke)
    manifest = ROOT / run_dir / "experiment.json"
    identity = {"recipe": "south_league_v2", "seed": args.seed, "smoke": smoke,
                "base_sha256": sha256(ROOT / BASE), "history_sha256": sha256(ROOT / HISTORY),
                "source_sha256": {name: sha256(ROOT / name) for name in TRAIN_SOURCES},
                "hands_per_update": 8 if smoke else args.hands_per_update,
                "hours": 0 if smoke else args.hours}
    if manifest.exists() and read_json(manifest)["identity"] != identity:
        raise ValueError("实验参数或起点权重已改变；请使用新seed建立独立实验")
    write_json(manifest, {"identity": identity, "command": command})
    run(command)


def candidates(run_dir):
    import torch

    by_update = {}
    for path in sorted(run_dir.glob("checkpoints/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        update = int(payload.get("ppo_update", 0))
        if update > 0:
            by_update[update] = path
    if not by_update:
        raise FileNotFoundError(f"没有训练后的候选权重：{run_dir}")
    return [by_update[update] for update in sorted(by_update)]


def battle(args, candidate, output, seed, groups):
    run([sys.executable, "battle_models.py", "--fast-forward-forced", "--rollout-cuda-graphs",
         "--async-min-workers", "1",
         "--models", f"candidate={candidate}", f"baseline={BASE}",
         "--output-dir", str(output), "--seed", str(seed),
         "--game-mode", "south", "--rounds-per-wind", "3", "--max-steps", "3000",
         "--min-groups", str(groups), "--max-groups", str(groups),
         "--batch-groups", "16", "--env-workers", str(args.env_workers), "--device", args.device])
    report = read_json(ROOT / output / "battle_report.json")
    progress = read_json(ROOT / output / "progress.json")
    errors = sum(len(item.get("errors", [])) for item in progress["lineups"].values())
    metrics = assess_report(report, groups, errors)
    return metrics


def assess_report(report, groups, errors=0):
    rows = {row["model_name"]: row for row in report["rankings"]}
    candidate, baseline = rows["candidate"], rows["baseline"]
    pair = next(row for row in report["pairwise"]
                if {row["model_a"], row["model_b"]} == {"candidate", "baseline"})
    sign = 1 if pair["model_a"] == "candidate" else -1
    low = pair["point_difference_ci95_low"] if sign == 1 else -pair["point_difference_ci95_high"]
    rank_gain = baseline["average_rank"] - candidate["average_rank"]
    deal_in_change = candidate["deal_in_rate"] - baseline["deal_in_rate"]
    utility_low = (pair["utility_difference_ci95_low"] if sign == 1
                   else -pair["utility_difference_ci95_high"])
    valid = (report["completed"] and report["game_mode"] == "south" and errors == 0
             and candidate["invalid_actions"] == baseline["invalid_actions"] == 0
             and pair["paired_seed_groups"] >= groups * 2)
    return {"valid": bool(valid), "errors": errors,
            "rank_improvement": rank_gain, "point_ci95_low": low,
            "rank_utility_ci95_low": utility_low,
            "deal_in_rate_change": deal_in_change,
            "average_rank": candidate["average_rank"],
            "pass": bool(valid and rank_gain >= 0.02 and low > 0 and utility_low > 0
                         and deal_in_change <= 0.01)}


def screen(args):
    output = Path("battle_results") / f"league_v2_{args.seed}"
    if (ROOT / output / "selection.json").exists():
        raise ValueError("候选已冻结；请直接运行confirm，避免接触终测后重新选优")
    rows = []
    for candidate in candidates(ROOT / "ppo_runs" / f"league_v2_{args.seed}"):
        relative = candidate.relative_to(ROOT)
        metrics = battle(args, relative, output / "screen" / candidate.stem, SCREEN_SEED, args.screen_groups)
        rows.append({"checkpoint": relative.as_posix(), "sha256": sha256(candidate), **metrics})
        write_json(ROOT / output / "screening.json", rows)
    eligible = [row for row in rows if row["valid"] and row["rank_improvement"] > 0
                and row["deal_in_rate_change"] <= 0.01]
    if not eligible:
        write_json(ROOT / output / "selection.json", {"selected": None, "status": "no_candidate"})
        print("筛选没有找到满足条件的候选，继续使用原模型。")
        return
    winner = min(eligible, key=lambda row: row["average_rank"])
    write_json(ROOT / output / "selection.json", {"selected": winner, "status": "frozen",
                                                   "base_sha256": sha256(ROOT / BASE),
                                                   "confirm_seeds": CONFIRM_SEEDS})
    print("已冻结终测候选：", winner["checkpoint"])


def confirm(args):
    output = Path("battle_results") / f"league_v2_{args.seed}"
    selection = read_json(ROOT / output / "selection.json")
    selected = selection["selected"]
    if selected is None:
        print("没有通过筛选的候选，继续使用原模型。")
        return
    candidate = Path(selected["checkpoint"])
    if sha256(ROOT / candidate) != selected["sha256"]:
        raise ValueError("冻结后的候选权重已改变，拒绝混用终测结果")
    if sha256(ROOT / BASE) != selection["base_sha256"]:
        raise ValueError("筛选后的基础模型已改变，拒绝混用终测结果")
    rows = [battle(args, candidate, output / "confirm" / str(seed), seed, args.confirm_groups)
            for seed in selection["confirm_seeds"]]
    status = "pass" if all(row["pass"] for row in rows) else "not_proven"
    write_json(ROOT / output / "promotion.json", {"status": status, "candidate": selected,
               "results": rows, "automatically_installed": False})
    print("终测结论：", status, "；通过后再替换应用权重。")


def bundle(args):
    files = [Path(name) for name in ("model.py", "ppo_train.py", "battle_models.py",
             "run_autodl_training.py", "run_autodl_campaign.py", "run_autodl_rank_campaign.py",
             "AUTODL_TRAINING.md", "requirements-training.txt",
             "training/__init__.py", "training/parallel_env.py", "training/ppo_support.py",
             "training/cuda_rollout.py",
             "tests/test_ppo_support.py", "tests/test_south_match.py", "tests/test_parallel_env.py",
             "tests/test_autodl_training.py", "tests/test_autodl_campaign.py",
             "tests/test_forced_forward.py", "tests/test_cuda_rollout.py")]
    files.extend(Path(name) for name in ("tests/test_rule_fastpath.py",
                                        "tests/test_furiten.py", "tests/test_last_tile_actions.py",
                                        "tests/test_kan_limit.py"))
    files += [path.relative_to(ROOT) for path in (ROOT / "mahjong_env").glob("*.py")
              if path.name not in {"deal.py", "vis.py"}]
    files += [BASE, HISTORY]
    archive = ROOT / "artifacts" / "MahjongTrioAI-AutoDL-v2.zip"
    archive.parent.mkdir(exist_ok=True)
    manifest = {path.as_posix(): sha256(ROOT / path) for path in files}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in files:
            package.write(ROOT / path, "MahjongTrioAI-training/" + path.as_posix())
        package.writestr("MahjongTrioAI-training/bundle_manifest.json",
                         json.dumps(manifest, indent=2))
    print(f"训练包：{archive} ({archive.stat().st_size / 1024**2:.2f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "screen", "confirm", "bundle"))
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--hands-per-update", type=int, default=128)
    parser.add_argument("--env-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screen-groups", type=int, default=128)
    parser.add_argument("--confirm-groups", type=int, default=1000)
    args = parser.parse_args()
    if not 0 < args.hours <= 24 or not 0 <= args.seed < 100_000_000:
        parser.error("hours需在(0,24]内，seed需在[0,100000000)内以隔离训练与评测牌山")
    if args.screen_groups < 2 or args.confirm_groups < 1000:
        parser.error("screen-groups至少2，confirm-groups至少1000")
    if args.stage in ("smoke", "train"):
        train(args, smoke=args.stage == "smoke")
    else:
        {"screen": screen, "confirm": confirm, "bundle": bundle}[args.stage](args)


if __name__ == "__main__":
    main()
