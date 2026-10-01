"""第 1--3 天：固定基线、训练多随机种子 PPO，并做独立配对评测。

该脚本只负责组织现有 train/evaluate 工具，不会在导入时启动训练。
推荐先运行 smoke 验证全流程，再运行 full 正式实验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from training.parallel_env import recommended_env_workers


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = Path("training_runs/rich_tile_transformer/best.pt")
DEFAULT_SEEDS = (20260909, 20260910, 20260911)
BC_NAME = "rich_bc"


@dataclass(frozen=True)
class Profile:
    updates: int
    hands_per_update: int
    evaluation_groups: int
    lineup_groups: int


PROFILES = {
    # 只验证命令、断点、模型加载和报告生成，不用于判断强弱。
    "smoke": Profile(2, 32, 16, 16),
    # 适合先判断 PPO 是否值得继续；每对模型约 2,000 组配对牌山。
    "screen": Profile(200, 256, 256, 1_000),
    # 正式第 1--3 天实验；每对模型恰好 10,000 组配对牌山。
    "full": Profile(1_000, 256, 256, 5_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行三麻 AI 第 1--3 天的可复现 BC/PPO 对照实验"
    )
    parser.add_argument(
        "command",
        choices=("preflight", "train", "evaluate", "summarize", "status", "all"),
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-seed", type=int, default=20261001)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument(
        "--env-workers",
        type=int,
        default=0,
        help=(
            "所有并行子任务共享的环境进程总预算；0 根据 CPU 自动选择。"
            "i7-12700H 自动为 12"
        ),
    )
    parser.add_argument(
        "--parallel-train-jobs",
        type=int,
        default=2,
        help="同时训练多少个随机种子；RTX 3060 6GB 推荐 2，3 可能接近显存上限",
    )
    parser.add_argument(
        "--parallel-eval-jobs",
        type=int,
        default=3,
        help="同时运行多少组 BC/PPO 独立配对评测",
    )
    parser.add_argument("--updates", type=int, help="覆盖 profile 的 PPO 更新数")
    parser.add_argument(
        "--hands-per-update", type=int, help="覆盖 profile 的每次更新小局数"
    )
    parser.add_argument(
        "--evaluation-groups", type=int, help="覆盖 PPO 内部固定评估牌山组数"
    )
    parser.add_argument(
        "--lineup-groups", type=int, help="覆盖独立锦标赛每种三模型阵容的牌山组数"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="从头训练并替换同一 run-id 的断点；不加则自动续跑",
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="preflight 时跳过 pytest"
    )
    return parser.parse_args()


def resolved_profile(args: argparse.Namespace) -> Profile:
    base = PROFILES[args.profile]
    profile = Profile(
        updates=args.updates or base.updates,
        hands_per_update=args.hands_per_update or base.hands_per_update,
        evaluation_groups=args.evaluation_groups or base.evaluation_groups,
        lineup_groups=args.lineup_groups or base.lineup_groups,
    )
    for name, value in asdict(profile).items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正数")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds 不能包含重复值")
    if len(args.seeds) < 3:
        raise ValueError("正式复现实验至少需要 3 个不同 PPO 随机种子")
    positive_args = {
        "batch-groups": args.batch_groups,
        "parallel-train-jobs": args.parallel_train_jobs,
        "parallel-eval-jobs": args.parallel_eval_jobs,
    }
    invalid = [name for name, value in positive_args.items() if value <= 0]
    if invalid:
        raise ValueError("以下参数必须为正数：" + "、".join(invalid))
    if args.env_workers < 0:
        raise ValueError("--env-workers 不能为负数；0 表示自动")
    return profile


def env_worker_budget(args: argparse.Namespace) -> int:
    return args.env_workers or recommended_env_workers()


def workers_per_child(args: argparse.Namespace, parallel_jobs: int) -> int:
    return max(1, env_worker_budget(args) // max(1, parallel_jobs))


def run_id(args: argparse.Namespace, profile: Profile) -> str:
    return (
        f"{args.profile}_u{profile.updates}"
        f"_h{profile.hands_per_update}"
        f"_e{profile.evaluation_groups}"
    )


def training_root(args: argparse.Namespace, profile: Profile) -> Path:
    return ROOT / "ppo_runs" / "day01_03" / run_id(args, profile)


def battle_root(args: argparse.Namespace, profile: Profile) -> Path:
    return (
        ROOT
        / "battle_results"
        / "day01_03"
        / run_id(args, profile)
        / f"pairwise_g{profile.lineup_groups}_seed{args.evaluation_seed}"
    )


def seed_dir(args: argparse.Namespace, profile: Profile, seed: int) -> Path:
    return training_root(args, profile) / f"seed_{seed}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_command(command: Iterable[str]) -> None:
    command = list(command)
    rendered = subprocess.list2cmdline(command)
    print(f"\n> {rendered}\n", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def stop_processes(processes: Iterable[subprocess.Popen]) -> None:
    processes = [process for process in processes if process.poll() is None]
    for process in processes:
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            pass
    deadline = time.monotonic() + 10
    while processes and time.monotonic() < deadline:
        processes = [process for process in processes if process.poll() is None]
        time.sleep(0.2)
    for process in processes:
        process.terminate()


def run_parallel_commands(
    jobs: Iterable[tuple[str, Iterable[str], Path]], max_jobs: int
) -> None:
    """用独立 Python 进程并行运行任务，输出分别写入日志文件。"""
    pending = [
        (name, list(command), log_path) for name, command, log_path in jobs
    ]
    active: List[Dict[str, Any]] = []
    failures = []
    last_status = 0.0
    child_env = os.environ.copy()
    # 每个进程主要依靠一个 Python 核心推进环境；限制底层数学库线程，
    # 避免多个进程在小型矩阵运算上互相抢占全部 CPU。
    child_env.setdefault("OMP_NUM_THREADS", "1")
    child_env.setdefault("MKL_NUM_THREADS", "1")
    child_env["PYTHONUNBUFFERED"] = "1"

    try:
        while pending or active:
            while pending and len(active) < max_jobs:
                name, command, log_path = pending.pop(0)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("a", encoding="utf-8")
                creationflags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                )
                print(
                    f"\n启动 {name}\n> {subprocess.list2cmdline(command)}"
                    f"\n日志：{log_path}\n",
                    flush=True,
                )
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    creationflags=creationflags,
                )
                active.append(
                    {
                        "name": name,
                        "process": process,
                        "log_handle": log_handle,
                        "log_path": log_path,
                    }
                )

            completed = []
            for job in active:
                return_code = job["process"].poll()
                if return_code is None:
                    continue
                job["log_handle"].close()
                completed.append(job)
                if return_code:
                    failures.append((job["name"], return_code, job["log_path"]))
                    print(
                        f"{job['name']} 失败，退出码 {return_code}；"
                        f"请查看 {job['log_path']}",
                        flush=True,
                    )
                else:
                    print(f"{job['name']} 完成", flush=True)
            active = [job for job in active if job not in completed]

            if failures:
                stop_processes(job["process"] for job in active)
                for job in active:
                    job["log_handle"].close()
                raise RuntimeError(f"并行任务失败：{failures}")
            now = time.monotonic()
            if active and now - last_status >= 30:
                names = "、".join(job["name"] for job in active)
                print(f"仍在运行：{names}", flush=True)
                last_status = now
            if pending or active:
                time.sleep(2)
    except KeyboardInterrupt:
        print("\n正在停止子进程；已保存的训练/对战断点可以续跑……", flush=True)
        stop_processes(job["process"] for job in active)
        for job in active:
            job["log_handle"].close()
        raise


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"{path} 不是带元数据的 checkpoint")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "model_name": payload.get("model_name"),
        "obs_channels": payload.get("obs_channels"),
        "observation_key": payload.get("observation_key"),
        "training_method": payload.get("training_method", "behavior_cloning"),
        "ppo_update": payload.get("ppo_update"),
        "ppo_total_hands": payload.get("ppo_total_hands"),
        "ppo_best_evaluation_points": payload.get("ppo_best_evaluation_points"),
    }


def require_rich_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"基础模型不存在：{path}")
    metadata = checkpoint_metadata(path)
    expected = {
        "model_name": "tile_transformer",
        "obs_channels": 101,
        "observation_key": "rich_observation",
    }
    mismatches = {
        key: (metadata.get(key), expected_value)
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"基础模型不是预期的 rich BC：{mismatches}")
    return metadata


def experiment_config(args: argparse.Namespace, profile: Profile) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "purpose": "day01_03_rich_bc_vs_multi_seed_ppo",
        "profile_name": args.profile,
        "run_id": run_id(args, profile),
        "profile": asdict(profile),
        "seeds": list(args.seeds),
        "device": args.device,
        "evaluation_seed": args.evaluation_seed,
        "batch_groups": args.batch_groups,
        "parallel_train_jobs": args.parallel_train_jobs,
        "parallel_eval_jobs": args.parallel_eval_jobs,
        "env_worker_budget": env_worker_budget(args),
        "env_workers_requested": args.env_workers,
        "base_model": str(args.base_model),
        # 两模型比较器会构造 A+A+B、A+B+B 两个镜像阵容。
        "pairwise_groups_per_model_pair": profile.lineup_groups * 2,
        "evaluation_unit": "single_hand",
        "warning": "当前环境尚非完整南风半庄；结果只衡量小局点数能力。",
    }


def do_preflight(args: argparse.Namespace, profile: Profile) -> None:
    metadata = require_rich_checkpoint(args.base_model)
    if not args.skip_tests:
        run_command([sys.executable, "-m", "pytest", "-q"])

    manifest = experiment_config(args, profile)
    manifest["base_checkpoint"] = metadata
    manifest["training_output_root"] = str(training_root(args, profile))
    manifest["battle_output_root"] = str(battle_root(args, profile))
    path = training_root(args, profile) / "experiment_manifest.json"
    write_json(path, manifest)
    print(f"预检通过；实验清单：{path}")


def completed_updates(path: Path) -> int:
    history_path = path / "history.json"
    if not history_path.exists():
        return 0
    history = load_json(history_path)
    if not history:
        return 0
    return int(history[-1]["update"])


def do_train(args: argparse.Namespace, profile: Profile) -> None:
    require_rich_checkpoint(args.base_model)
    jobs = []
    parallel_jobs = min(args.parallel_train_jobs, len(args.seeds))
    child_workers = workers_per_child(args, parallel_jobs)
    print(
        f"训练 CPU 预算：总计 {env_worker_budget(args)} 个环境进程；"
        f"最多 {parallel_jobs} 个训练并发，每个训练 {child_workers} 个"
    )
    for seed in args.seeds:
        output_dir = seed_dir(args, profile, seed)
        done = completed_updates(output_dir)
        if done >= profile.updates and not args.fresh:
            print(f"跳过 seed={seed}：已完成 {done}/{profile.updates} 次更新")
            continue
        command = [
            sys.executable,
            "ppo_train.py",
            "--base-model",
            str(args.base_model),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--updates",
            str(profile.updates),
            "--hands-per-update",
            str(profile.hands_per_update),
            "--evaluation-groups",
            str(profile.evaluation_groups),
            "--device",
            args.device,
            "--env-workers",
            str(child_workers),
        ]
        if args.fresh:
            command.append("--fresh")
        jobs.append(
            (
                f"PPO seed={seed}",
                command,
                output_dir / "console.log",
            )
        )
    if jobs:
        run_parallel_commands(jobs, min(args.parallel_train_jobs, len(jobs)))


def require_finished_training(
    args: argparse.Namespace, profile: Profile
) -> List[Path]:
    checkpoints = []
    incomplete = []
    for seed in args.seeds:
        directory = seed_dir(args, profile, seed)
        done = completed_updates(directory)
        checkpoint = directory / "best.pt"
        if done < profile.updates or not checkpoint.exists():
            incomplete.append(f"seed={seed}: {done}/{profile.updates}")
        else:
            checkpoints.append(checkpoint)
    if incomplete:
        raise RuntimeError("训练尚未完成，请先运行 train：" + "; ".join(incomplete))
    return checkpoints


def do_evaluate(args: argparse.Namespace, profile: Profile) -> None:
    require_rich_checkpoint(args.base_model)
    checkpoints = require_finished_training(args, profile)
    jobs = []
    parallel_jobs = min(args.parallel_eval_jobs, len(checkpoints))
    child_workers = workers_per_child(args, parallel_jobs)
    print(
        f"评测 CPU 预算：总计 {env_worker_budget(args)} 个环境进程；"
        f"最多 {parallel_jobs} 个评测并发，每个评测 {child_workers} 个"
    )
    for seed, checkpoint in zip(args.seeds, checkpoints):
        model = f"ppo_seed_{seed}"
        output_dir = battle_root(args, profile) / model
        command = [
            sys.executable,
            "battle_models.py",
            "--models",
            f"{BC_NAME}={args.base_model}",
            f"{model}={checkpoint}",
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.evaluation_seed),
            "--min-groups",
            str(profile.lineup_groups),
            "--max-groups",
            str(profile.lineup_groups),
            # min=max 已强制跑满；1 只是满足现有参数的正数校验。
            "--target-ci",
            "1",
            "--batch-groups",
            str(args.batch_groups),
            "--device",
            args.device,
            "--env-workers",
            str(child_workers),
        ]
        if args.fresh:
            command.append("--fresh")
        jobs.append((f"评测 {model}", command, output_dir / "console.log"))
    run_parallel_commands(jobs, min(args.parallel_eval_jobs, len(jobs)))


def normalized_pairwise(row: Dict[str, Any], model: str) -> Dict[str, Any]:
    if row["model_a"] == model and row["model_b"] == BC_NAME:
        return {
            "model": model,
            "paired_seed_groups": row["paired_seed_groups"],
            "points_vs_bc": row["mean_point_difference_a_minus_b"],
            "ci95_low": row["point_difference_ci95_low"],
            "ci95_high": row["point_difference_ci95_high"],
            "utility_vs_bc": row["mean_utility_difference_a_minus_b"],
        }
    if row["model_a"] == BC_NAME and row["model_b"] == model:
        return {
            "model": model,
            "paired_seed_groups": row["paired_seed_groups"],
            "points_vs_bc": -row["mean_point_difference_a_minus_b"],
            "ci95_low": -row["point_difference_ci95_high"],
            "ci95_high": -row["point_difference_ci95_low"],
            "utility_vs_bc": -row["mean_utility_difference_a_minus_b"],
        }
    raise ValueError(f"pairwise 行不包含 {model} 与 {BC_NAME}")


def best_internal_evaluation(directory: Path) -> Dict[str, Any]:
    history = load_json(directory / "history.json")
    candidates = [row for row in history if row.get("evaluation") is not None]
    best = max(candidates, key=lambda row: row["evaluation"]["average_points"])
    return {
        "update": best["update"],
        "total_hands": best["total_hands"],
        **best["evaluation"],
    }


def make_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# 第 1--3 天：rich BC / 多种子 PPO 实验报告",
        "",
        f"- Run ID：`{summary['config']['run_id']}`",
        f"- 独立评测牌山种子：`{summary['config']['evaluation_seed']}`",
        f"- 每对模型配对牌山组数：{summary['config']['pairwise_groups_per_model_pair']:,}",
        "- 评测单位：单个小局（当前尚不是完整南风半庄）",
        "",
        "## PPO 相对 rich BC",
        "",
        "| PPO 种子 | 最佳内部更新 | 对 BC 点差/局 | 95% CI | 配对组数 | 结论 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["ppo_vs_bc"]:
        if row["ci95_low"] > 0:
            conclusion = "显著更好"
        elif row["ci95_high"] < 0:
            conclusion = "显著更差"
        else:
            conclusion = "无法区分"
        lines.append(
            f"| {row['seed']} | {row['best_internal']['update']} | "
            f"{row['points_vs_bc']:+.1f} | "
            f"[{row['ci95_low']:+.1f}, {row['ci95_high']:+.1f}] | "
            f"{row['paired_seed_groups']:,} | {conclusion} |"
        )
    lines.extend(
        [
            "",
            "## 总结",
            "",
            f"- 显著优于 BC 的种子：{summary['significantly_better_seeds']} / {summary['seed_count']}",
            f"- 三个种子的描述性平均点差：{summary['descriptive_mean_points_vs_bc']:+.1f} 点/局",
            f"- 全部模型非法动作合计：{summary['total_invalid_actions']}",
            "",
            "> 注意：跨随机种子的描述性平均值不是新的置信区间。是否进入下一阶段，"
            "应以至少 2/3 个种子的独立配对区间为正、且非法动作数为 0 作为门槛。",
            "",
        ]
    )
    return "\n".join(lines)


def do_summarize(args: argparse.Namespace, profile: Profile) -> None:
    checkpoints = require_finished_training(args, profile)
    rows = []
    rankings = []
    all_pairwise = []
    source_reports = []
    for seed, checkpoint in zip(args.seeds, checkpoints):
        model = f"ppo_seed_{seed}"
        report_path = battle_root(args, profile) / model / "battle_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"评测报告不存在，请先运行 evaluate：{report_path}"
            )
        report = load_json(report_path)
        pair = report["pairwise"][0]
        normalized = normalized_pairwise(pair, model)
        rankings.extend(report["rankings"])
        all_pairwise.extend(report["pairwise"])
        source_reports.append(str(report_path))
        rows.append(
            {
                "seed": seed,
                "checkpoint": checkpoint_metadata(checkpoint),
                "best_internal": best_internal_evaluation(checkpoint.parent),
                **normalized,
            }
        )
    summary = {
        "config": experiment_config(args, profile),
        "base_checkpoint": checkpoint_metadata(args.base_model),
        "ppo_vs_bc": rows,
        "seed_count": len(rows),
        "significantly_better_seeds": sum(row["ci95_low"] > 0 for row in rows),
        "descriptive_mean_points_vs_bc": sum(
            row["points_vs_bc"] for row in rows
        )
        / len(rows),
        "total_invalid_actions": sum(
            int(row.get("invalid_actions", 0)) for row in rankings
        ),
        "rankings": rankings,
        "all_pairwise": all_pairwise,
        "source_reports": source_reports,
    }
    output_dir = battle_root(args, profile)
    json_path = output_dir / "experiment_summary.json"
    markdown_path = output_dir / "experiment_summary.md"
    write_json(json_path, summary)
    markdown_path.write_text(make_markdown(summary), encoding="utf-8")
    print(make_markdown(summary))
    print(f"\nJSON：{json_path}\nMarkdown：{markdown_path}")


def do_status(args: argparse.Namespace, profile: Profile) -> None:
    print(f"Run ID: {run_id(args, profile)}")
    for seed in args.seeds:
        directory = seed_dir(args, profile, seed)
        done = completed_updates(directory)
        best = "-"
        history_path = directory / "history.json"
        if history_path.exists() and done:
            best_row = best_internal_evaluation(directory)
            best = (
                f"update={best_row['update']}, "
                f"internal={best_row['average_points']:+.1f} 点/局"
            )
        print(f"seed={seed}: {done}/{profile.updates}; {best}")

    print("独立配对评测：")
    for seed in args.seeds:
        model = f"ppo_seed_{seed}"
        progress_path = battle_root(args, profile) / model / "progress.json"
        if not progress_path.exists():
            print(f"  {model}: 尚未开始")
            continue
        progress = load_json(progress_path)
        states = list(progress.get("lineups", {}).values())
        groups = sum(len(state.get("groups", [])) for state in states)
        errors = sum(len(state.get("errors", [])) for state in states)
        complete = all(state.get("complete", False) for state in states)
        print(
            f"  {model}: {groups}/{profile.lineup_groups * 2} 配对组; "
            f"异常={errors}; 完成={complete}"
        )


def main() -> None:
    args = parse_args()
    if not args.base_model.is_absolute():
        args.base_model = ROOT / args.base_model
    profile = resolved_profile(args)
    if args.command == "preflight":
        do_preflight(args, profile)
    elif args.command == "train":
        do_train(args, profile)
    elif args.command == "evaluate":
        do_evaluate(args, profile)
    elif args.command == "summarize":
        do_summarize(args, profile)
    elif args.command == "status":
        do_status(args, profile)
    else:
        do_preflight(args, profile)
        do_train(args, profile)
        do_evaluate(args, profile)
        do_summarize(args, profile)


if __name__ == "__main__":
    main()
