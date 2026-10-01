"""运行无中途评估的 8 小时 PPO，并在结束后统一比较五个时间点。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="PPO 8小时训练时长对照实验")
    parser.add_argument(
        "command", choices=("train", "evaluate", "copilot", "all", "status"),
        help="训练、内部循环赛、逐个对比Copilot、依次训练评测或查看状态",
    )
    parser.add_argument(
        "--base-model", type=Path,
        default=Path("ppo_runs/south_seed_20260909/last.pt"),
        help="8小时实验起点；默认使用独立测评更好的 u300",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("ppo_runs/south_8h_from_u300")
    )
    parser.add_argument(
        "--battle-dir", type=Path,
        default=Path("battle_results/south_8h_checkpoint_league"),
    )
    parser.add_argument(
        "--copilot-dir", type=Path,
        default=Path("battle_results/south_8h_vs_copilot"),
    )
    parser.add_argument(
        "--mortal-model", type=Path,
        default=Path("integrations/MahjongCopilot/models/mortal_3p.pth"),
    )
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--checkpoint-count", type=int, default=5)
    parser.add_argument("--training-seed", type=int, default=20260912)
    parser.add_argument("--evaluation-seed", type=int, default=20261202)
    parser.add_argument("--copilot-seed", type=int, default=20261203)
    parser.add_argument("--hands-per-update", type=int, default=32)
    parser.add_argument("--actor-lr", type=float, default=2e-6)
    parser.add_argument("--critic-lr", type=float, default=6e-5)
    parser.add_argument("--min-groups", type=int, default=300)
    parser.add_argument("--max-groups", type=int, default=1000)
    parser.add_argument("--target-ci", type=float, default=100.0)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--env-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh-training", action="store_true")
    parser.add_argument("--fresh-evaluation", action="store_true")
    parser.add_argument("--fresh-copilot", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.hours <= 0:
        raise ValueError("--hours 必须为正数")
    if not 3 <= args.checkpoint_count <= 8:
        raise ValueError("--checkpoint-count 必须在3到8之间")
    if args.hands_per_update <= 0 or args.batch_groups <= 0:
        raise ValueError("并行批量必须为正数")
    if args.min_groups < 2 or args.max_groups < args.min_groups:
        raise ValueError("测评组数范围无效")
    if args.env_workers < 0:
        raise ValueError("--env-workers 不能为负数")


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_command(command: List[str]) -> None:
    print("\n即将执行：")
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def training_command(args) -> List[str]:
    interval_minutes = args.hours * 60 / (args.checkpoint_count - 1)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "ppo_train.py"),
        "--base-model", str(absolute(args.base_model)),
        "--output-dir", str(absolute(args.run_dir)),
        "--game-mode", "south",
        "--rounds-per-wind", "3",
        "--seed", str(args.training_seed),
        "--updates", "999999",
        "--hands-per-update", str(args.hands_per_update),
        "--evaluation-interval", "0",
        "--evaluation-groups", "1",
        "--actor-lr", str(args.actor_lr),
        "--critic-lr", str(args.critic_lr),
        "--max-hours", str(args.hours),
        "--checkpoint-minutes", str(interval_minutes),
        "--max-steps", "3000",
        "--env-workers", str(args.env_workers),
        "--device", args.device,
    ]
    if args.fresh_training:
        command.append("--fresh")
    return command


def snapshot_metadata(path: Path) -> Dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return {
        "path": path,
        "update": int(payload.get("ppo_update", 0)),
        "elapsed_seconds": float(payload.get("ppo_elapsed_seconds", 0.0)),
    }


def select_snapshots(args) -> List[Dict]:
    checkpoint_dir = absolute(args.run_dir) / "checkpoints"
    records = [
        snapshot_metadata(path)
        for path in checkpoint_dir.glob("*.pt")
    ]
    if not records:
        raise FileNotFoundError(f"没有找到训练时间点：{checkpoint_dir}")
    # 同一更新可能同时命中整点文件和 final 文件，只保留一个。
    by_update = {}
    for record in sorted(records, key=lambda row: row["elapsed_seconds"]):
        by_update.setdefault(record["update"], record)
    available = list(by_update.values())
    targets = [
        args.hours * 3600 * index / (args.checkpoint_count - 1)
        for index in range(args.checkpoint_count)
    ]
    selected = []
    unused = available[:]
    for target in targets:
        if not unused:
            break
        chosen = min(
            unused,
            key=lambda row: abs(row["elapsed_seconds"] - target),
        )
        unused.remove(chosen)
        selected.append(chosen)
    selected.sort(key=lambda row: row["elapsed_seconds"])
    if len(selected) != args.checkpoint_count:
        raise ValueError(
            f"需要 {args.checkpoint_count} 个不同时间点，实际只有 {len(selected)} 个；"
            "请确认训练已经完成"
        )
    for index, record in enumerate(selected):
        record["name"] = f"t{index}_{record['elapsed_seconds'] / 3600:.1f}h"
    return selected


def evaluation_command(args, selected: List[Dict]) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "battle_models.py"),
        "--game-mode", "south",
        "--rounds-per-wind", "3",
        "--models",
    ]
    command.extend(
        f"{record['name']}={record['path']}" for record in selected
    )
    command.extend([
        "--output-dir", str(absolute(args.battle_dir)),
        "--seed", str(args.evaluation_seed),
        "--min-groups", str(args.min_groups),
        "--max-groups", str(args.max_groups),
        "--target-ci", str(args.target_ci),
        "--batch-groups", str(args.batch_groups),
        "--max-steps", "3000",
        "--env-workers", str(args.env_workers),
        "--device", args.device,
    ])
    if args.fresh_evaluation:
        command.append("--fresh")
    return command


def copilot_command(args, record: Dict) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "compare_copilot_model.py"),
        "--our-model", str(record["path"]),
        "--our-name", record["name"],
        "--mortal-model", str(absolute(args.mortal_model)),
        "--mortal-name", "MahjongCopilot-Mortal",
        "--game-mode", "south",
        "--rounds-per-wind", "3",
        "--seed", str(args.copilot_seed),
        "--min-groups", str(args.min_groups),
        "--max-groups", str(args.max_groups),
        "--target-ci", str(args.target_ci),
        "--batch-groups", str(args.batch_groups),
        "--env-workers", str(args.env_workers),
        "--max-steps", "3000",
        "--output-dir", str(absolute(args.copilot_dir) / record["name"]),
        "--device", args.device,
    ]
    if args.fresh_copilot:
        command.append("--fresh")
    return command


def normalized_pair(pair: Dict, late: str, early: str) -> Dict:
    if pair["model_a"] == late and pair["model_b"] == early:
        return {
            "mean": pair["mean_point_difference_a_minus_b"],
            "low": pair["point_difference_ci95_low"],
            "high": pair["point_difference_ci95_high"],
        }
    if pair["model_a"] == early and pair["model_b"] == late:
        return {
            "mean": -pair["mean_point_difference_a_minus_b"],
            "low": -pair["point_difference_ci95_high"],
            "high": -pair["point_difference_ci95_low"],
        }
    raise ValueError("pairwise 行不包含指定的早晚时间点")


def summarize(args, selected: List[Dict]) -> Dict:
    report_path = absolute(args.battle_dir) / "battle_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = {row["model_name"]: row for row in report["rankings"]}
    early_name = selected[0]["name"]
    late_name = selected[-1]["name"]
    pair = next(
        row for row in report["pairwise"]
        if {row["model_a"], row["model_b"]} == {early_name, late_name}
    )
    advantage = normalized_pair(pair, late_name, early_name)
    chronological = [rows[record["name"]] for record in selected]
    adjacent_rank_improvements = sum(
        right["average_rank"] < left["average_rank"]
        for left, right in zip(chronological, chronological[1:])
    )
    early = chronological[0]
    late = chronological[-1]
    rank_improvement = early["average_rank"] - late["average_rank"]
    passes = (
        rank_improvement >= 0.02
        and advantage["low"] > 0
        and late["deal_in_rate"] <= early["deal_in_rate"] + 0.01
        and adjacent_rank_improvements >= len(chronological) - 2
        and all(row["invalid_actions"] == 0 for row in chronological)
    )
    result = {
        "selected_checkpoints": [
            {
                "name": record["name"],
                "path": str(record["path"]),
                "update": record["update"],
                "elapsed_hours": record["elapsed_seconds"] / 3600,
                "metrics": rows[record["name"]],
            }
            for record in selected
        ],
        "late_minus_early_points": advantage,
        "late_minus_early_rank_improvement": rank_improvement,
        "adjacent_rank_improvements": adjacent_rank_improvements,
        "adjacent_comparisons": len(chronological) - 1,
        "three_day_training_gate": "pass" if passes else "fail",
        "conclusion": (
            "当前0至8小时区间支持延长训练，但三天任务仍应定期保存并早停。"
            if passes
            else "当前证据不支持直接训练三天；曲线已平台、波动或发生退化。"
        ),
    }
    output = absolute(args.battle_dir) / "duration_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n训练时长结论：{result['conclusion']}")
    print(f"三天训练准入：{result['three_day_training_gate']}")
    print(f"汇总：{output}")
    return result


def summarize_copilot(args, selected: List[Dict]) -> List[Dict]:
    rows = []
    mortal_name = "MahjongCopilot-Mortal"
    for record in selected:
        report_path = (
            absolute(args.copilot_dir) / record["name"] / "battle_report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rankings = {row["model_name"]: row for row in report["rankings"]}
        candidate = rankings[record["name"]]
        mortal = rankings[mortal_name]
        pair = next(
            row for row in report["pairwise"]
            if {row["model_a"], row["model_b"]}
            == {record["name"], mortal_name}
        )
        advantage = normalized_pair(pair, record["name"], mortal_name)
        rows.append({
            "model_name": record["name"],
            "checkpoint": str(record["path"]),
            "update": record["update"],
            "elapsed_hours": record["elapsed_seconds"] / 3600,
            "average_rank": candidate["average_rank"],
            "first_rate": candidate["first_rate"],
            "third_rate": candidate["third_rate"],
            "win_rate": candidate["win_rate"],
            "tsumo_rate": candidate["tsumo_rate"],
            "deal_in_rate": candidate["deal_in_rate"],
            "average_points": candidate["average_points"],
            "points_vs_copilot": advantage["mean"],
            "points_vs_copilot_ci95_low": advantage["low"],
            "points_vs_copilot_ci95_high": advantage["high"],
            "copilot_average_rank": mortal["average_rank"],
            "copilot_average_points": mortal["average_points"],
            "invalid_actions": candidate["invalid_actions"],
            "mortal_bridge_fallbacks": report.get("comparison", {})
            .get("mortal_bridge", {})
            .get("fallbacks", 0),
        })
    rows.sort(
        key=lambda row: (
            row["points_vs_copilot"],
            -row["average_rank"],
            row["first_rate"],
        ),
        reverse=True,
    )
    output_dir = absolute(args.copilot_dir)
    json_path = output_dir / "copilot_comparison_summary.json"
    csv_path = output_dir / "copilot_comparison_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n各训练时间点相对 MahjongCopilot 排名")
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank}. {row['model_name']} | 顺位 {row['average_rank']:.3f} | "
            f"对Copilot {row['points_vs_copilot']:+.1f}点 "
            f"[{row['points_vs_copilot_ci95_low']:+.1f}, "
            f"{row['points_vs_copilot_ci95_high']:+.1f}] | "
            f"胡牌 {row['win_rate']:.2%} | 点炮 {row['deal_in_rate']:.2%}"
        )
    print(f"总表：{csv_path}")
    return rows


def show_status(args):
    run_dir = absolute(args.run_dir)
    history_path = run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in history)
        update = history[-1]["update"] if history else 0
        print(f"训练：update={update}，累计={elapsed / 3600:.2f}/{args.hours:.2f}小时")
    else:
        print("训练尚未开始")
    checkpoint_dir = run_dir / "checkpoints"
    print(f"时间点模型：{len(list(checkpoint_dir.glob('*.pt')))} 个")
    report = absolute(args.battle_dir) / "battle_report.json"
    print(f"循环赛报告：{'已生成' if report.exists() else '尚未生成'}")
    copilot_report = absolute(args.copilot_dir) / "copilot_comparison_summary.json"
    print(f"Copilot 总表：{'已生成' if copilot_report.exists() else '尚未生成'}")


def main():
    args = parse_args()
    validate_args(args)
    if args.command == "status":
        show_status(args)
        return
    if args.command in ("train", "all"):
        if not absolute(args.base_model).is_file():
            raise FileNotFoundError(absolute(args.base_model))
        run_command(training_command(args))
    if args.command in ("evaluate", "all"):
        selected = select_snapshots(args)
        manifest_path = absolute(args.battle_dir) / "selected_checkpoints.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                [
                    {**record, "path": str(record["path"])}
                    for record in selected
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        run_command(evaluation_command(args, selected))
        summarize(args, selected)
    if args.command == "copilot":
        if not absolute(args.mortal_model).is_file():
            raise FileNotFoundError(absolute(args.mortal_model))
        selected = select_snapshots(args)
        for record in selected:
            run_command(copilot_command(args, record))
        summarize_copilot(args, selected)


if __name__ == "__main__":
    main()
