"""分析 PPO 固定评估随更新次数的趋势，不启动训练或对局。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 history.json，判断 PPO 是否仍随训练时长改善"
    )
    parser.add_argument(
        "--history", type=Path,
        default=Path("ppo_runs/south_seed_20260909/history.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("battle_results/ppo_time_curve_20260909"),
    )
    return parser.parse_args()


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            result[order[position]] = rank
        cursor = end
    return result


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return correlation(average_ranks(left), average_ranks(right))


def linear_trend(updates: Sequence[float], values: Sequence[float]) -> Dict:
    count = len(updates)
    x_mean = statistics.fmean(updates)
    y_mean = statistics.fmean(values)
    denominator = sum((value - x_mean) ** 2 for value in updates)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(updates, values))
        / denominator
        if denominator
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    if count > 2 and denominator:
        residual = sum(
            (y - (intercept + slope * x)) ** 2
            for x, y in zip(updates, values)
        )
        standard_error = math.sqrt(residual / (count - 2) / denominator)
        half = 1.96 * standard_error
    else:
        half = math.inf
    return {
        "slope_per_100_updates": slope * 100,
        "ci95_low_per_100_updates": (slope - half) * 100,
        "ci95_high_per_100_updates": (slope + half) * 100,
    }


def analyze(rows: List[Dict]) -> Dict:
    evaluations = [row for row in rows if row.get("evaluation") is not None]
    if len(evaluations) < 4:
        raise ValueError("至少需要4个固定评估时间点；请缩短 --evaluation-interval")
    curve = [
        {
            "update": int(row["update"]),
            "total_matches": int(row["total_hands"]),
            "average_rank": float(row["evaluation"]["average_rank"]),
            "average_points": float(row["evaluation"]["average_points"]),
            "first_rate": float(row["evaluation"].get("first_rate", 0.0)),
            "third_rate": float(row["evaluation"].get("third_rate", 0.0)),
        }
        for row in evaluations
    ]
    updates = [row["update"] for row in curve]
    ranks = [row["average_rank"] for row in curve]
    points = [row["average_points"] for row in curve]
    window = max(2, len(curve) // 3)
    early = curve[:window]
    late = curve[-window:]
    early_rank = statistics.fmean(row["average_rank"] for row in early)
    late_rank = statistics.fmean(row["average_rank"] for row in late)
    early_points = statistics.fmean(row["average_points"] for row in early)
    late_points = statistics.fmean(row["average_points"] for row in late)
    rank_trend = linear_trend(updates, ranks)
    point_trend = linear_trend(updates, points)
    rank_rho = spearman(updates, ranks)
    point_rho = spearman(updates, points)
    direction_support = (
        rank_trend["slope_per_100_updates"] < 0
        and point_trend["slope_per_100_updates"] > 0
        and late_rank < early_rank
        and late_points > early_points
    )
    strong_statistical_support = (
        rank_trend["ci95_high_per_100_updates"] < 0
        and point_trend["ci95_low_per_100_updates"] > 0
    )
    if strong_statistical_support:
        verdict = "strong_internal_support"
        recommendation = "内部曲线强力支持继续训练，但仍需用独立 checkpoint 对局确认。"
    elif direction_support and rank_rho <= -0.4 and point_rho >= 0.4:
        verdict = "moderate_internal_support"
        recommendation = "曲线方向支持先做6至12小时续训试验，不足以直接证明三天都有效。"
    else:
        verdict = "not_supported_or_plateau"
        recommendation = "没有证明训练越久越好；应降低学习率、改奖励或使用早停。"
    best = max(
        curve,
        key=lambda row: -row["average_rank"] + row["average_points"] / 1_000_000,
    )
    return {
        "evaluation_points": len(curve),
        "curve": curve,
        "rank_trend": rank_trend,
        "point_trend": point_trend,
        "spearman_update_vs_rank": rank_rho,
        "spearman_update_vs_points": point_rho,
        "early_window": {
            "count": window,
            "average_rank": early_rank,
            "average_points": early_points,
        },
        "late_window": {
            "count": window,
            "average_rank": late_rank,
            "average_points": late_points,
        },
        "late_minus_early": {
            "rank_improvement": early_rank - late_rank,
            "point_improvement": late_points - early_points,
        },
        "best_internal_evaluation": best,
        "verdict": verdict,
        "recommendation": recommendation,
        "caveat": (
            "history 使用训练期间反复查看的固定评估牌山，只能作为内部趋势证据；"
            "三天训练决策应再比较 update 0、中期和末期 checkpoint 的独立对局。"
        ),
    }


def write_outputs(output_dir: Path, result: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["curve"][0]))
        writer.writeheader()
        writer.writerows(result["curve"])
    summary = [
        "# PPO 训练时长趋势",
        "",
        f"- 固定评估时间点：{result['evaluation_points']}",
        f"- 顺位趋势：每100次更新 {result['rank_trend']['slope_per_100_updates']:+.4f}",
        f"- 点数趋势：每100次更新 {result['point_trend']['slope_per_100_updates']:+.1f}",
        f"- 更新次数与顺位 Spearman：{result['spearman_update_vs_rank']:+.3f}（负数更好）",
        f"- 更新次数与点数 Spearman：{result['spearman_update_vs_points']:+.3f}（正数更好）",
        f"- 后段相对前段顺位改善：{result['late_minus_early']['rank_improvement']:+.4f}",
        f"- 后段相对前段点数改善：{result['late_minus_early']['point_improvement']:+.1f}",
        f"- 判断：`{result['verdict']}`",
        f"- 建议：{result['recommendation']}",
        "",
        f"> {result['caveat']}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main():
    args = parse_args()
    if not args.history.is_file():
        raise FileNotFoundError(args.history)
    rows = json.loads(args.history.read_text(encoding="utf-8"))
    result = analyze(rows)
    write_outputs(args.output_dir, result)
    print("\n".join((args.output_dir / "summary.md").read_text(encoding="utf-8").splitlines()[2:]))
    print(f"\n详细结果：{args.output_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
