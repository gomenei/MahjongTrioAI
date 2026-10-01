"""在完全相同的测试集上重新评估并排名所有候选模型。"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from model import count_parameters, load_checkpoint_model
from training.data import build_loader, prepare_cache, prepare_splits
from training.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="公平比较训练完成的三人麻将模型")
    parser.add_argument("--data", type=Path, default=Path("data/data.pkl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/training_cache"))
    parser.add_argument("--runs-dir", type=Path, default=Path("training_runs"))
    parser.add_argument("--checkpoints", type=Path, nargs="*", help="只比较指定权重；不填则自动发现")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="默认精确复用旧 deeptrain.py 的 20%% baseline 测试集",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument(
        "--sort-by",
        choices=(
            "decision_accuracy", "accuracy", "macro_action_accuracy",
            "action_type_accuracy", "top3_accuracy", "loss",
        ),
        default="decision_accuracy",
    )
    parser.add_argument(
        "--exclude-legacy",
        action="store_true",
        help="不把现有 model/model.pt baseline 加入比较",
    )
    parser.add_argument(
        "--export-best",
        type=Path,
        help="可选：把排名第一的权重复制到此处，例如 model/model.pt",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def discover_checkpoints(args):
    if args.checkpoints:
        paths = list(args.checkpoints)
    else:
        paths = sorted(args.runs_dir.glob("*/best.pt"))
        legacy = Path("model/model.pt")
        if not args.exclude_legacy and legacy.exists():
            paths.append(legacy)

    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def display_name(path: Path, metadata: dict) -> str:
    name = metadata.get("model_name", path.stem)
    if metadata.get("legacy"):
        return f"{name}（旧 baseline）"
    return str(name)


def write_reports(results, runs_dir: Path, sort_by: str):
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / "comparison.json"
    csv_path = runs_dir / "comparison.csv"
    report = {"sort_by": sort_by, "models": results}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = (
        "rank", "model_name", "checkpoint", "parameters", "samples", "decision_samples",
        "forced_samples", "decision_accuracy", "decision_loss", "accuracy", "top3_accuracy",
        "top5_accuracy", "action_type_accuracy", "macro_action_accuracy", "loss",
        "samples_per_second",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields})
    return json_path, csv_path


def main():
    args = parse_args()
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("验证集比例与测试集比例之和必须小于 1")
    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError("没有发现 best.pt；请先运行一键训练，或通过 --checkpoints 指定权重。")
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError("以下权重不存在：" + "、".join(missing))

    device = select_device(args.device)
    metadata = prepare_cache(args.data, args.cache_dir)
    splits = prepare_splits(
        args.cache_dir,
        metadata["samples"],
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    test_loader = build_loader(
        args.cache_dir,
        splits["test"],
        args.batch_size,
        args.workers,
        device.type == "cuda",
    )

    results = []
    for path in checkpoints:
        print(f"正在评估：{path}")
        model, checkpoint_metadata = load_checkpoint_model(path, map_location=device)
        model = model.to(device)
        metrics = evaluate_model(model, test_loader, device)
        results.append(
            {
                "model_name": display_name(path, checkpoint_metadata),
                "checkpoint": str(path.resolve()),
                "parameters": count_parameters(model),
                **metrics,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reverse = args.sort_by != "loss"
    results.sort(key=lambda item: item[args.sort_by], reverse=reverse)
    for rank, item in enumerate(results, 1):
        item["rank"] = rank

    print("\n统一测试集比较结果")
    print(f"{'排名':<6}{'模型':<28}{'决策准确':>10}{'总准确':>10}{'Top-3':>10}{'宏平均':>10}{'Loss':>11}")
    for item in results:
        print(
            f"{item['rank']:<6}{item['model_name']:<28}"
            f"{item['decision_accuracy']:>9.3%}{item['accuracy']:>10.3%}"
            f"{item['top3_accuracy']:>10.3%}{item['macro_action_accuracy']:>10.3%}"
            f"{item['loss']:>11.5f}"
        )

    json_path, csv_path = write_reports(results, args.runs_dir, args.sort_by)
    print(f"\n详细报告：{json_path}\n表格报告：{csv_path}")
    if args.export_best:
        destination = args.export_best
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(results[0]["checkpoint"])
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        print(f"第一名模型已复制到：{destination}")


if __name__ == "__main__":
    main()
