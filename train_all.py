"""一键训练所有候选模型；不会在导入时自动开始训练。"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import MODEL_REGISTRY, count_parameters, create_model
from training.data import build_loaders, prepare_cache, prepare_splits
from training.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="训练并保存全部三人麻将候选模型")
    parser.add_argument("--data", type=Path, default=Path("data/data.pkl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/training_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs"))
    parser.add_argument(
        "--observation-key",
        choices=("observation", "rich_observation"),
        default="observation",
        help="旧 6 层特征或新 101 层公共信息 one-hot 特征",
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0, help="Windows 推荐保持 0；内存充足时可设为 2-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="默认 0.2，以精确复用旧 deeptrain.py 留出的 baseline 测试集",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(inputs, targets, device):
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    return inputs, targets.to(device, non_blocking=True)


def serializable_training_args(args):
    """checkpoint 只写入安全的基本类型，便于 weights_only 模式加载。"""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def masked_cross_entropy(logits, targets, action_mask, label_smoothing):
    """仅在合法动作之间做标签平滑，避免给非法动作分配概率。"""
    if label_smoothing <= 0:
        return F.cross_entropy(logits, targets)
    legal = action_mask > 0
    log_probabilities = F.log_softmax(logits, dim=1)
    negative_log_likelihood = F.nll_loss(log_probabilities, targets, reduction="none")
    legal_log_probabilities = log_probabilities.masked_fill(~legal, 0.0)
    legal_counts = legal.sum(dim=1).clamp_min(1)
    smooth_loss = -legal_log_probabilities.sum(dim=1) / legal_counts
    return (
        (1.0 - label_smoothing) * negative_log_likelihood
        + label_smoothing * smooth_loss
    ).mean()


def train_one_model(name, args, split_paths, device):
    seed_everything(args.seed)
    train_loader, validation_loader, test_loader = build_loaders(
        args.cache_dir,
        split_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=device.type == "cuda",
        seed=args.seed,
    )
    model = create_model(name, obs_channels=args.obs_channels).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    history = []
    best_decision_accuracy = -1.0
    stale_epochs = 0

    print(f"\n{'=' * 70}\n训练 {name} | 参数量 {parameters:,} | 设备 {device}\n{'=' * 70}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        progress = tqdm(train_loader, desc=f"{name} {epoch}/{args.epochs}", dynamic_ncols=True)
        for inputs, targets in progress:
            inputs, targets = move_batch(inputs, targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = masked_cross_entropy(
                    logits, targets, inputs["action_mask"], args.label_smoothing
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.shape[0]
            train_loss += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_samples += batch_size
            progress.set_postfix(
                loss=f"{train_loss / train_samples:.4f}",
                acc=f"{train_correct / train_samples:.4f}",
            )

        validation = evaluate_model(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(train_samples, 1),
            "train_accuracy": train_correct / max(train_samples, 1),
            "validation": validation,
        }
        history.append(record)
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Epoch {epoch:02d} | train {record['train_accuracy']:.4%} | "
            f"val {validation['accuracy']:.4%} | decision {validation['decision_accuracy']:.4%} | "
            f"top3 {validation['top3_accuracy']:.4%} | "
            f"loss {validation['loss']:.5f}"
        )

        if validation["decision_accuracy"] > best_decision_accuracy:
            best_decision_accuracy = validation["decision_accuracy"]
            stale_epochs = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_name": name,
                    "obs_channels": args.obs_channels,
                    "observation_key": args.observation_key,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "parameters": parameters,
                    "validation_metrics": validation,
                    "training_args": serializable_training_args(args),
                },
                best_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            print(f"{name} 连续 {args.patience} 轮未提升，提前停止。")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device)
    checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, best_path)
    print(
        f"{name} 测试决策准确率：{test_metrics['decision_accuracy']:.4%}，"
        f"总准确率：{test_metrics['accuracy']:.4%}，模型保存到 {best_path}"
    )

    del model, optimizer, scheduler, scaler, train_loader, validation_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model_name": name,
        "checkpoint": str(best_path),
        "parameters": parameters,
        "best_validation_decision_accuracy": best_decision_accuracy,
        "test_metrics": test_metrics,
    }


def main():
    args = parse_args()
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("验证集比例与测试集比例之和必须小于 1")
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = prepare_cache(
        args.data,
        args.cache_dir,
        observation_key=args.observation_key,
    )
    args.obs_channels = int(
        np.load(args.cache_dir / "observations.npy", mmap_mode="r").shape[1]
    )
    split_paths = prepare_splits(
        args.cache_dir,
        metadata["samples"],
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    started = time.time()
    summaries = []
    for name in args.models:
        summaries.append(train_one_model(name, args, split_paths, device))
    summaries.sort(key=lambda item: item["test_metrics"]["decision_accuracy"], reverse=True)
    report = {
        "data": str(args.data),
        "observation_key": args.observation_key,
        "obs_channels": args.obs_channels,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "models": summaries,
    }
    report_path = args.output_dir / "training_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n训练全部完成。当前测试集排名：")
    for rank, item in enumerate(summaries, 1):
        print(
            f"{rank}. {item['model_name']}: "
            f"决策 {item['test_metrics']['decision_accuracy']:.4%} / "
            f"总计 {item['test_metrics']['accuracy']:.4%}"
        )
    print(f"汇总报告：{report_path}")


if __name__ == "__main__":
    main()
