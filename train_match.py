"""从完整牌谱训练带场况信息的 121 层 Tile Transformer（BC v2）。"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mahjong_env.deal import deal
from mahjong_env.feature import FeatureAgent
from model import count_parameters, create_model, load_checkpoint_model
from train_all import masked_cross_entropy, seed_everything, select_device
from train_rich import initialize_from_old
from training.evaluate import evaluate_model


FORMAT = "mahjong_match_memmap_v2"
MODEL_NAME = "tile_transformer"


def parse_args():
    parser = argparse.ArgumentParser(description="训练带南风场场况的 BC v2")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, default=Path("replays"))
    parser.add_argument("--data", type=Path, default=Path("data/match/data.pkl"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("training_runs/match_tile_transformer"),
    )
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.max_files == 0 or args.max_files < -1:
        raise ValueError("--max-files 应为正数或 -1")
    if args.validation_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("验证集和测试集比例必须为正数")
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("验证集和测试集比例之和必须小于1")
    for name in ("epochs", "batch_size", "patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为正数")
    if args.workers < 0:
        raise ValueError("--workers 不能为负数")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("学习率必须为正数，权重衰减不能为负数")


def specs(samples: int):
    return {
        "match_observation": {
            "file": "match_observations.f32",
            "dtype": "float32",
            "shape": [samples, FeatureAgent.MATCH_OBS_SIZE, 30],
        },
        "rich_observation": {
            "file": "rich_observations.f32",
            "dtype": "float32",
            "shape": [samples, FeatureAgent.RICH_OBS_SIZE, 30],
        },
        "action_mask": {
            "file": "action_masks.u8",
            "dtype": "uint8",
            "shape": [samples, FeatureAgent.ACT_SIZE],
        },
        "target": {
            "file": "targets.i16",
            "dtype": "int16",
            "shape": [samples],
        },
    }


def array_path(data_path: Path, spec):
    return data_path.parent / spec["file"]


def load_manifest(path: Path):
    with path.open("rb") as handle:
        manifest = pickle.load(handle)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} 不是 BC v2 场况数据")
    for spec in manifest["arrays"].values():
        target = array_path(path, spec)
        if not target.exists():
            raise FileNotFoundError(target)
        expected_bytes = math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
        if target.stat().st_size != expected_bytes:
            raise ValueError(
                f"{target} 文件大小异常：应为 {expected_bytes} 字节，"
                f"实际为 {target.stat().st_size} 字节；请加 --rebuild-data 重建"
            )
    return manifest


def replay_arrays(path: Path):
    raw = deal(path)
    state = raw["state"]
    count = len(raw["label"])
    match = np.asarray(state["match_observation"], dtype=np.float32)
    rich = np.asarray(state["rich_observation"], dtype=np.float32)
    masks = np.asarray(state["action_mask"], dtype=np.uint8)
    labels = np.asarray(raw["label"])
    targets = (
        labels.astype(np.int16)
        if labels.ndim == 1
        else labels.argmax(axis=1).astype(np.int16)
    )
    expected = {
        "match": (count, FeatureAgent.MATCH_OBS_SIZE, 30),
        "rich": (count, FeatureAgent.RICH_OBS_SIZE, 30),
        "masks": (count, FeatureAgent.ACT_SIZE),
        "targets": (count,),
    }
    actual = {
        "match": match.shape,
        "rich": rich.shape,
        "masks": masks.shape,
        "targets": targets.shape,
    }
    if actual != expected:
        raise ValueError(f"牌谱样本形状错误：{actual}，预期 {expected}")
    if count and not np.array_equal(match[:, :FeatureAgent.RICH_OBS_SIZE], rich):
        raise ValueError("match_observation 前101层与 rich_observation 不一致")
    if count and not np.all(masks[np.arange(count), targets] > 0):
        raise ValueError("实际动作不在 action mask 中")
    if not np.all(np.isfinite(match)) or np.any(match < 0) or np.any(match > 1):
        raise ValueError("场况特征必须是 0 到 1 的有限值")
    return match, rich, masks, targets


def generate_dataset(args):
    if args.data.exists() and not args.rebuild_data:
        return load_manifest(args.data)
    replay_files = sorted(args.replay_dir.rglob("*.json"))
    if args.max_files > 0:
        replay_files = replay_files[:args.max_files]
    if not replay_files:
        raise FileNotFoundError(f"没有找到牌谱：{args.replay_dir}")

    args.data.parent.mkdir(parents=True, exist_ok=True)
    temporary = {
        key: (args.data.parent / spec["file"]).with_suffix(
            Path(spec["file"]).suffix + ".tmp"
        )
        for key, spec in specs(-1).items()
    }
    handles = {key: path.open("wb") for key, path in temporary.items()}
    files = []
    errors = []
    samples = 0
    started = time.perf_counter()
    try:
        for index, path in enumerate(tqdm(replay_files, desc="生成 BC v2 数据"), 1):
            try:
                match, rich, masks, targets = replay_arrays(path)
                if not len(targets):
                    continue
                match.tofile(handles["match_observation"])
                rich.tofile(handles["rich_observation"])
                masks.tofile(handles["action_mask"])
                targets.tofile(handles["target"])
                files.append(
                    {"path": str(path.resolve()), "start": samples, "samples": len(targets)}
                )
                samples += len(targets)
            except Exception as exc:
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    finally:
        for handle in handles.values():
            handle.close()
    if samples == 0:
        raise RuntimeError("没有生成任何有效样本")

    final_specs = specs(samples)
    for key, spec in final_specs.items():
        expected_bytes = int(np.prod(spec["shape"], dtype=np.int64)) * np.dtype(spec["dtype"]).itemsize
        if temporary[key].stat().st_size != expected_bytes:
            raise RuntimeError(f"临时数组大小错误：{temporary[key]}")
    for key, spec in final_specs.items():
        os.replace(temporary[key], array_path(args.data, spec))
    manifest = {
        "format": FORMAT,
        "version": 2,
        "feature_schema": "public_match_context_v2",
        "samples": samples,
        "files": files,
        "errors": errors,
        "arrays": final_specs,
        "generated_seconds": time.perf_counter() - started,
    }
    with args.data.open("wb") as handle:
        pickle.dump(manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)
    args.data.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def split_files(args, manifest):
    directory = args.data.parent / f"split_seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.npy" for name in ("train", "validation", "test")}
    if all(path.exists() for path in paths.values()) and not args.rebuild_data:
        return paths
    files = manifest["files"]
    if len(files) < 10:
        raise ValueError("至少需要10份有效牌谱")
    order = np.random.RandomState(args.seed).permutation(len(files))
    test_count = max(1, int(len(files) * args.test_ratio))
    validation_count = max(1, int(len(files) * args.validation_ratio))
    groups = {
        "test": order[:test_count],
        "validation": order[test_count:test_count + validation_count],
        "train": order[test_count + validation_count:],
    }
    for name, selected in groups.items():
        values = np.concatenate([
            np.arange(files[int(i)]["start"], files[int(i)]["start"] + files[int(i)]["samples"], dtype=np.int64)
            for i in selected
        ])
        np.save(paths[name], values)
    return paths


class MatchDataset(Dataset):
    def __init__(self, data_path, manifest, indices_path, rich_only=False):
        self.data_path = Path(data_path)
        self.manifest = manifest
        self.indices_path = Path(indices_path)
        self.rich_only = rich_only
        self._open()

    def _open(self):
        key = "rich_observation" if self.rich_only else "match_observation"
        spec = self.manifest["arrays"][key]
        self.observations = np.memmap(
            array_path(self.data_path, spec), mode="r", dtype=spec["dtype"], shape=tuple(spec["shape"])
        )
        mask_spec = self.manifest["arrays"]["action_mask"]
        target_spec = self.manifest["arrays"]["target"]
        self.masks = np.memmap(array_path(self.data_path, mask_spec), mode="r", dtype=mask_spec["dtype"], shape=tuple(mask_spec["shape"]))
        self.targets = np.memmap(array_path(self.data_path, target_spec), mode="r", dtype=target_spec["dtype"], shape=tuple(target_spec["shape"]))
        self.indices = np.load(self.indices_path, mmap_mode="r")

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("observations", "masks", "targets", "indices"):
            state[key] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        inputs = {
            "observation": torch.from_numpy(np.array(self.observations[index], copy=True)),
            "action_mask": torch.from_numpy(np.array(self.masks[index], copy=True)),
        }
        return inputs, torch.tensor(int(self.targets[index]), dtype=torch.long)


def loader(args, manifest, path, rich_only=False, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return DataLoader(
        MatchDataset(args.data, manifest, path, rich_only),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=args.workers,
        pin_memory=select_device(args.device).type == "cuda",
        persistent_workers=args.workers > 0,
    )


def train(args, manifest, splits):
    seed_everything(args.seed)
    device = select_device(args.device)
    base, base_metadata = load_checkpoint_model(args.base_model, map_location=device)
    # load_checkpoint_model 只负责按 map_location 读取权重；新建的模型实例默认
    # 仍在 CPU。评估数据会被 evaluate_model 移到 device，因此这里必须显式
    # 移动基础模型，CUDA 训练时否则会出现 cuda:0 / cpu 混用。
    base = base.to(device).eval()
    if base_metadata.get("model_name") != MODEL_NAME:
        raise ValueError("BC v2 当前只支持从 Tile Transformer 初始化")
    if int(base_metadata.get("obs_channels", 6)) != FeatureAgent.RICH_OBS_SIZE:
        raise ValueError("--base-model 必须是 101 层 rich BC/PPO 模型")

    train_loader = loader(args, manifest, splits["train"], shuffle=True)
    validation_loader = loader(args, manifest, splits["validation"])
    test_loader = loader(args, manifest, splits["test"])
    base_test_loader = loader(args, manifest, splits["test"], rich_only=True)
    base_metrics = evaluate_model(base, base_test_loader, device)

    model = create_model(MODEL_NAME, obs_channels=FeatureAgent.MATCH_OBS_SIZE).to(device)
    initialize_from_old(model, base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    history_path = args.output_dir / "history.json"
    history = []
    best_accuracy = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        samples = 0
        progress = tqdm(train_loader, desc=f"match BC {epoch}/{args.epochs}", dynamic_ncols=True)
        for inputs, targets in progress:
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = masked_cross_entropy(logits, targets, inputs["action_mask"], args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            size = len(targets)
            loss_sum += loss.item() * size
            correct += (logits.argmax(1) == targets).sum().item()
            samples += size
            progress.set_postfix(loss=f"{loss_sum/max(samples,1):.4f}", acc=f"{correct/max(samples,1):.4f}")
        validation = evaluate_model(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": loss_sum / max(samples, 1), "train_accuracy": correct / max(samples, 1), "validation": validation}
        history.append(record)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        checkpoint = {
            "format_version": 2,
            "training_method": "behavior_cloning_match_context",
            "model_name": MODEL_NAME,
            "obs_channels": FeatureAgent.MATCH_OBS_SIZE,
            "observation_key": FeatureAgent.MATCH_OBSERVATION_KEY,
            "feature_schema": manifest["feature_schema"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "parameters": count_parameters(model),
            "validation_metrics": validation,
            "initialized_from": str(args.base_model.resolve()),
        }
        torch.save(checkpoint, last_path)
        if validation["decision_accuracy"] > best_accuracy:
            best_accuracy = validation["decision_accuracy"]
            stale = 0
            torch.save(checkpoint, best_path)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["test_metrics"] = evaluate_model(model, test_loader, device)
    checkpoint["comparison_base_metrics"] = base_metrics
    torch.save(checkpoint, best_path)
    print(f"BC v2 完成：{best_path}")


def main():
    args = parse_args()
    validate_args(args)
    if not args.base_model.exists():
        raise FileNotFoundError(args.base_model)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = generate_dataset(args)
    if args.generate_only:
        return
    train(args, manifest, split_files(args, manifest))


if __name__ == "__main__":
    main()
