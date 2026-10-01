"""全量生成新 observation 数据，并只训练 Tile Transformer。

data.pkl 保存数据集清单，大数组以连续内存映射文件存放。
这样可以一次遍历全部牌谱，不需要把十几 GB 特征同时放入内存。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mahjong_env.deal import deal
from mahjong_env.feature import FeatureAgent
from model import count_parameters, create_model, load_checkpoint_model
from train_all import masked_cross_entropy, seed_everything, select_device
from training.evaluate import evaluate_model


DATASET_FORMAT = "mahjong_rich_memmap_v1"
MODEL_NAME = "tile_transformer"


def parse_args():
    parser = argparse.ArgumentParser(
        description="全量生成 101 层特征 data.pkl，并训练最佳 Tile Transformer"
    )
    parser.add_argument(
        "--replay-dir", type=Path, default=Path("replays")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/rich/data.pkl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training_runs/rich_tile_transformer")
    )
    parser.add_argument(
        "--old-model",
        type=Path,
        default=Path("training_runs/tile_transformer/best.pt"),
    )
    parser.add_argument(
        "--max-files", type=int, default=-1,
        help="-1 表示全部；可设小数做快速测试",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="忽略现有 data.pkl 并重新生成全部数组",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="只生成 data.pkl，不开始训练",
    )
    parser.add_argument(
        "--recover-temp", action="store_true",
        help="恢复已经完整写完、但在最终大小校验时报错的 .tmp 数组",
    )
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="不复用旧模型权重，随机初始化新模型",
    )
    return parser.parse_args()


def validate_args(args):
    if args.max_files == 0 or args.max_files < -1:
        raise ValueError("--max-files 应为正数，或使用 -1 处理全部")
    if args.validation_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("验证集和测试集比例必须大于0")
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("验证集和测试集比例之和必须小于1")
    for name in ("epochs", "batch_size", "patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于0")


def array_specs(sample_count=None):
    first = -1 if sample_count is None else int(sample_count)
    return {
        "observation": {
            "file": "rich_observations.f32",
            "dtype": "float32",
            "shape": [first, FeatureAgent.RICH_OBS_SIZE, 30],
        },
        "legacy_observation": {
            "file": "legacy_observations.f32",
            "dtype": "float32",
            "shape": [first, FeatureAgent.OBS_SIZE, 30],
        },
        "action_mask": {
            "file": "action_masks.u8",
            "dtype": "uint8",
            "shape": [first, FeatureAgent.ACT_SIZE],
        },
        "target": {
            "file": "targets.i16",
            "dtype": "int16",
            "shape": [first],
        },
    }


def resolve_array_path(data_path: Path, spec: Dict) -> Path:
    return data_path.parent / spec["file"]


def expected_nbytes(spec: Dict) -> int:
    # Windows 上 NumPy 的默认整数可能是 32 位。101x30 的全量数据超过
    # 2^31 个元素时，np.prod(list) 会在转成 Python int 之前就已经溢出。
    return math.prod(int(size) for size in spec["shape"]) * np.dtype(
        spec["dtype"]
    ).itemsize


def count_replay_samples(path: Path) -> int:
    """只读牌谱事件并计算样本数，用于崩溃后重建精确的牌谱边界。

    这里不再运行模拟器，因此恢复十几 GB 的完整临时数组时不需要重新做一遍
    特征生成。计数规则与 deal() 中追加训练样本的位置保持一致。
    """
    with path.open("r", encoding="utf-8") as handle:
        replay = json.load(handle)
    actions = replay["data"]["data"]["actions"]
    samples = 0
    is_operation = True
    is_babei_operation = True
    for action in actions:
        if action.get("type") != 1:
            continue
        result = action["result"]
        result_data = result["data"]
        name = result["name"]
        if "RecordNewRound" in name:
            is_operation = True
            is_babei_operation = True
        elif "RecordDiscardTile" in name:
            samples += 1
            if "operations" in result_data:
                is_operation = False
        elif "RecordChiPengGang" in name:
            if result_data["type"] in (1, 2):
                samples += 1
                is_operation = True
        elif "RecordAnGangAddGang" in name:
            samples += 1
        elif "RecordDealTile" in name:
            if not is_babei_operation:
                samples += 2
                is_babei_operation = True
            if not is_operation:
                samples += 2
                is_operation = True
        elif "RecordHule" in name:
            samples += len(result_data["hules"])
            is_babei_operation = True
        elif "RecordBaBei" in name:
            samples += 1
            if "operations" in result_data:
                is_babei_operation = False
    return samples


def finish_dataset(
    args, replay_files, temporary_paths, sample_count, files, errors, started,
    recovered=False,
):
    if sample_count == 0:
        raise RuntimeError("没有生成任何合法样本")
    final_specs = array_specs(sample_count)
    for key, spec in final_specs.items():
        temporary = temporary_paths[key]
        actual = temporary.stat().st_size
        expected = expected_nbytes(spec)
        if actual != expected:
            raise RuntimeError(
                f"临时数组大小不正确：{temporary}（实际 {actual:,}，"
                f"应为 {expected:,} 字节）"
            )

    # 所有数组都验证成功后才逐个替换，避免因某个数组不完整而得到半套正式文件。
    for key, spec in final_specs.items():
        os.replace(temporary_paths[key], resolve_array_path(args.data, spec))

    manifest = {
        "format": DATASET_FORMAT,
        "version": 1,
        "feature_schema": "public_one_hot_v1",
        "samples": sample_count,
        "replay_root": str(args.replay_dir.resolve()),
        "requested_files": len(replay_files),
        "successful_replays": len(files),
        "files": files,
        "errors": errors,
        "arrays": final_specs,
        "generated_seconds": time.perf_counter() - started,
        "recovered_from_temp": recovered,
    }
    temporary_manifest = args.data.with_suffix(args.data.suffix + ".tmp")
    with temporary_manifest.open("wb") as handle:
        pickle.dump(manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_manifest, args.data)
    json_path = args.data.with_suffix(".json")
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_bytes = sum(expected_nbytes(spec) for spec in final_specs.values())
    verb = "恢复" if recovered else "生成"
    print(
        f"data.pkl {verb}完成：{args.data}\n"
        f"成功牌谱 {len(files):,} | 失败 {len(errors):,} | "
        f"样本 {sample_count:,} | 数组 {total_bytes / (1024 ** 3):.2f} GiB"
    )
    return manifest


def recover_temp_dataset(args, replay_files, temporary_paths, started):
    missing = [path for path in temporary_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "无法恢复，缺少临时数组：" + ", ".join(str(path) for path in missing)
        )
    target_spec = array_specs()["target"]
    target_bytes = temporary_paths["target"].stat().st_size
    itemsize = np.dtype(target_spec["dtype"]).itemsize
    if target_bytes % itemsize:
        raise RuntimeError("targets.i16.tmp 的字节数不完整，不能安全恢复")
    sample_count = target_bytes // itemsize
    final_specs = array_specs(sample_count)
    for key, spec in final_specs.items():
        actual = temporary_paths[key].stat().st_size
        expected = expected_nbytes(spec)
        if actual != expected:
            raise RuntimeError(
                f"不能恢复 {temporary_paths[key]}：实际 {actual:,}，"
                f"按 {sample_count:,} 个样本应为 {expected:,} 字节"
            )

    print(
        f"发现完整临时数组，共 {sample_count:,} 个样本；"
        "正在快速扫描牌谱以重建精确的文件边界……"
    )
    files = []
    offset = 0
    for path in tqdm(replay_files, desc="恢复牌谱边界", dynamic_ncols=True):
        count = count_replay_samples(path)
        if count:
            files.append(
                {"path": str(path.resolve()), "start": offset, "samples": count}
            )
            offset += count
    if offset != sample_count:
        raise RuntimeError(
            f"临时数组有 {sample_count:,} 个样本，但牌谱事件计数为 {offset:,}；"
            "为避免生成错位数据，已停止恢复"
        )
    return finish_dataset(
        args, replay_files, temporary_paths, sample_count, files, [], started,
        recovered=True,
    )


def load_dataset_manifest(data_path: Path):
    with data_path.open("rb") as handle:
        manifest = pickle.load(handle)
    if not isinstance(manifest, dict) or manifest.get("format") != DATASET_FORMAT:
        raise ValueError(f"{data_path} 不是新版全量特征 data.pkl")
    for spec in manifest["arrays"].values():
        path = resolve_array_path(data_path, spec)
        if not path.exists():
            raise FileNotFoundError(f"数据数组不存在：{path}")
        if path.stat().st_size != expected_nbytes(spec):
            raise ValueError(f"数据数组大小不正确：{path}")
    return manifest


def validate_replay_data(data, path: Path):
    state = data.get("state", {})
    lengths = {
        "observation": len(state.get("observation", [])),
        "rich_observation": len(state.get("rich_observation", [])),
        "action_mask": len(state.get("action_mask", [])),
        "label": len(data.get("label", [])),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"成对样本数不一致：{lengths}")
    count = lengths["label"]
    if count == 0:
        return None
    legacy = np.asarray(state["observation"], dtype=np.float32)
    rich = np.asarray(state["rich_observation"], dtype=np.float32)
    masks = np.asarray(state["action_mask"], dtype=np.uint8)
    labels = np.asarray(data["label"])
    targets = (
        labels.astype(np.int16)
        if labels.ndim == 1 else np.argmax(labels, axis=1).astype(np.int16)
    )
    if legacy.shape != (count, FeatureAgent.OBS_SIZE, 30):
        raise ValueError(f"旧 observation 形状错误：{legacy.shape}")
    if rich.shape != (count, FeatureAgent.RICH_OBS_SIZE, 30):
        raise ValueError(f"新 observation 形状错误：{rich.shape}")
    if not np.array_equal(legacy, rich[:, :FeatureAgent.OBS_SIZE]):
        raise ValueError("新 observation 前6层与旧 observation 不一致")
    if not np.all((rich == 0) | (rich == 1)):
        raise ValueError("新 observation 含有非 0/1 数值")
    legal = masks[np.arange(count), targets.astype(np.int64)] > 0
    if not np.all(legal):
        raise ValueError(
            f"{int((~legal).sum())} 个实际动作与 action_mask 冲突"
        )
    return legacy, rich, masks, targets


def generate_dataset(args):
    if args.data.exists() and not args.rebuild_data:
        manifest = load_dataset_manifest(args.data)
        print(
            f"复用现有数据：{args.data} | "
            f"{manifest['samples']:,} 样本 | {len(manifest['files'])} 份牌谱"
        )
        return manifest
    if not args.replay_dir.exists():
        raise FileNotFoundError(f"牌谱目录不存在：{args.replay_dir}")

    replay_files = sorted(args.replay_dir.rglob("*.json"))
    if args.max_files > 0:
        replay_files = replay_files[:args.max_files]
    if not replay_files:
        raise FileNotFoundError(f"没有找到 JSON 牌谱：{args.replay_dir}")

    args.data.parent.mkdir(parents=True, exist_ok=True)
    specs = array_specs()
    temporary_paths = {
        key: resolve_array_path(args.data, spec).with_suffix(
            resolve_array_path(args.data, spec).suffix + ".tmp"
        )
        for key, spec in specs.items()
    }
    if args.recover_temp:
        return recover_temp_dataset(
            args, replay_files, temporary_paths, time.perf_counter()
        )
    existing_temps = [path for path in temporary_paths.values() if path.exists()]
    if existing_temps and not args.rebuild_data:
        raise RuntimeError(
            "检测到上次留下的临时数组。若上次已经显示全部牌谱生成完成，请运行 "
            "python train_rich.py --recover-temp；若要放弃临时数据并重做，请加 "
            "--rebuild-data。"
        )
    handles = {
        key: path.open("wb") for key, path in temporary_paths.items()
    }
    files = []
    errors = []
    sample_count = 0
    started = time.perf_counter()
    try:
        for file_index, path in enumerate(replay_files, 1):
            try:
                arrays = validate_replay_data(deal(path), path)
                if arrays is None:
                    continue
                legacy, rich, masks, targets = arrays
                count = len(targets)
                rich.tofile(handles["observation"])
                legacy.tofile(handles["legacy_observation"])
                masks.tofile(handles["action_mask"])
                targets.tofile(handles["target"])
                files.append(
                    {
                        "path": str(path.resolve()),
                        "start": sample_count,
                        "samples": count,
                    }
                )
                sample_count += count
            except Exception as exc:
                errors.append(
                    {
                        "path": str(path.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if file_index == 1 or file_index % 50 == 0 or file_index == len(replay_files):
                elapsed = max(time.perf_counter() - started, 1e-9)
                rate = file_index / elapsed
                remaining_hours = (len(replay_files) - file_index) / max(rate, 1e-9) / 3600
                print(
                    f"生成 {file_index:,}/{len(replay_files):,} | "
                    f"成功 {len(files):,} | 失败 {len(errors):,} | "
                    f"样本 {sample_count:,} | 预计剩余 {remaining_hours:.2f}小时"
                )
    finally:
        for handle in handles.values():
            handle.close()

    return finish_dataset(
        args, replay_files, temporary_paths, sample_count, files, errors, started
    )


def prepare_file_splits(args, manifest):
    split_dir = args.data.parent / f"split_seed_{args.seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: split_dir / f"{name}.npy" for name in ("train", "validation", "test")}
    signature = {
        "samples": manifest["samples"],
        "files": manifest["files"],
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
    }
    metadata_path = split_dir / "metadata.json"
    if metadata_path.exists() and all(path.exists() for path in paths.values()):
        if json.loads(metadata_path.read_text(encoding="utf-8")) == signature:
            return paths

    files = manifest["files"]
    if len(files) < 10:
        raise ValueError("至少需要10份成功牌谱才能划分数据集")
    order = np.random.RandomState(args.seed).permutation(len(files))
    test_files = max(1, int(len(files) * args.test_ratio))
    validation_files = max(1, int(len(files) * args.validation_ratio))
    groups = {
        "test": order[:test_files],
        "validation": order[test_files:test_files + validation_files],
        "train": order[test_files + validation_files:],
    }
    for name, selected in groups.items():
        indices = np.concatenate(
            [
                np.arange(
                    files[int(index)]["start"],
                    files[int(index)]["start"] + files[int(index)]["samples"],
                    dtype=np.int64,
                )
                for index in selected
            ]
        )
        np.save(paths[name], indices)
    metadata_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


class RichMemmapDataset(Dataset):
    def __init__(self, data_path: Path, manifest: Dict, indices_path: Path, legacy=False):
        self.data_path = Path(data_path)
        self.manifest = manifest
        self.indices_path = Path(indices_path)
        self.legacy = legacy
        self.observations = None
        self.masks = None
        self.targets = None
        self.indices = None
        self._open()

    def _memmap(self, key):
        spec = self.manifest["arrays"][key]
        return np.memmap(
            resolve_array_path(self.data_path, spec), mode="r",
            dtype=np.dtype(spec["dtype"]), shape=tuple(spec["shape"]),
        )

    def _open(self):
        key = "legacy_observation" if self.legacy else "observation"
        self.observations = self._memmap(key)
        self.masks = self._memmap("action_mask")
        self.targets = self._memmap("target")
        self.indices = np.load(self.indices_path, mmap_mode="r")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(observations=None, masks=None, targets=None, indices=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        observation = torch.from_numpy(np.array(self.observations[index], copy=True))
        mask = torch.from_numpy(np.array(self.masks[index], copy=True))
        target = torch.tensor(int(self.targets[index]), dtype=torch.long)
        return {"observation": observation, "action_mask": mask}, target


def build_loader(args, manifest, indices_path, legacy=False, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return DataLoader(
        RichMemmapDataset(args.data, manifest, indices_path, legacy=legacy),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=args.workers,
        pin_memory=select_device(args.device).type == "cuda",
        persistent_workers=args.workers > 0,
    )


def initialize_from_old(new_model, old_model):
    """复制旧模型全部可兼容权重，新增 95 个输入通道初始权重为0。"""
    new_state = new_model.state_dict()
    old_state = old_model.state_dict()
    copied = []
    for key, target in new_state.items():
        source = old_state.get(key)
        if source is None:
            continue
        if source.shape == target.shape:
            target.copy_(source)
            copied.append(key)
        elif key == "input_projection.weight" and source.shape[0] == target.shape[0]:
            target.zero_()
            target[:, :source.shape[1]].copy_(source)
            copied.append(key)
    new_model.load_state_dict(new_state)
    return copied


def move_batch(inputs, targets, device):
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    return inputs, targets.to(device, non_blocking=True)


def train_rich_model(args, manifest, splits, device, old_model, old_metadata):
    seed_everything(args.seed)
    train_loader = build_loader(args, manifest, splits["train"], shuffle=True)
    validation_loader = build_loader(args, manifest, splits["validation"])
    test_loader = build_loader(args, manifest, splits["test"])
    legacy_test_loader = build_loader(args, manifest, splits["test"], legacy=True)

    old_model = old_model.to(device).eval()
    old_metrics = evaluate_model(old_model, legacy_test_loader, device)

    model = create_model(MODEL_NAME, obs_channels=FeatureAgent.RICH_OBS_SIZE).to(device)
    copied = []
    if not args.from_scratch:
        copied = initialize_from_old(model, old_model)
    initial_metrics = evaluate_model(model, test_loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    history_path = args.output_dir / "history.json"
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_accuracy = -1.0
    stale = 0
    history = []
    print(
        f"\n训练 {MODEL_NAME} | 新特征 {FeatureAgent.RICH_OBS_SIZE}x30 | "
        f"参数 {parameters:,} | 设备 {device}"
    )
    print(
        f"同一测试集起点：旧模型 {old_metrics['decision_accuracy']:.4%} | "
        f"新模型训练前 {initial_metrics['decision_accuracy']:.4%}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        progress = tqdm(train_loader, desc=f"rich {epoch}/{args.epochs}", dynamic_ncols=True)
        for inputs, targets in progress:
            inputs, targets = move_batch(inputs, targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(inputs)
                loss = masked_cross_entropy(
                    logits, targets, inputs["action_mask"], args.label_smoothing
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            size = targets.shape[0]
            train_loss += loss.item() * size
            train_correct += (logits.argmax(1) == targets).sum().item()
            train_samples += size
            progress.set_postfix(
                loss=f"{train_loss / max(train_samples, 1):.4f}",
                acc=f"{train_correct / max(train_samples, 1):.4f}",
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
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        checkpoint = {
            "format_version": 1,
            "model_name": MODEL_NAME,
            "obs_channels": FeatureAgent.RICH_OBS_SIZE,
            "observation_key": "rich_observation",
            "feature_schema": manifest["feature_schema"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "parameters": parameters,
            "validation_metrics": validation,
            "initialized_from": None if args.from_scratch else str(args.old_model.resolve()),
            "training_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        torch.save(checkpoint, last_path)
        improved = validation["decision_accuracy"] > best_accuracy
        if improved:
            best_accuracy = validation["decision_accuracy"]
            stale = 0
            torch.save(checkpoint, best_path)
        else:
            stale += 1
        scheduler.step()
        print(
            f"Epoch {epoch:02d} | train {record['train_accuracy']:.4%} | "
            f"val decision {validation['decision_accuracy']:.4%} | "
            f"top3 {validation['top3_accuracy']:.4%} | loss {validation['loss']:.5f}"
        )
        if stale >= args.patience:
            print(f"连续 {args.patience} 轮没有提升，提前停止。")
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    new_metrics = evaluate_model(model, test_loader, device)
    best["test_metrics"] = new_metrics
    best["comparison_old_metrics"] = old_metrics
    torch.save(best, best_path)
    return {
        "old_checkpoint": str(args.old_model.resolve()),
        "new_checkpoint": str(best_path.resolve()),
        "old_checkpoint_historical_test": old_metadata.get("test_metrics"),
        "same_test_old": old_metrics,
        "same_test_new_before_training": initial_metrics,
        "same_test_new": new_metrics,
        "decision_accuracy_delta": (
            new_metrics["decision_accuracy"] - old_metrics["decision_accuracy"]
        ),
        "relative_error_reduction": (
            new_metrics["decision_accuracy"] - old_metrics["decision_accuracy"]
        ) / max(1.0 - old_metrics["decision_accuracy"], 1e-12),
        "parameters": parameters,
        "copied_state_tensors": len(copied),
    }


def main():
    args = parse_args()
    validate_args(args)
    if not args.old_model.exists():
        raise FileNotFoundError(f"旧最佳模型不存在：{args.old_model}")
    manifest = generate_dataset(args)
    if args.generate_only:
        return
    splits = prepare_file_splits(args, manifest)
    device = select_device(args.device)
    old_model, old_metadata = load_checkpoint_model(args.old_model, map_location=device)
    if old_metadata.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"--old-model 必须是 {MODEL_NAME}，实际为 {old_metadata.get('model_name')}"
        )
    if int(old_metadata.get("obs_channels", FeatureAgent.OBS_SIZE)) != FeatureAgent.OBS_SIZE:
        raise ValueError("--old-model 必须使用旧 6 层 observation")

    started = time.perf_counter()
    comparison = train_rich_model(
        args, manifest, splits, device, old_model, old_metadata
    )
    comparison.update(
        {
            "data": str(args.data.resolve()),
            "samples": manifest["samples"],
            "successful_replays": len(manifest["files"]),
            "failed_replays": len(manifest["errors"]),
            "split_unit": "replay_file",
            "training_seconds": time.perf_counter() - started,
        }
    )
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    old_accuracy = comparison["same_test_old"]["decision_accuracy"]
    new_accuracy = comparison["same_test_new"]["decision_accuracy"]
    print("\n新旧特征对比完成")
    print(f"旧模型同测试集：{old_accuracy:.4%}")
    print(f"新模型同测试集：{new_accuracy:.4%}")
    print(f"决策准确率提升：{new_accuracy - old_accuracy:+.4%}")
    print(f"相对错误减少：{comparison['relative_error_reduction']:+.2%}")
    print(f"最佳模型：{comparison['new_checkpoint']}")
    print(f"完整报告：{report_path}")


if __name__ == "__main__":
    main()
