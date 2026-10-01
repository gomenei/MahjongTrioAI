import gc
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CACHE_VERSION = 1


def _cache_metadata(
    data_path: Path,
    sample_count: int = 0,
    observation_key: str = "observation",
    observation_shape=None,
) -> Dict:
    stat = data_path.stat()
    metadata = {
        "version": CACHE_VERSION,
        "source": str(data_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "samples": sample_count,
        "observation_key": observation_key,
    }
    if observation_shape is not None:
        metadata["observation_shape"] = list(observation_shape)
    return metadata


def prepare_cache(
    data_path: Path,
    cache_dir: Path,
    chunk_size: int = 8192,
    observation_key: str = "observation",
) -> Dict:
    """首次运行时把巨大的 pickle 转成可复用的内存映射数组。"""
    data_path = Path(data_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    required = [cache_dir / "observations.npy", cache_dir / "masks.npy", cache_dir / "targets.npy"]
    expected = _cache_metadata(data_path, observation_key=observation_key)

    if metadata_path.exists() and all(path.exists() for path in required):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = ("version", "source", "source_size", "source_mtime_ns")
        same_source = all(metadata.get(key) == expected[key] for key in comparable)
        cached_key = metadata.get("observation_key", "observation")
        if same_source and cached_key == observation_key:
            return metadata

    print(f"首次运行：正在读取并缓存 {data_path}。这一步只执行一次。")
    with data_path.open("rb") as file:
        raw = pickle.load(file)

    try:
        observations = raw["state"][observation_key]
    except KeyError as exc:
        available = "、".join(raw.get("state", {}).keys())
        raise KeyError(
            f"数据中没有 {observation_key!r}，可用字段：{available}"
        ) from exc
    masks = raw["state"]["action_mask"]
    labels = raw["label"]
    sample_count = len(labels)
    if not (len(observations) == len(masks) == sample_count):
        raise ValueError("observation、action_mask 和 label 样本数不一致")

    if sample_count == 0:
        raise ValueError("训练数据中没有样本")
    observation_shape = tuple(np.asarray(observations[0]).shape)
    if len(observation_shape) != 2 or observation_shape[1] != 30:
        raise ValueError(
            f"observation 形状应为 (channels, 30)，实际为 {observation_shape}"
        )
    obs_memmap = np.lib.format.open_memmap(
        required[0], mode="w+", dtype=np.float32,
        shape=(sample_count, *observation_shape),
    )
    mask_memmap = np.lib.format.open_memmap(
        required[1], mode="w+", dtype=np.uint8, shape=(sample_count, 177)
    )
    target_memmap = np.lib.format.open_memmap(
        required[2], mode="w+", dtype=np.int64, shape=(sample_count,)
    )

    for start in range(0, sample_count, chunk_size):
        end = min(sample_count, start + chunk_size)
        obs_chunk = np.asarray(observations[start:end], dtype=np.float32)
        mask_chunk = np.asarray(masks[start:end], dtype=np.uint8)
        label_chunk = np.asarray(labels[start:end])
        obs_memmap[start:end] = obs_chunk
        mask_memmap[start:end] = mask_chunk
        target_memmap[start:end] = (
            label_chunk.astype(np.int64)
            if label_chunk.ndim == 1
            else np.argmax(label_chunk, axis=1).astype(np.int64)
        )
        if start % (chunk_size * 50) == 0:
            print(f"缓存进度：{end:,}/{sample_count:,}")

    obs_memmap.flush()
    mask_memmap.flush()
    target_memmap.flush()
    del raw, observations, masks, labels, obs_memmap, mask_memmap, target_memmap
    gc.collect()

    metadata = _cache_metadata(
        data_path,
        sample_count,
        observation_key=observation_key,
        observation_shape=observation_shape,
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def prepare_splits(
    cache_dir: Path,
    sample_count: int,
    seed: int = 42,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Dict[str, Path]:
    split_dir = Path(cache_dir) / f"split_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.json"
    paths = {name: split_dir / f"{name}.npy" for name in ("train", "validation", "test")}
    config = {
        "samples": sample_count,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
    }
    if metadata_path.exists() and all(path.exists() for path in paths.values()):
        if json.loads(metadata_path.read_text(encoding="utf-8")) == config:
            return paths

    # 必须沿用旧 deeptrain.py 的 RandomState/permutation，才能让现有 baseline
    # 与新模型共享同一个、baseline 从未训练过的测试集。
    rng = np.random.RandomState(seed)
    indices = rng.permutation(sample_count).astype(np.int64, copy=False)
    test_count = int(sample_count * test_ratio)
    validation_count = int(sample_count * validation_ratio)
    np.save(paths["test"], indices[:test_count])
    np.save(paths["validation"], indices[test_count:test_count + validation_count])
    np.save(paths["train"], indices[test_count + validation_count:])
    metadata_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths


class MahjongMemmapDataset(Dataset):
    def __init__(self, cache_dir: Path, indices_path: Path):
        self.cache_dir = Path(cache_dir)
        self.indices_path = Path(indices_path)
        self._observations = None
        self._masks = None
        self._targets = None
        self._indices = None
        self._open()

    def _open(self):
        self._observations = np.load(self.cache_dir / "observations.npy", mmap_mode="r")
        self._masks = np.load(self.cache_dir / "masks.npy", mmap_mode="r")
        self._targets = np.load(self.cache_dir / "targets.npy", mmap_mode="r")
        self._indices = np.load(self.indices_path, mmap_mode="r")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update({"_observations": None, "_masks": None, "_targets": None, "_indices": None})
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, item):
        index = int(self._indices[item])
        observation = torch.from_numpy(np.array(self._observations[index], copy=True))
        mask = torch.from_numpy(np.array(self._masks[index], copy=True))
        target = torch.tensor(int(self._targets[index]), dtype=torch.long)
        return {"observation": observation, "action_mask": mask}, target


def build_loader(
    cache_dir: Path,
    indices_path: Path,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """为一个固定索引文件构造 DataLoader，供训练和统一比较共同使用。"""
    dataset = MahjongMemmapDataset(cache_dir, indices_path)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def build_loaders(
    cache_dir: Path,
    split_paths: Dict[str, Path],
    batch_size: int,
    workers: int,
    pin_memory: bool,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = build_loader(
        cache_dir, split_paths["train"], batch_size, workers, pin_memory, True, seed
    )
    validation_loader = build_loader(
        cache_dir, split_paths["validation"], batch_size, workers, pin_memory
    )
    test_loader = build_loader(
        cache_dir, split_paths["test"], batch_size, workers, pin_memory
    )
    return train_loader, validation_loader, test_loader
