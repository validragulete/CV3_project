from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TelemetryWindowDataset(Dataset):

    def __init__(self, images: np.ndarray, labels: np.ndarray, scenario_ids: list[str]):
        assert images.ndim == 4 and images.shape[1] == 1
        assert len(images) == len(labels) == len(scenario_ids)
        self.images = torch.from_numpy(np.asarray(images, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.scenario_ids = scenario_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


@dataclass
class DataBundle:
    loaders: dict[str, DataLoader]
    datasets: dict[str, TelemetryWindowDataset]
    feature_names: list[str]
    split_groups: dict[str, list[str]]
    summary: dict[str, dict[str, int]]


def prepare_data(
    csv_path: str | Path,
    splits: dict[str, list[str]],
    window: int = 32,
    step: int = 16,
    batch_size: int = 32,
    seed: int = 42,
) -> DataBundle:
    if window < 4 or window % 4 != 0 or step <= 0:
        raise ValueError("window must be a multiple of 4 (>=4); step must be positive")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("splits must have exactly train, val, test")
    groups = {k: [str(x) for x in v] for k, v in splits.items()}
    all_groups = [g for ids in groups.values() for g in ids]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("Split group IDs must be disjoint across train/val/test")

    df = pd.read_csv(csv_path, low_memory=False)
    needed = {"scenario", "timestamp", "label"}
    if missing := (needed - set(df.columns)):
        raise ValueError(f"Missing canonical CSV columns: {sorted(missing)}")
    df["scenario"] = df["scenario"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamp: inspect source CSV and timestamp format")
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Only verified binary labels (0,1) are supported")
    known_groups = set(df["scenario"].unique())
    if unused := set(all_groups) - known_groups:
        raise ValueError(f"Requested scenario(s) do not exist: {sorted(unused)}")
    if extra := known_groups - set(all_groups):
        raise ValueError(f"Assign ALL scenarios to a split or filter CSV first: {sorted(extra)}")
    feature_cols = [c for c in df.columns if c not in {"scenario", "timestamp", "label"}]
    if not feature_cols:
        raise ValueError("No numeric telemetry features found")
    for name in feature_cols:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    train = df[df.scenario.isin(groups["train"])]
    if train.empty:
        raise ValueError("Train split is empty")
    before = list(feature_cols)
    feature_cols = [c for c in feature_cols
                    if train[c].notna().sum() >= 5
                    and train[c].nunique(dropna=True) > 1]
    removed = sorted(set(before) - set(feature_cols))
    if removed:
        print('Dropped empty/constant features based on TRAIN only:', removed)
    if len(feature_cols) < 4:
        raise ValueError(f'Train has only {len(feature_cols)} informative features: '
                         f'{feature_cols}. Check telemetry aggregation.')
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(train[feature_cols])
    scaler.fit(x_train)
    padded_features = int(np.ceil(len(feature_cols) / 4) * 4)

    datasets = {}
    summary = {}
    for part in ("train", "val", "test"):
        subset = df[df.scenario.isin(groups[part])].copy()
        if subset.empty:
            raise ValueError(f"{part} split is empty")
        xs = scaler.transform(imputer.transform(subset[feature_cols]))
        xs = np.clip(xs, -5.0, 5.0).astype(np.float32)
        subset["__row_number"] = np.arange(len(subset))
        images, y, ids = [], [], []
        for scenario_id, scene in subset.groupby("scenario", sort=False):
            scene = scene.sort_values("timestamp", kind="stable")
            indexes = scene["__row_number"].to_numpy()
            labels = scene["label"].to_numpy(dtype=np.int64)
            if len(indexes) < window:
                raise ValueError(f"Scenario {scenario_id} has fewer than {window} rows")
            for start in range(0, len(indexes) - window + 1, step):
                chunk = xs[indexes[start:start + window]]
                if padded_features != len(feature_cols):
                    chunk = np.pad(chunk, ((0, 0), (0, padded_features - len(feature_cols))))
                images.append(chunk[None, ...])
                y.append(int(labels[start:start + window].any()))
                ids.append(scenario_id)
        if not images:
            raise ValueError(f"No windows in split {part}")
        dataset = TelemetryWindowDataset(np.stack(images), np.asarray(y), ids)
        datasets[part] = dataset
        summary[part] = {
            "windows": len(dataset),
            "normal": int((dataset.labels == 0).sum()),
            "anomalous": int((dataset.labels == 1).sum()),
            "scenarios": len(groups[part]),
        }

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        part: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(part == "train"),
            num_workers=0,
            generator=generator if part == "train" else None,
        )
        for part, dataset in datasets.items()
    }
    return DataBundle(loaders, datasets, feature_cols, groups, summary)
