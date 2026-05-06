"""Shared helpers for simple tabular CSV training scripts."""

from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def load_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return list(reader.fieldnames), rows


def infer_target_column(columns: Sequence[str]) -> str:
    preferred = ["Score", "target", "label", "y"]
    lower_map = {column.lower(): column for column in columns}
    for candidate in preferred:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if not columns:
        raise ValueError("CSV file has no columns.")
    return columns[-1]


def train_test_indices(count: int, test_size: float = 0.25, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("Need at least two rows to train a model.")
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    split = int(math.floor(count * (1.0 - test_size)))
    split = max(1, min(count - 1, split))
    return indices[:split], indices[split:]


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_datetime(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    attempts = [
        text,
        text.replace(" ", "T"),
    ]
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]

    for candidate in attempts:
        try:
            return dt.datetime.fromisoformat(candidate).timestamp()
        except ValueError:
            pass

    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def mean_and_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return 0.0, 1.0
    mean = float(array.mean())
    std = float(array.std())
    if std == 0.0:
        std = 1.0
    return mean, std


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if total == 0.0 else float(1.0 - np.sum(residuals ** 2) / total)
    return {"mae": mae, "rmse": rmse, "r2": r2}


@dataclass
class TabularPreprocessor:
    numeric_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    datetime_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    category_values: Dict[str, List[str]] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    def fit(self, rows: Sequence[Dict[str, str]], feature_columns: Sequence[str]) -> "TabularPreprocessor":
        self.numeric_columns = []
        self.datetime_columns = []
        self.categorical_columns = []
        self.numeric_stats = {}
        self.datetime_stats = {}
        self.category_values = {}
        self.feature_names = []

        for column in feature_columns:
            values = [row.get(column, "") for row in rows]
            lower_name = column.lower()

            numeric_values = [parsed for value in values if (parsed := parse_float(value)) is not None]
            datetime_values = [parsed for value in values if (parsed := parse_datetime(value)) is not None]
            non_empty_count = len([value for value in values if str(value).strip()])

            if values and len(numeric_values) >= max(1, int(0.8 * non_empty_count)):
                self.numeric_columns.append(column)
                self.numeric_stats[column] = mean_and_std(numeric_values)
                self.feature_names.append(column)
                continue

            if ("date" in lower_name or "time" in lower_name) and len(datetime_values) >= max(1, int(0.8 * non_empty_count)):
                self.datetime_columns.append(column)
                self.datetime_stats[column] = mean_and_std(datetime_values)
                self.feature_names.append(column)
                continue

            self.categorical_columns.append(column)
            categories = sorted({str(value).strip() or "<missing>" for value in values})
            self.category_values[column] = categories
            self.feature_names.extend([f"{column}={category}" for category in categories])

        return self

    def transform(self, rows: Sequence[Dict[str, str]]) -> np.ndarray:
        if not self.feature_names:
            raise ValueError("Preprocessor has not been fit yet.")

        matrix: List[List[float]] = []
        for row in rows:
            features: List[float] = []

            for column in self.numeric_columns:
                value = parse_float(row.get(column, ""))
                mean, std = self.numeric_stats[column]
                normalized = (value if value is not None else mean) - mean
                features.append(float(normalized / std))

            for column in self.datetime_columns:
                value = parse_datetime(row.get(column, ""))
                mean, std = self.datetime_stats[column]
                normalized = (value if value is not None else mean) - mean
                features.append(float(normalized / std))

            for column in self.categorical_columns:
                value = str(row.get(column, "")).strip() or "<missing>"
                for category in self.category_values[column]:
                    features.append(1.0 if value == category else 0.0)

            matrix.append(features)

        return np.asarray(matrix, dtype=float)

    def fit_transform(self, rows: Sequence[Dict[str, str]], feature_columns: Sequence[str]) -> np.ndarray:
        return self.fit(rows, feature_columns).transform(rows)


def extract_target(rows: Sequence[Dict[str, str]], target_column: str) -> np.ndarray:
    target_values: List[float] = []
    for index, row in enumerate(rows):
        raw_value = row.get(target_column, "")
        numeric_value = parse_float(raw_value)
        if numeric_value is None:
            raise ValueError(f"Target column '{target_column}' must be numeric. Row {index + 1} has value {raw_value!r}.")
        target_values.append(numeric_value)
    return np.asarray(target_values, dtype=float)
