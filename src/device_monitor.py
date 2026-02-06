"""Device state evaluation and warning using dynamic thresholds + decision tree."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class DynamicThresholdConfig:
    """Configuration for dynamic thresholding."""

    window: int = 30
    z_score: float = 2.0
    ewma_span: int = 20
    min_periods: int = 10


class DynamicThreshold:
    """Compute dynamic thresholds for multiple metrics."""

    def __init__(self, config: DynamicThresholdConfig | None = None) -> None:
        self.config = config or DynamicThresholdConfig()

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return thresholds per metric using rolling stats and EWMA.

        Args:
            data: DataFrame indexed by time, columns as metrics.

        Returns:
            DataFrame with columns for low/high thresholds per metric.
        """
        cfg = self.config
        thresholds = {}
        for col in data.columns:
            series = data[col].astype(float)
            rolling_mean = series.rolling(cfg.window, min_periods=cfg.min_periods).mean()
            rolling_std = series.rolling(cfg.window, min_periods=cfg.min_periods).std(ddof=0)
            ewma = series.ewm(span=cfg.ewma_span, min_periods=cfg.min_periods).mean()
            center = 0.6 * rolling_mean + 0.4 * ewma
            spread = rolling_std.fillna(0)
            low = center - cfg.z_score * spread
            high = center + cfg.z_score * spread
            thresholds[f"{col}_low"] = low
            thresholds[f"{col}_high"] = high
        return pd.DataFrame(thresholds, index=data.index)


class DeviceStateEvaluator:
    """Evaluate device state using multi-source data and a decision tree model."""

    def __init__(
        self,
        model: DecisionTreeClassifier | None = None,
        threshold_config: DynamicThresholdConfig | None = None,
    ) -> None:
        self.model = model or DecisionTreeClassifier(max_depth=4, random_state=42)
        self.thresholds = DynamicThreshold(threshold_config)
        self._is_fitted = False

    @staticmethod
    def _feature_engineering(data: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw data and thresholds."""
        features = {}
        for col in data.columns:
            low = thresholds[f"{col}_low"]
            high = thresholds[f"{col}_high"]
            value = data[col]
            deviation = np.where(value > high, value - high, np.where(value < low, low - value, 0.0))
            features[f"{col}_deviation"] = deviation
            features[f"{col}_above_high"] = (value > high).astype(int)
            features[f"{col}_below_low"] = (value < low).astype(int)
            features[f"{col}_normalized"] = (value - low) / (high - low).replace(0, np.nan)
        return pd.DataFrame(features, index=data.index).fillna(0)

    def fit(self, data: pd.DataFrame, labels: Iterable[int]) -> None:
        """Fit decision tree using labeled data.

        Labels:
            0 = normal, 1 = warning, 2 = critical
        """
        thresholds = self.thresholds.compute(data)
        features = self._feature_engineering(data, thresholds)
        self.model.fit(features, np.asarray(list(labels)))
        self._is_fitted = True

    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate device state and warning level based on input data.

        Returns DataFrame with columns:
            - state (0/1/2)
            - risk_score (0-1)
        """
        thresholds = self.thresholds.compute(data)
        features = self._feature_engineering(data, thresholds)

        if not self._is_fitted:
            pseudo_labels = self._heuristic_labels(features)
            self.model.fit(features, pseudo_labels)
            self._is_fitted = True

        proba = self.model.predict_proba(features)
        state = self.model.predict(features)
        risk_score = self._risk_score(state, proba)
        return pd.DataFrame({"state": state, "risk_score": risk_score}, index=data.index)

    @staticmethod
    def _risk_score(state: np.ndarray, proba: np.ndarray) -> np.ndarray:
        weights = np.array([0.1, 0.5, 1.0])
        scored = (proba * weights).sum(axis=1)
        return np.clip(scored + 0.1 * state, 0, 1)

    @staticmethod
    def _heuristic_labels(features: pd.DataFrame) -> np.ndarray:
        deviation_sum = features.filter(like="_deviation").sum(axis=1)
        alerts = features.filter(like="_above_high").sum(axis=1) + features.filter(
            like="_below_low"
        ).sum(axis=1)
        labels = np.zeros(len(features), dtype=int)
        labels[(alerts >= 1) | (deviation_sum > deviation_sum.quantile(0.75))] = 1
        labels[(alerts >= 2) | (deviation_sum > deviation_sum.quantile(0.9))] = 2
        return labels


def simulate_device_data(
    periods: int = 200, seed: int = 7
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic multi-source data and labels for demos/tests."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=periods, freq="H")
    vibration = rng.normal(0.5, 0.05, size=periods)
    temperature = rng.normal(45, 2.0, size=periods)
    pressure = rng.normal(1.2, 0.08, size=periods)
    vibration[150:] += rng.normal(0.2, 0.05, size=periods - 150)
    temperature[140:] += rng.normal(5, 1.5, size=periods - 140)
    pressure[160:] += rng.normal(0.3, 0.05, size=periods - 160)

    data = pd.DataFrame(
        {
            "vibration": vibration,
            "temperature": temperature,
            "pressure": pressure,
        },
        index=index,
    )
    labels = pd.Series(np.zeros(periods, dtype=int), index=index)
    labels[(index >= index[140]) & (index < index[160])] = 1
    labels[index >= index[160]] = 2
    return data, labels


if __name__ == "__main__":
    data, labels = simulate_device_data()
    evaluator = DeviceStateEvaluator()
    evaluator.fit(data, labels)
    result = evaluator.evaluate(data)
    print(result.tail(10))
