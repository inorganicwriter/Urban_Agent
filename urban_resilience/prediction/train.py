"""Event-level cross-validation and training orchestration for resilience predictors.

Guarantees strict event-level splitting (no sample from a test event appears in
training/validation folds), guarding against temporal and geographical data leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from urban_resilience.prediction.features import CANONICAL_FEATURE_COLUMNS, TARGET_COLUMNS
from urban_resilience.prediction.predictor import ResiliencePredictor

logger = logging.getLogger(__name__)


class EventLevelSplitter:
    """Splits tabular event datasets grouped strictly by event_id to prevent data leakage."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=n_splits)

    def split(
        self,
        df: pd.DataFrame,
        event_col: str = "event_id",
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yields train and test index arrays ensuring complete event-level separation."""
        if event_col not in df.columns:
            raise ValueError(f"Required group column '{event_col}' not found in DataFrame.")

        unique_events = df[event_col].unique()
        actual_splits = min(self.n_splits, len(unique_events))
        if actual_splits < 2:
            raise ValueError(f"Need at least 2 distinct events for event-level split. Found: {len(unique_events)}")

        groups = df[event_col].values
        gkf = GroupKFold(n_splits=actual_splits)
        for train_idx, val_idx in gkf.split(df, groups=groups):
            train_events = set(df.iloc[train_idx][event_col])
            val_events = set(df.iloc[val_idx][event_col])
            overlap = train_events.intersection(val_events)
            if overlap:
                raise RuntimeError(f"Event-level leakage detected! Overlapping events: {overlap}")
            yield train_idx, val_idx


def evaluate_event_holdout(
    predictor: ResiliencePredictor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Dict[str, float]:
    """Computes evaluation metrics on an unseen event holdout dataset."""
    preds = predictor.predict(X_test)
    metrics = {}

    for target in TARGET_COLUMNS:
        y_true = y_test[target].values
        y_pred = preds[target]

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        metrics[f"{target}_rmse"] = round(rmse, 4)
        metrics[f"{target}_mae"] = round(mae, 4)

    return metrics


def train_resilience_model(
    train_df: pd.DataFrame,
    model_type: str = "hist_gbdt",
    model_version: str = "1.0.0",
    event_col: str = "event_id",
) -> Tuple[ResiliencePredictor, Dict[str, Any]]:
    """Trains a multi-target ResiliencePredictor with event-level cross-validation validation.

    Args:
        train_df: Full DataFrame containing CANONICAL_FEATURE_COLUMNS, TARGET_COLUMNS, and event_col.
        model_type: 'hist_gbdt', 'random_forest', or 'ridge'.
        model_version: String identifier for trained model version.
        event_col: Column name identifying event grouping.

    Returns:
        Tuple of (fitted ResiliencePredictor, cv_metrics_dict).
    """
    splitter = EventLevelSplitter(n_splits=3)
    cv_scores: List[Dict[str, float]] = []

    # Run CV if multiple events exist
    if len(train_df[event_col].unique()) >= 2:
        for fold, (t_idx, v_idx) in enumerate(splitter.split(train_df, event_col=event_col)):
            fold_train = train_df.iloc[t_idx]
            fold_val = train_df.iloc[v_idx]

            fold_model = ResiliencePredictor(model_type=model_type, model_version=f"{model_version}_fold{fold}")
            fold_model.fit(fold_train[CANONICAL_FEATURE_COLUMNS], fold_train[TARGET_COLUMNS])
            scores = evaluate_event_holdout(fold_model, fold_val[CANONICAL_FEATURE_COLUMNS], fold_val[TARGET_COLUMNS])
            cv_scores.append(scores)

    # Train final model on full dataset
    final_model = ResiliencePredictor(model_type=model_type, model_version=model_version)
    final_model.fit(train_df[CANONICAL_FEATURE_COLUMNS], train_df[TARGET_COLUMNS])

    summary_metrics = {
        "model_type": model_type,
        "model_version": model_version,
        "sample_count": len(train_df),
        "event_count": len(train_df[event_col].unique()),
        "cv_folds": len(cv_scores),
    }

    if cv_scores:
        for key in cv_scores[0].keys():
            summary_metrics[f"mean_{key}"] = round(float(np.mean([s[key] for s in cv_scores])), 4)

    return final_model, summary_metrics
