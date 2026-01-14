"""Evaluate model performances with metrics."""

# %% IMPORTS

from __future__ import annotations

import abc
import typing as T

import mlflow
import numpy as np
import pydantic as pdt
from mlflow.metrics import MetricValue
from sklearn import metrics as skmetrics
from typing_extensions import Annotated, TypeAlias, override

from fatigue.core import models, schemas

# %% TYPINGS

MlflowMetric: TypeAlias = MetricValue
MlflowThreshold: TypeAlias = mlflow.models.MetricThreshold
MlflowModelValidationFailedException: TypeAlias = (
    mlflow.models.evaluation.validation.ModelValidationFailedException
)

# A dictionary of metric names and their values (e.g., {"mae": 0.15})
Metrics = dict[str, float]

# %% METRICS


class Metric(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Use metrics to evaluate model performance."""

    KIND: str

    @abc.abstractmethod
    def score(self, targets: schemas.Targets, outputs: schemas.Outputs) -> Metrics:
        """Score the outputs against the targets.

        Args:
            targets (schemas.Targets): expected values.
            outputs (schemas.Outputs): predicted values.

        Returns:
            Metrics: key-value pairs of metrics.
        """

    def scorer(
        self, model: models.Model, inputs: schemas.ModelInputs, targets: schemas.Targets
    ) -> Metrics:
        """Score model outputs against targets."""
        outputs = model.predict(inputs=inputs)
        return self.score(targets=targets, outputs=outputs)


class RegressionMetrics(Metric):
    """Standard regression metrics (MAE, RMSE, R2)."""

    KIND: T.Literal["RegressionMetrics"] = "RegressionMetrics"

    @override
    def score(self, targets: schemas.Targets, outputs: schemas.Outputs) -> Metrics:
        y_true = targets["fatigue_score"].to_numpy()
        y_pred = outputs["prediction"].to_numpy()

        mae = skmetrics.mean_absolute_error(y_true, y_pred)
        mse = skmetrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = skmetrics.r2_score(y_true, y_pred)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
        }


class BinaryClassificationMetrics(Metric):
    """Classification metrics based on a specific threshold.
    Since the model is a Regressor, we must threshold the outputs to get
    binary class predictions (0 or 1).
    """

    KIND: T.Literal["BinaryClassificationMetrics"] = "BinaryClassificationMetrics"

    # The threshold to trigger an alert
    threshold: float = 0.5

    # What value in the target counts as "Fatigued" (Ground Truth)?
    # Usually 0.5, but configurable if your data is different.
    target_threshold: float = 0.5

    @override
    def score(self, targets: schemas.Targets, outputs: schemas.Outputs) -> Metrics:
        # 1. Get Continuous Scores
        y_true_score = targets["fatigue_score"].to_numpy()
        y_pred_score = outputs["prediction"].to_numpy()

        # 2. Convert to Binary Classes based on Thresholds
        y_true_binary = (y_true_score >= self.target_threshold).astype(int)
        y_pred_binary = (y_pred_score >= self.threshold).astype(int)

        # 3. Calculate Metrics
        # Recall is the priority for safety (finding all fatigued drivers)
        recall = skmetrics.recall_score(y_true_binary, y_pred_binary, zero_division=0)
        precision = skmetrics.precision_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = skmetrics.f1_score(y_true_binary, y_pred_binary, zero_division=0)
        accuracy = skmetrics.accuracy_score(y_true_binary, y_pred_binary)

        # ROC-AUC uses the raw probabilities, not the thresholded binary
        try:
            roc_auc = skmetrics.roc_auc_score(y_true_binary, y_pred_score)
        except ValueError:
            # Handles edge case where only one class is present in the batch
            roc_auc = 0.0

        return {
            f"recall_at_{self.threshold}": float(recall),
            f"precision_at_{self.threshold}": float(precision),
            f"f1_at_{self.threshold}": float(f1),
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
        }


MetricKind = T.Union[RegressionMetrics, BinaryClassificationMetrics]
MetricsKind: TypeAlias = list[Annotated[MetricKind, pdt.Field(discriminator="KIND")]]

# %% THRESHOLDS


class Threshold(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """A project threshold for a metric."""

    threshold: T.Union[int, float]
    greater_is_better: bool

    def to_mlflow(self) -> MlflowThreshold:
        return MlflowThreshold(threshold=self.threshold, greater_is_better=self.greater_is_better)
