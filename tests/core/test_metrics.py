# %% IMPORTS

import pandas as pd
import pytest

from fatigue.core import metrics, schemas

# %% FIXTURES


@pytest.fixture
def regression_data():
    """Mock data for regression testing."""
    targets = pd.DataFrame({"fatigue_score": [0.1, 0.5, 0.9], "index": [0, 1, 2]}).set_index(
        "index"
    )
    outputs = pd.DataFrame({"prediction": [0.1, 0.5, 0.9], "index": [0, 1, 2]}).set_index("index")
    return targets, outputs


@pytest.fixture
def classification_data():
    """
    Mock data for classification testing.
    Target: [0, 1] (Awake, Fatigued)
    Preds:  [0.2, 0.8] (Correct)
    """
    targets = pd.DataFrame({"fatigue_score": [0.0, 1.0], "index": [0, 1]}).set_index("index")
    outputs = pd.DataFrame({"prediction": [0.2, 0.8], "index": [0, 1]}).set_index("index")
    return targets, outputs


# %% TESTS


def test_regression_metrics(regression_data) -> None:
    """Test standard regression metrics (MAE, RMSE)."""
    # given
    targets, outputs = regression_data
    metric = metrics.RegressionMetrics()

    # when
    scores = metric.score(
        targets=schemas.TargetsSchema.validate(targets),
        outputs=schemas.OutputsSchema.validate(outputs),
    )

    # then
    assert scores["mae"] == 0.0
    assert scores["rmse"] == 0.0
    assert scores["r2"] == 1.0


def test_binary_classification_metrics(classification_data) -> None:
    """Test classification metrics (Recall, Precision) with thresholding."""
    # given
    targets, outputs = classification_data

    # We use a threshold of 0.5.
    # Pred 0.2 < 0.5 -> Class 0 (Correct)
    # Pred 0.8 >= 0.5 -> Class 1 (Correct)
    metric = metrics.BinaryClassificationMetrics(threshold=0.5)

    # when
    scores = metric.score(
        targets=schemas.TargetsSchema.validate(targets),
        outputs=schemas.OutputsSchema.validate(outputs),
    )

    # then
    # Since predictions are perfect:
    assert scores["recall_at_0.5"] == 1.0
    assert scores["precision_at_0.5"] == 1.0
    assert scores["accuracy"] == 1.0
    assert scores["roc_auc"] == 1.0


def test_binary_classification_threshold_logic() -> None:
    """Test that changing the threshold changes the result."""
    # given
    # A prediction of 0.4 is technically "Awake" (Class 0) if threshold is 0.5
    # But it is "Fatigued" (Class 1) if threshold is 0.3
    targets = pd.DataFrame({"fatigue_score": [1.0], "index": [0]}).set_index(
        "index"
    )  # Ground Truth: Fatigued
    outputs = pd.DataFrame({"prediction": [0.4], "index": [0]}).set_index("index")  # Pred: 0.4

    metric_high_thresh = metrics.BinaryClassificationMetrics(threshold=0.5)
    metric_low_thresh = metrics.BinaryClassificationMetrics(threshold=0.3)

    # when
    # Case A: Threshold 0.5 -> Predicts 0 (Miss)
    scores_high = metric_high_thresh.score(
        targets=schemas.TargetsSchema.validate(targets),
        outputs=schemas.OutputsSchema.validate(outputs),
    )

    # Case B: Threshold 0.3 -> Predicts 1 (Hit)
    scores_low = metric_low_thresh.score(
        targets=schemas.TargetsSchema.validate(targets),
        outputs=schemas.OutputsSchema.validate(outputs),
    )

    # then
    assert scores_high["recall_at_0.5"] == 0.0  # Missed it
    assert scores_low["recall_at_0.3"] == 1.0  # Caught it
