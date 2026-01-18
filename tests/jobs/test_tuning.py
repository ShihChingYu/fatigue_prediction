"""Test the Hyperparameter Tuning Job."""

# %% IMPORTS

import _pytest.capture as pc
import numpy as np
import pandas as pd
import pytest

from fatigue import jobs
from fatigue.io import datasets, services

# %% FIXTURES


@pytest.fixture
def tuning_inputs_reader(tmp_path) -> datasets.ParquetReader:
    """Create a temporary Parquet file with dummy input data."""
    # Create data matching ModelInputsSchema exactly
    data = {
        "mean_hr_5min": np.random.rand(20),
        "hr_volatility_5min": np.random.rand(20),
        "hr_mean_total": np.random.rand(20),
        "hr_std_total": np.random.rand(20),
        "hr_zscore": np.random.rand(20),
        "hr_jumpiness_5min": np.random.rand(20),
        "stress_cv": np.random.rand(20),
        "cum_sleep_debt": np.random.rand(20),
        "hours_awake": np.random.rand(20),
        "sleep_inertia_idx": np.random.rand(20),
        "circadian_sin": np.random.rand(20),
        "circadian_cos": np.random.rand(20),
    }
    # Add user_id for GroupKFold (2 groups for n_splits=2)
    data["user_id"] = np.array([1.0] * 10 + [2.0] * 10)

    df = pd.DataFrame(data)
    path = tmp_path / "inputs_sample.parquet"
    df.to_parquet(path)
    return datasets.ParquetReader(path=str(path))


@pytest.fixture
def tuning_targets_reader(tmp_path) -> datasets.ParquetReader:
    """Create a temporary Parquet file with dummy target data."""
    df = pd.DataFrame(
        {
            "fatigue_score": np.random.rand(20),  # Regression target
            "user_id": ["user_A"] * 10 + ["user_B"] * 10,
            "HRTIME": pd.date_range("2023-01-01", periods=20, freq="min"),
        }
    )
    path = tmp_path / "targets_sample.parquet"
    df.to_parquet(path)
    return datasets.ParquetReader(path=str(path))


# %% TESTS


def test_tuning_job(
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    tuning_inputs_reader: datasets.ParquetReader,
    tuning_targets_reader: datasets.ParquetReader,
    capsys: pc.CaptureFixture[str],
) -> None:
    # given
    run_config = services.MlflowService.RunConfig(
        name="TuningTest", tags={"context": "test"}, description="Unit test for TuningJob."
    )

    # We use n_splits=2 because our dummy data only has 2 users (groups)
    # We use n_trials=2 to make the test run instantly
    n_trials = 2
    n_splits = 2

    # when
    job = jobs.TuningJob(
        logger_service=logger_service,
        alerts_service=alerts_service,
        mlflow_service=mlflow_service,
        run_config=run_config,
        inputs=tuning_inputs_reader,
        targets=tuning_targets_reader,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    with job as runner:
        out = runner.run()

    # then
    # 1. Check return variables exist
    assert set(out).issuperset({"best_params", "best_rmse", "best_recall", "study"}), (
        "Output should contain best scores and params"
    )

    # 2. Check Data Logic
    assert out["best_rmse"] >= 0, "RMSE must be positive (or zero)"
    assert 0 <= out["best_recall"] <= 1.0, "Recall must be between 0 and 1"

    # 3. Check Optuna execution
    assert len(out["study"].trials) == n_trials, "Optuna should have run exactly n_trials"

    # 4. Check MLflow Tracking
    client = mlflow_service.client()
    experiment = client.get_experiment_by_name(name=mlflow_service.experiment_name)
    assert experiment is not None, "Experiment should be created"

    # Check that the run was logged
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0, "At least one run (the parent run) should be logged"

    # Check that parameters were logged to MLflow
    last_run = runs[0]

    # Check Lineage
    assert out["inputs"].shape == (20, 12 + 1), "Inputs dataframe shape mismatch"
    assert out["targets"].shape == (20, 3), "Targets dataframe shape mismatch"
    assert "rmse" in last_run.data.metrics
