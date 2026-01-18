"""Pytest fixtures for generating dummy data."""

import numpy as np
import pandas as pd
import pytest

from fatigue.io import datasets, services

# --- 1. RAW DATA FIXTURES (Matches Schemas exactly) ---
N_SAMPLES = 10


@pytest.fixture
def sample_raw_hr_data() -> pd.DataFrame:
    """Generate fake Raw Heart Rate data (Schema: id, HRTIME, HR)."""
    return pd.DataFrame(
        {
            "id": range(10),
            "HRTIME": pd.date_range("2023-01-01", periods=10, freq="min").astype(str),
            "HR": np.random.randint(60, 100, 10).astype(float),
        }
    )


@pytest.fixture
def sample_raw_sleep_data() -> pd.DataFrame:
    """Generate fake Raw Sleep data (Schema: id, START, END, rating, comments)."""
    return pd.DataFrame(
        {
            "id": range(5),
            "START": pd.date_range("2023-01-01 22:00", periods=5, freq="h").astype(str),
            "END": pd.date_range("2023-01-01 23:00", periods=5, freq="h").astype(str),
            "rating": [5.0] * 5,
            "comments": ["Good sleep"] * 5,
        }
    )


@pytest.fixture
def sample_raw_pvt_data() -> pd.DataFrame:
    """Generate fake Raw PVT data (Schema: TESTID, TAPTIME, etc)."""
    return pd.DataFrame(
        {
            "id": range(10),
            "TESTID": range(10),
            "TESTSTART": ["2023-01-01 09:00:00"] * 10,
            "TRIALID": range(10),
            "TRIALNAME": ["Trial_1"] * 10,
            "TRIALSTART": ["09:00:00"] * 10,
            "TAPTIME": np.random.randint(200, 500, 10),
        }
    )


# --- 2. PROCESSED DATA READERS (Stage 2) ---


@pytest.fixture
def inputs_reader(tmp_path) -> datasets.ParquetReader:
    """Generate a reader for processed Model Inputs."""
    df = pd.DataFrame(
        {
            "mean_hr_5min": np.random.rand(N_SAMPLES),
            "hr_volatility_5min": np.random.rand(N_SAMPLES),
            "hr_mean_total": np.random.rand(N_SAMPLES),
            "hr_std_total": np.random.rand(N_SAMPLES),
            "hr_zscore": np.random.rand(N_SAMPLES),
            "hr_jumpiness_5min": np.random.rand(N_SAMPLES),
            "stress_cv": np.random.rand(N_SAMPLES),
            "cum_sleep_debt": np.random.rand(N_SAMPLES),
            "hours_awake": np.random.rand(N_SAMPLES),
            "sleep_inertia_idx": np.random.rand(N_SAMPLES),
            "circadian_sin": np.random.rand(N_SAMPLES),
            "circadian_cos": np.random.rand(N_SAMPLES),
            # Note: Schema might restrict user_id, check your schema definition!
            # If your ModelInputsSchema allows user_id, keep this.
            "user_id": ["user_A"] * N_SAMPLES,
        }
    )
    path = tmp_path / "inputs.parquet"
    df.to_parquet(path)
    return datasets.ParquetReader(path=str(path))


@pytest.fixture
def targets_reader(tmp_path) -> datasets.ParquetReader:
    """Generate a reader for Target variables."""
    df = pd.DataFrame(
        {
            "fatigue_score": np.random.rand(N_SAMPLES),
            "user_id": ["user_A"] * N_SAMPLES,
            "HRTIME": pd.date_range("2023-01-01", periods=N_SAMPLES, freq="min"),
        }
    )
    path = tmp_path / "targets.parquet"
    df.to_parquet(path)
    return datasets.ParquetReader(path=str(path))


@pytest.fixture
def outputs_reader(tmp_path) -> datasets.ParquetReader:
    """Generate a reader for Model Predictions."""
    df = pd.DataFrame(
        {
            "prediction": np.random.rand(10),
            "user_id": ["user_A"] * 10,
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="min"),
        }
    )
    path = tmp_path / "outputs.parquet"
    df.to_parquet(path)
    return datasets.ParquetReader(path=str(path))


@pytest.fixture
def logger_service() -> services.LoggerService:
    """Provide a logger service for tests."""
    return services.LoggerService()


@pytest.fixture
def alerts_service() -> services.AlertsService:
    """Provide an alerts service (can be mocked to avoid spamming Slack)."""
    # For tests, we usually keep it enabled to verify it doesn't crash,
    # but you could set enable=False if you want silence.
    return services.AlertsService(enable=False)


@pytest.fixture
def mlflow_service(tmp_path) -> services.MlflowService:
    """Provide the MLflow service pointing to your test experiment."""
    tracking_uri = f"file://{tmp_path}/mlruns"

    return services.MlflowService(
        # Use a local folder for unit tests to avoid network lag/spamming Azure
        # Or use your real Azure URI if you specifically want to test connectivity.
        tracking_uri=str(tracking_uri),
        registry_uri=str(tracking_uri),
        experiment_name="Unit_Tests",
        registry_name="fatigue_test_model",
    )
