"""Test the Explanation Job logic."""

import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
from loguru import logger as loguru_logger
from mlflow.entities.model_registry import ModelVersion
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fatigue import jobs
from fatigue.io import datasets, services

# --- FIXTURES ---


@pytest.fixture
def fake_pipeline(inputs_reader):
    """Create a real (but tiny) Scikit-Learn pipeline for SHAP to analyze."""
    real_data = inputs_reader.read()
    if "user_id" in real_data.columns:
        X = real_data.drop(columns=["user_id"])
    else:
        X = real_data

    y = np.random.rand(len(X))
    # SHAP needs a real tree model to run TreeExplainer. We can't mock this easily.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42)),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def mock_mlflow_load(mocker, fake_pipeline):
    """Mock mlflow.pyfunc.load_model to return our fake pipeline."""
    # 1. Create the Mock wrapper structure
    # Code calls: loaded_pyfunc.unwrap_python_model().model

    mock_wrapper = mocker.Mock()
    mock_wrapper.model = fake_pipeline  # The real sklearn object

    mock_pyfunc = mocker.Mock()
    mock_pyfunc.unwrap_python_model.return_value = mock_wrapper

    # 2. Patch the mlflow load function
    mocker.patch("mlflow.pyfunc.load_model", return_value=mock_pyfunc)
    return mock_pyfunc


@pytest.fixture
def model_version(mlflow_service: services.MlflowService) -> ModelVersion:
    """Create a dummy registered model version."""
    client = mlflow_service.client()
    name = mlflow_service.registry_name
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass
    return client.create_model_version(name, "source", "run_id")


@pytest.fixture
def writers(tmp_path):
    """Create temporary writers for the job outputs."""
    return (
        datasets.ParquetWriter(path=str(tmp_path / "global.parquet")),
        datasets.ParquetWriter(path=str(tmp_path / "local.parquet")),
    )


# --- TEST ---


def test_explanation_job(
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.ParquetReader,  # From conftest (10 rows)
    writers: tuple,
    model_version: ModelVersion,
    mock_mlflow_load,  # The critical mock
    capsys,
) -> None:
    # --- 0. FORCE GLOBAL STATE ---
    mlflow.set_tracking_uri(mlflow_service.tracking_uri)
    mlflow.set_registry_uri(mlflow_service.registry_uri)

    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="INFO")

    global_writer, local_writer = writers

    # given
    job = jobs.ExplanationsJob(
        logger_service=logger_service,
        alerts_service=alerts_service,
        mlflow_service=mlflow_service,
        # Data
        inputs_samples=inputs_reader,
        # Outputs
        models_explanations=global_writer,
        samples_explanations=local_writer,
        # Config
        alias_or_version=model_version.version,
    )

    # when
    out = job.run()

    # then
    # --- 1. Verify Outputs (Global Importance) ---
    global_df = pd.read_parquet(global_writer.path)
    assert not global_df.empty
    assert "importance" in global_df.columns
    # Check that it sorted them (highest importance first)
    assert global_df.iloc[0]["importance"] >= global_df.iloc[-1]["importance"]

    # --- 2. Verify Outputs (Local SHAP) ---
    local_df = pd.read_parquet(local_writer.path)
    assert len(local_df) == 10  # Matches input rows
    # Columns should match feature names (col_0, col_1...)
    # Note: Our fake pipeline has 5 cols, but inputs_reader has 12 cols (conftest).
    # The Scaler will adapt, but SHAP output cols depend on the model.
    # We just check the file was written and has content.
    assert not local_df.empty

    # --- 3. Verify Artifacts ---
    client = mlflow_service.client()
    run_id = out["run"].info.run_id
    artifacts = [f.path for f in client.list_artifacts(run_id)]

    assert "shap_summary.png" in artifacts
    assert "global_feature_importance.html" in artifacts

    # --- 4. Verify Logs ---
    captured = capsys.readouterr()
    all_logs = captured.out + captured.err
    assert "Explanations generated" in all_logs
