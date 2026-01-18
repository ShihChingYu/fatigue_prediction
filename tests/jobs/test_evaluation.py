"""Test the Evaluation Job logic."""

import typing as T
from typing import cast
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.entities.model_registry import ModelVersion

from fatigue import jobs
from fatigue.io import datasets, registries, services

# --- FIXTURES ---


@pytest.fixture
def model_version(mlflow_service: services.MlflowService) -> ModelVersion:
    """Create a dummy registered model version."""
    client = mlflow_service.client()
    name = mlflow_service.registry_name

    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass

    mv = client.create_model_version(
        name=name, source="run_id_placeholder", run_id="run_id_placeholder"
    )
    return mv


@pytest.fixture
def mock_loader(mocker):
    """Mock the Model Loader to return a fake model object."""
    fake_model = mocker.Mock()
    fake_model.predict.return_value = pd.DataFrame({"prediction": np.random.uniform(0, 1, 10)})
    mocker.patch("fatigue.io.registries.CustomLoader.load", return_value=fake_model)
    return registries.CustomLoader()


# --- TESTS ---


def test_evaluation_job(
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.ParquetReader,  # From conftest (10 rows)
    targets_reader: datasets.ParquetReader,  # From conftest (10 rows)
    model_version: ModelVersion,  # Created above
    mock_loader: registries.CustomLoader,  # Mocked above
    capsys,
) -> None:
    # --- 0. FORCE GLOBAL STATE (Sync Test & Job) ---
    mlflow.set_tracking_uri(mlflow_service.tracking_uri)
    mlflow.set_registry_uri(mlflow_service.registry_uri)
    # -----------------------------------------------

    # given
    job = jobs.EvaluationsJob(
        logger_service=logger_service,
        alerts_service=alerts_service,
        mlflow_service=mlflow_service,
        # Data
        inputs_test=inputs_reader,
        targets_test=targets_reader,
        # Model Configuration
        alias_or_version=model_version.version,
        loader=mock_loader,
        fatigue_threshold=0.5,
    )

    # when
    out = job.run()

    # then
    # --- 1. Verify Logic Variables ---
    assert "rmse" in out
    assert "report_dict" in out
    # Check that we successfully categorized classes
    assert "Fatigued" in out["report_dict"]
    assert "Awake" in out["report_dict"]

    # --- 2. Verify MLflow Metrics ---
    client = mlflow_service.client()
    run_id = out["run"].info.run_id
    run = client.get_run(run_id)

    # Check that your custom metrics were actually logged
    metrics = run.data.metrics
    assert "test_rmse" in metrics
    assert "test_fatigued_recall" in metrics
    assert "test_accuracy" in metrics

    # --- 3. Verify Artifacts (Plots & Reports) ---
    artifacts = client.list_artifacts(run_id)
    artifact_names = [f.path for f in artifacts]

    assert "test_pr_curve.png" in artifact_names, "PR Curve plot missing!"
    assert "test_performance_report.txt" in artifact_names, "Text Report missing!"

    # --- 4. Verify Model Loading ---
    # Ensure the loader was called with the correct URI structure
    expected_uri = f"models:/{mlflow_service.registry_name}/{model_version.version}"
    cast(MagicMock, mock_loader.load).assert_called_with(uri=expected_uri)

    load_mock = T.cast(MagicMock, mock_loader.load)
    load_mock.assert_called_with(uri=expected_uri)
