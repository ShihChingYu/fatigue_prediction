"""Test the Inference Job logic."""

import typing as T
from typing import cast
from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
from mlflow.entities.model_registry import ModelVersion

from fatigue import jobs
from fatigue.io import datasets, registries, services

# --- FIXTURES ---


@pytest.fixture
def model_version(mlflow_service: services.MlflowService) -> ModelVersion:
    """Create a dummy registered model version for testing inference."""
    client = mlflow_service.client()
    name = mlflow_service.registry_name

    # 1. Ensure Model Exists
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass

    # 2. Create Dummy Version
    # We create a version but we don't need real weights because we mock the loader below
    mv = client.create_model_version(
        name=name, source="run_id_placeholder", run_id="run_id_placeholder"
    )
    return mv


@pytest.fixture
def mock_loader(mocker):
    """Mock the Model Loader to return a fake model object."""
    # We mock this because we don't want to actually load a heavy model from disk in unit tests
    fake_model = mocker.Mock()
    # Mock predict to return random floats
    fake_model.predict.return_value = pd.DataFrame(
        {"prediction": [0.1, 0.9, 0.2, 0.8, 0.1] * 2}
    )  # 10 rows total

    loader = mocker.Mock(spec=registries.CustomLoader)
    loader.load.return_value = fake_model
    return loader


@pytest.fixture
def output_writer(tmp_path) -> datasets.ParquetWriter:
    """Create a temporary writer for predictions."""
    return datasets.ParquetWriter(path=str(tmp_path / "predictions.parquet"))


# --- TESTS ---


def test_inference_job(
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.ParquetReader,  # From conftest (10 rows)
    targets_reader: datasets.ParquetReader,  # Acts as IDs reader (has user_id)
    output_writer: datasets.ParquetWriter,
    model_version: ModelVersion,  # Created above
    mock_loader: registries.CustomLoader,  # Mocked above
) -> None:
    # --- 0. FORCE GLOBAL STATE ---
    mlflow.set_tracking_uri(mlflow_service.tracking_uri)
    mlflow.set_registry_uri(mlflow_service.registry_uri)
    # -----------------------------

    # given
    job = jobs.InferenceJob(
        logger_service=logger_service,
        alerts_service=alerts_service,
        mlflow_service=mlflow_service,
        # Data
        inputs=inputs_reader,
        ids=targets_reader,  # We use targets because it has 'user_id' and 'HRTIME'
        outputs=output_writer,
        # Model
        alias_or_version=model_version.version,
        loader=mock_loader,  # Inject mock so we don't fail on loading fake model URI
        # Config
        limit=None,
        fatigue_threshold=0.5,
    )

    # when
    out = job.run()

    # then
    # --- 1. Verify Output File ---mock_loader
    assert "results" in out
    results_df = out["results"]  # The dataframe just written

    # Check dimensions
    assert len(results_df) == 10

    # Check Columns (Result Stitching)
    assert "user_id" in results_df.columns, "Should retain IDs"
    assert "fatigue_probability" in results_df.columns
    assert "fatigue_event_alert" in results_df.columns
    assert "status" in results_df.columns

    # --- 2. Verify MLflow Logging ---
    client = mlflow_service.client()
    run_id = out["run"].info.run_id
    run = client.get_run(run_id)

    # Check metrics logged
    assert "inf_total_samples" in run.data.metrics
    assert run.data.metrics["inf_total_samples"] == 10.0

    # --- 3. Verify Model Interaction ---
    # Ensure the loader was called with the correct URI structure
    expected_uri = f"models:/{mlflow_service.registry_name}/{model_version.version}"
    cast(MagicMock, mock_loader.load).assert_called_with(uri=expected_uri)

    load_mock = T.cast(MagicMock, mock_loader.load)
    load_mock.assert_called_with(uri=expected_uri)
