"""Test the Training Job logic."""

# %% IMPORTS

import _pytest.capture as pc
import mlflow

from fatigue import jobs
from fatigue.io import datasets, services

# %% JOBS


def test_training_job(
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.ParquetReader,  # From conftest.py
    targets_reader: datasets.ParquetReader,  # From conftest.py
    capsys: pc.CaptureFixture[str],
) -> None:
    mlflow.set_tracking_uri(mlflow_service.tracking_uri)
    mlflow.set_registry_uri(mlflow_service.registry_uri)
    inputs_df_debug = inputs_reader.read()
    targets_df_debug = targets_reader.read()
    assert len(inputs_df_debug) == len(targets_df_debug)

    run_config = services.MlflowService.RunConfig(
        name="TrainingTest", tags={"context": "test"}, description="Unit test for TrainingJob."
    )

    # We define the hyperparams directly here (mimicking the YAML)
    job = jobs.TrainingJob(
        logger_service=logger_service,
        alerts_service=alerts_service,
        mlflow_service=mlflow_service,
        run_config=run_config,
        inputs_train=inputs_reader,
        targets_train=targets_reader,
        n_estimators=10,
        max_depth=5,
    )

    out = job.run()

    # then
    # --- 1. Variable Existence Check ---
    expected_vars = {
        "run",
        "pipeline",
        "inputs_train",
        "targets_train",
        "model_version",
    }
    assert expected_vars.issubset(out.keys()), f"Missing variables: {expected_vars - out.keys()}"

    # --- 2. Data Integrity ---
    assert out["inputs_train"].shape[0] > 0, "Inputs dataframe should not be empty"
    assert out["targets_train"].shape[0] > 0, "Targets dataframe should not be empty"
    # Verify we are predicting the correct target
    assert "fatigue_score" in out["targets_train"].columns, "Targets should contain fatigue_score"

    # --- 3. MLflow Tracking ---
    client = mlflow_service.client()
    run_id = out["run"].info.run_id
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"

    # --- 4. Model Registry ---
    assert out["model_version"] is not None, "Model Version object should exist"

    reg_name = mlflow_service.registry_name
    mv = client.get_model_version(reg_name, out["model_version"].version)

    assert mv.name == reg_name, "Registered model name mismatch"
    assert mv.run_id == run_id, "Registered model should point to this run"

    # --- 5. Alerting ---
    captured = capsys.readouterr()
    # Your code prints "Training Complete" in the alert title
    assert "Training Complete" in captured.out or "Model Registered" in captured.out
