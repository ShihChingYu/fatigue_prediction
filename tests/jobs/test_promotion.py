"""Test the Promotion Job logic."""

# %% IMPORTS

import typing as T

import _pytest.capture as pc
import mlflow
import pytest
from mlflow.entities.model_registry import ModelVersion

from fatigue import jobs
from fatigue.io import services

# %% FIXTURES


@pytest.fixture
def model_version(mlflow_service: services.MlflowService) -> ModelVersion:
    """Create a dummy registered model version for testing promotion."""
    client = mlflow_service.client()
    name = mlflow_service.registry_name

    # 1. Ensure the registered model exists
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass  # Already exists

    # 2. Create a dummy version (using a placeholder source)
    # We don't need a real model artifact, just the registry metadata
    mv = client.create_model_version(
        name=name, source="run_id_placeholder", run_id="run_id_placeholder"
    )
    return mv


# %% JOBS


@pytest.mark.parametrize(
    "version_arg",
    [
        None,  # Case A: Auto-select latest version
        1,  # Case B: Explicit version number
        pytest.param(
            999,  # Case C: Non-existent version
            marks=pytest.mark.xfail(
                reason="Version does not exist.",
                raises=(mlflow.exceptions.MlflowException, IndexError),
            ),
        ),
    ],
)
def test_promotion_job(
    version_arg: T.Optional[int],
    mlflow_service: services.MlflowService,
    alerts_service: services.AlertsService,
    logger_service: services.LoggerService,
    model_version: ModelVersion,  # Injected from fixture above
    capsys: pc.CaptureFixture[str],
) -> None:
    mlflow.set_tracking_uri(mlflow_service.tracking_uri)
    mlflow.set_registry_uri(mlflow_service.registry_uri)

    target_stage = "Production"

    # when
    job = jobs.PromotionJob(
        logger_service=logger_service,
        alerts_service=alerts_service,
        mlflow_service=mlflow_service,
        version=version_arg,
        stage=target_stage,
    )

    # Note: We bypass the context manager if it resets attributes (as seen in training)
    # or if your base class is fixed, use 'with job as runner:'
    out = job.run()

    # then
    # --- 1. Variable Existence ---
    expected_vars = {"client", "name", "version", "run"}
    assert expected_vars.issubset(out.keys())

    # --- 2. Logic Verification ---
    client = mlflow_service.client()
    promoted_version = str(out["version"])

    # Refetch the model version from MLflow to check its new state
    updated_mv = client.get_model_version(
        name=mlflow_service.registry_name, version=promoted_version
    )

    # [CRITICAL] Check that the Stage was updated (Not Alias)
    assert updated_mv.current_stage == target_stage, (
        f"Model should be in {target_stage} stage, but is {updated_mv.current_stage}"
    )

    # --- 3. Audit Trail Checks ---
    # Check if we logged parameters to the MLflow run
    run_id = out["run"].info.run_id
    run = client.get_run(run_id)

    assert run.data.params["promoted_version"] == promoted_version
    assert run.data.params["promoted_alias"] == target_stage

    # --- 4. Alerting ---
    captured = capsys.readouterr()
    assert "Promotion Job Finished" in captured.out
