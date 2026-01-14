"""Tests for the training job."""

# %% IMPORTS

import typing as T

import pandas as pd

# Internal imports
from fatigue.core import models, schemas
from fatigue.io import datasets, registries, services
from fatigue.jobs import training
from fatigue.utils import signers

# %% HELPERS


class MockReader(datasets.Reader):
    """A helper to feed fake dataframe into the job logic."""

    KIND: T.Literal["MockReader"] = "MockReader"
    _df: pd.DataFrame

    # We use object.__setattr__ to bypass Pydantic immutability checks
    def __init__(self, df: pd.DataFrame):
        object.__setattr__(self, "_df", df)

    def read(self) -> pd.DataFrame:
        return self._df


# %% TESTS


def test_training_job(
    mlflow_service: services.MlflowService,
    inputs: schemas.Inputs,  # From conftest.py
    targets: schemas.Targets,  # From conftest.py
) -> None:
    # 1. Setup Fake Data Readers
    # We wrap the conftest data in our MockReader
    reader_inputs = MockReader(df=inputs)
    reader_targets = MockReader(df=targets)

    # 2. Configure the Job
    # We manually initialize the class, overriding the default ParquetReaders
    job = training.TrainingJob.model_construct(
        # Inject Fake Data for ALL 4 slots
        inputs_train=reader_inputs,
        targets_train=reader_targets,
        inputs_test=reader_inputs,  # Reuse same fake data for test
        targets_test=reader_targets,
        # Fast Hyperparameters for testing
        n_estimators=2,
        max_depth=2,
        # Services
        mlflow_service=mlflow_service,
        run_config=services.MlflowService.RunConfig(name="Test"),
        model=models.FatigueRandomForestModel(),
        saver=registries.CustomSaver(),
        signer=signers.InferSigner(),
        registry=registries.MlflowRegister(),
        fatigue_threshold=0.7,
        test_size=0.2,
        random_state=42,
    )

    # 3. Run the Job
    # This executes the whole pipeline: Read -> Train -> Log -> Register
    results = job.run()

    # 4. Verify Results

    # Check MLflow Logging
    client = mlflow_service.client()
    run_id = results["run"].info.run_id
    run_data = client.get_run(run_id).data

    # - Did we log the metrics we care about?
    assert "val_rmse" in run_data.metrics
    assert "val_recall" in run_data.metrics

    # - Did we log the threshold?
    assert "threshold" in run_data.metrics

    # - Did we register a model version?
    assert "model_version" in results
    print(f"Test Passed! Model Version: {results['model_version'].version}")
