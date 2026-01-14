"""Configuration for the tests."""

# %% IMPORTS

import typing as T

import numpy as np
import pandas as pd
import pytest

# Replace 'fatigue' with your package name
from fatigue.core import models, schemas
from fatigue.io import datasets, services
from fatigue.utils import signers

# %% CONFIGS

N_SAMPLES = 50
N_FEATURES = 12

# %% FIXTURES - FAKE DATA GENERATORS


@pytest.fixture(scope="session")
def sample_raw_hr_data() -> pd.DataFrame:
    """Return a fake dataframe matching HeartRateRawSchema."""
    return pd.DataFrame(
        {
            "id": np.random.randint(1, 100, size=N_SAMPLES),
            "HRTIME": ["10:00:00"] * N_SAMPLES,
            # Ensure HR > 0 as per your strict schema
            "HR": np.random.uniform(50, 180, size=N_SAMPLES),
        }
    )


@pytest.fixture(scope="session")
def sample_raw_sleep_data() -> pd.DataFrame:
    """Return a fake dataframe matching SleepRawSchema."""
    return pd.DataFrame(
        {
            "id": np.random.randint(1, 100, size=N_SAMPLES),
            "START": ["22:00:00"] * N_SAMPLES,
            "END": ["06:00:00"] * N_SAMPLES,
            "rating": np.random.uniform(1, 10, size=N_SAMPLES),
            "comments": [None] * N_SAMPLES,
        }
    )


@pytest.fixture(scope="session")
def sample_raw_pvt_data() -> pd.DataFrame:
    """Return a fake dataframe matching PVTRawSchema."""
    return pd.DataFrame(
        {
            "id": np.random.randint(1, 100, size=N_SAMPLES),
            "TESTID": np.arange(N_SAMPLES),
            "TESTSTART": ["12:00:00"] * N_SAMPLES,
            "TRIALID": np.arange(N_SAMPLES),
            "TRIALNAME": ["TEST"] * N_SAMPLES,
            "TRIALSTART": ["12:00:05"] * N_SAMPLES,
            "TAPTIME": np.random.randint(200, 500, size=N_SAMPLES),
        }
    )


@pytest.fixture(scope="session")
def inputs_samples() -> schemas.ModelInputs:
    """Generate fake Model Inputs."""
    df = pd.DataFrame(
        {
            "mean_hr_5min": np.random.randn(N_SAMPLES),
            "hr_volatility_5min": np.random.randn(N_SAMPLES),
            "hr_jumpiness_5min": np.random.randn(N_SAMPLES),
            "hr_mean_total": np.random.randn(N_SAMPLES),
            "hr_std_total": np.random.randn(N_SAMPLES),
            "stress_cv": np.random.randn(N_SAMPLES),
            "hours_awake": np.random.randn(N_SAMPLES),
            "cum_sleep_debt": np.random.randn(N_SAMPLES),
            "sleep_inertia_idx": np.random.randn(N_SAMPLES),
            "circadian_sin": np.random.uniform(-1, 1, size=N_SAMPLES),
            "circadian_cos": np.random.uniform(-1, 1, size=N_SAMPLES),
            "hr_zscore": np.random.randn(N_SAMPLES),
            "user_id": ["user_1"] * N_SAMPLES,
            "index": np.arange(N_SAMPLES),
        }
    )
    df = df.set_index("index")
    return schemas.ModelInputsSchema.validate(df)


@pytest.fixture(scope="session")
def targets_samples() -> schemas.Targets:
    """Generate fake Targets."""
    df = pd.DataFrame(
        {"fatigue_score": np.random.uniform(0, 1, size=N_SAMPLES), "index": np.arange(N_SAMPLES)}
    )
    df = df.set_index("index")
    return schemas.TargetsSchema.validate(df)


@pytest.fixture(scope="session")
def outputs_samples() -> schemas.Outputs:
    """Generate fake Outputs."""
    df = pd.DataFrame(
        {"prediction": np.random.uniform(0, 1, size=N_SAMPLES), "index": np.arange(N_SAMPLES)}
    )
    df = df.set_index("index")
    return schemas.OutputsSchema.validate(df)


# %% FIXTURES - DATASETS (READERS)


# We mock the Readers to return our fake data instead of reading from disk
class MockReader(datasets.Reader):
    """Mock Reader to return in-memory dataframes."""

    KIND: T.Literal["MockReader"] = "MockReader"
    _df: pd.DataFrame = pd.DataFrame()  # Private storage

    def read(self) -> pd.DataFrame:
        return self._df

    def lineage(self, name, data, targets=None, predictions=None):
        return None


@pytest.fixture(scope="session")
def inputs_reader(inputs_samples: pd.DataFrame) -> datasets.Reader:
    reader = MockReader()
    reader._df = inputs_samples
    return reader


@pytest.fixture(scope="session")
def targets_reader(targets_samples: pd.DataFrame) -> datasets.Reader:
    reader = MockReader()
    reader._df = targets_samples
    return reader


@pytest.fixture(scope="session")
def outputs_reader(outputs_samples: pd.DataFrame) -> datasets.Reader:
    reader = MockReader()
    reader._df = outputs_samples
    return reader


# %% FIXTURES - MODEL TRAINING


@pytest.fixture(scope="session")
def train_test_sets(
    inputs_samples: schemas.ModelInputs,
    targets_samples: schemas.Targets,
) -> tuple[schemas.ModelInputs, schemas.Targets, schemas.ModelInputs, schemas.Targets]:
    """Split the fake data into train/test sets."""
    # Simple manual split for testing
    split_idx = int(N_SAMPLES * 0.8)

    inputs_train = inputs_samples.iloc[:split_idx]
    inputs_test = inputs_samples.iloc[split_idx:]

    targets_train = targets_samples.iloc[:split_idx]
    targets_test = targets_samples.iloc[split_idx:]

    return inputs_train, targets_train, inputs_test, targets_test


@pytest.fixture(scope="session")
def model(train_test_sets: tuple) -> models.FatigueRandomForestModel:
    """Return a trained model for testing."""
    model = models.FatigueRandomForestModel(n_estimators=5, max_depth=3)
    inputs_train, targets_train, _, _ = train_test_sets
    model.fit(inputs=inputs_train, targets=targets_train)
    return model


@pytest.fixture
def inputs(inputs_samples):
    """Alias inputs_samples to inputs for generic tests."""
    return inputs_samples


@pytest.fixture
def outputs(outputs_samples):
    """Alias outputs_samples to outputs for generic tests."""
    return outputs_samples


@pytest.fixture
def targets(targets_samples):
    """Alias targets_samples to targets for generic tests."""
    return targets_samples


@pytest.fixture
def signature(inputs, outputs):
    return signers.InferSigner().sign(inputs, outputs)


@pytest.fixture
def mlflow_service():
    # Initialize the real service with default testing values
    service = services.MlflowService(
        tracking_uri="./mlruns_test",  # Separate test folder
        registry_uri="./mlruns_test",
        experiment_name="fatigue_test",
    )
    service.start()
    return service
