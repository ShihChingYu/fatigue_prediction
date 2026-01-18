"""Tests for Pandera Schemas (Data Validation)."""

import numpy as np
import pandas as pd

from fatigue.core import schemas
from fatigue.io import datasets

# %% RAW DATA SCHEMAS (STAGE 1)


def test_heart_rate_raw_schema(sample_raw_hr_data: pd.DataFrame) -> None:
    """Test that a sample raw HR file passes validation."""
    schema = schemas.HeartRateRawSchema
    assert schema.check(sample_raw_hr_data) is not None


def test_sleep_raw_schema(sample_raw_sleep_data: pd.DataFrame) -> None:
    """Test that a sample raw Sleep file passes validation."""
    schema = schemas.SleepRawSchema
    assert schema.check(sample_raw_sleep_data) is not None


def test_pvt_raw_schema(sample_raw_pvt_data: pd.DataFrame) -> None:
    """Test that a sample raw PVT file passes validation."""
    schema = schemas.PVTRawSchema
    assert schema.check(sample_raw_pvt_data) is not None


# %% MODEL DATA SCHEMAS (STAGE 2)


def test_model_inputs_schema(inputs_reader: datasets.ParquetReader) -> None:
    """Test that the engineered training features pass validation."""
    schema = schemas.ModelInputsSchema
    data = inputs_reader.read()
    assert schema.check(data) is not None


def test_targets_schema(targets_reader: datasets.ParquetReader) -> None:
    """Test that the target variable (fatigue_score) passes validation."""
    schema = schemas.TargetsSchema
    data = targets_reader.read()
    assert schema.check(data) is not None


def test_outputs_schema(outputs_reader: datasets.ParquetReader) -> None:
    """Test that the model predictions pass validation."""
    schema = schemas.OutputsSchema
    data = outputs_reader.read()
    assert schema.check(data) is not None


# %% EXPLANATION SCHEMAS (STAGE 3)


def test_shap_values_schema() -> None:
    """Test valid SHAP values structure (Manual creation)."""
    # given
    # when: we create dummy SHAP data (feature names + values)
    data = pd.DataFrame(
        {
            "mean_hr_5min": np.random.rand(5),
            "hr_volatility_5min": np.random.rand(5),
            "hours_awake": np.random.rand(5),
        }
    )
    validated = schemas.SHAPValuesSchema.validate(data)
    assert isinstance(validated, pd.DataFrame)

    pass


def test_feature_importances_schema() -> None:
    """Test valid Feature Importance structure."""
    # given
    schema = schemas.FeatureImportancesSchema
    # when
    data = pd.DataFrame({"Feature": ["mean_hr", "hours_awake"], "Importance": [0.8, 0.2]})
    # then
    assert schema.check(data) is not None
