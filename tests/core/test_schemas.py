# %% IMPORTS

import pandas as pd

from fatigue.core import models, schemas
from fatigue.io import datasets

# %% RAW DATA SCHEMAS (STAGE 1)


def test_heart_rate_raw_schema(sample_raw_hr_data: pd.DataFrame) -> None:
    """Test that a sample raw HR file passes validation."""
    # given
    schema = schemas.HeartRateRawSchema
    # when
    data = sample_raw_hr_data
    # then
    assert schema.check(data) is not None, "Raw HR data should be valid!"


def test_sleep_raw_schema(sample_raw_sleep_data: pd.DataFrame) -> None:
    """Test that a sample raw Sleep file passes validation."""
    # given
    schema = schemas.SleepRawSchema
    # when
    data = sample_raw_sleep_data
    # then
    assert schema.check(data) is not None, "Raw Sleep data should be valid!"


def test_pvt_raw_schema(sample_raw_pvt_data: pd.DataFrame) -> None:
    """Test that a sample raw PVT file passes validation."""
    # given
    schema = schemas.PVTRawSchema
    # when
    data = sample_raw_pvt_data
    # then
    assert schema.check(data) is not None, "Raw PVT data should be valid!"


# %% MODEL DATA SCHEMAS (STAGE 2)


def test_model_inputs_schema(inputs_reader: datasets.Reader) -> None:
    """Test that the engineered training features pass validation."""
    # given
    schema = schemas.ModelInputsSchema
    # when
    data = inputs_reader.read()
    # then
    assert schema.check(data) is not None, "Model inputs data should be valid!"


def test_targets_schema(targets_reader: datasets.Reader) -> None:
    """Test that the target variable (fatigue_score) passes validation."""
    # given
    schema = schemas.TargetsSchema
    # when
    data = targets_reader.read()
    # then
    assert schema.check(data) is not None, "Targets data should be valid!"


def test_outputs_schema(outputs_reader: datasets.Reader) -> None:
    """Test that the model predictions pass validation."""
    # given
    schema = schemas.OutputsSchema
    # when
    data = outputs_reader.read()
    # then
    assert schema.check(data) is not None, "Outputs data should be valid!"


def test_shap_values_schema(
    model: models.Model,
    train_test_sets: tuple[
        schemas.ModelInputs, schemas.Targets, schemas.ModelInputs, schemas.Targets
    ],
) -> None:
    """Test that SHAP values generation returns valid dataframe structure."""
    # given
    schema = schemas.SHAPValuesSchema
    _, _, inputs_test, _ = train_test_sets
    # when
    data = model.explain_samples(inputs=inputs_test)
    # then
    assert schema.check(data) is not None, "SHAP values data should be valid!"


def test_feature_importances_schema(model: models.Model) -> None:
    """Test that global feature importance returns valid dataframe structure."""
    # given
    schema = schemas.FeatureImportancesSchema
    # when
    data = model.explain_model()
    # then
    assert schema.check(data) is not None, "Feature importance data should be valid!"
