"""Define and validate dataframe schemas."""

# %% IMPORTS

import typing as T

import pandas as pd
import pandera as pa
import pandera.typing as papd

# %% TYPES

# Generic type for a dataframe container
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# %% SCHEMAS


class Schema(pa.DataFrameModel):
    """Base class for a dataframe schema.

    Use a schema to type the dataframe object.
    e.g., to communicate and validate its fields.
    """

    class Config:
        """Default configurations for all schemas.

        Parameters:
            coerce (bool): convert data type if possible.
            strict (bool): ensure the data type is correct.
        """

        coerce: bool = True
        strict: bool = True

    @classmethod
    def check(cls: T.Type[TSchema], data: pd.DataFrame) -> papd.DataFrame[TSchema]:
        """Check the dataframe with this schema.

        Args:
            data (pd.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return T.cast(papd.DataFrame[TSchema], cls.validate(data))


# ==========================================
# STAGE 1: RAW DATA SCHEMAS (Per User)
# ==========================================


class HeartRateRawSchema(Schema):
    """Schema for raw heart rate data (before cleaning)."""

    id: papd.Series[papd.Int64] = pa.Field()
    HRTIME: papd.Series[papd.String] = pa.Field()  # Object -> String

    # Nullable=True because count (973,713) < total (974,767)
    HR: papd.Series[papd.Float64] = pa.Field(nullable=True, gt=0)


class SleepRawSchema(Schema):
    """Schema for raw sleep data (before cleaning)."""

    id: papd.Series[papd.Int64] = pa.Field()
    START: papd.Series[papd.String] = pa.Field()
    END: papd.Series[papd.String] = pa.Field()
    rating: papd.Series[papd.Float64] = pa.Field(gt=0)

    # Nullable=True because only 11 out of 86 rows have comments
    comments: papd.Series[papd.String] = pa.Field(nullable=True)


class PVTRawSchema(Schema):
    """Schema for raw Psychomotor Vigilance Task (PVT) data (before cleaning)."""

    id: papd.Series[papd.Int64] = pa.Field()
    TESTID: papd.Series[papd.Int64] = pa.Field()
    TESTSTART: papd.Series[papd.String] = pa.Field()
    TRIALID: papd.Series[papd.Int64] = pa.Field()
    TRIALNAME: papd.Series[papd.String] = pa.Field()
    TRIALSTART: papd.Series[papd.String] = pa.Field()

    # TAPTIME is usually ms, so it must be positive
    TAPTIME: papd.Series[papd.Int64] = pa.Field(ge=0)


# ==========================================
# STAGE 2: MODEL DATA SCHEMAS (Dataset Level)
# ==========================================


class ModelInputsSchema(Schema):
    """Schema for the project inputs."""

    # Index appears to be integer based (125 to 1337)
    index: papd.Index[papd.Int64] = pa.Field(check_name=False)

    # Features
    mean_hr_5min: papd.Series[papd.Float64] = pa.Field()
    hr_volatility_5min: papd.Series[papd.Float64] = pa.Field()

    # nullable=True is required because count (1004) < total (1032)
    hr_jumpiness_5min: papd.Series[papd.Float64] = pa.Field(nullable=True)

    hr_mean_total: papd.Series[papd.Float64] = pa.Field()
    hr_std_total: papd.Series[papd.Float64] = pa.Field()
    stress_cv: papd.Series[papd.Float64] = pa.Field()
    hours_awake: papd.Series[papd.Float64] = pa.Field()  # Assuming can't be negative
    cum_sleep_debt: papd.Series[papd.Float64] = pa.Field()
    sleep_inertia_idx: papd.Series[papd.Float64] = pa.Field()

    # Mathematical constraints for Sine/Cosine
    circadian_sin: papd.Series[papd.Float64] = pa.Field(ge=-1.0, le=1.0)
    circadian_cos: papd.Series[papd.Float64] = pa.Field(ge=-1.0, le=1.0)

    hr_zscore: papd.Series[papd.Float64] = pa.Field()

    # Categorical/String ID
    user_id: papd.Series[papd.String] = pa.Field()


ModelInputs = papd.DataFrame[ModelInputsSchema]


class TargetsSchema(Schema):
    """Schema for the project target."""

    index: papd.Index[papd.Int64] = pa.Field(check_name=False)

    # Target is bounded 0-1 based on your describe()
    fatigue_score: papd.Series[papd.Float64] = pa.Field(ge=0.0, le=1.0)


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the project output."""

    index: papd.Index[papd.Int64] = pa.Field(check_name=False)
    prediction: papd.Series[papd.Float64] = pa.Field(ge=0.0, le=1.0)


Outputs = papd.DataFrame[OutputsSchema]


class SHAPValuesSchema(Schema):
    """Schema for the project shap values."""

    class Config:
        """Default configurations this schema.

        Parameters:
            dtype (str): dataframe default data type.
            strict (bool): ensure the data type is correct.
        """

        dtype: str = "float32"
        strict: bool = False


SHAPValues = papd.DataFrame[SHAPValuesSchema]


class FeatureImportancesSchema(Schema):
    """Schema for the project feature importances."""

    Feature: papd.Series[papd.String] = pa.Field()
    Importance: papd.Series[papd.Float64] = pa.Field()


FeatureImportances = papd.DataFrame[FeatureImportancesSchema]

Inputs = ModelInputs
