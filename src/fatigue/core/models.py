"""Define trainable machine learning models."""

# %% IMPORTS

import abc
import typing as T

import pandas as pd
import pydantic as pdt
import shap
from sklearn import compose, ensemble, impute, pipeline, preprocessing
from typing_extensions import Self, override

# Adjust this import to match your actual package name
from fatigue.core import schemas

# %% TYPES

# Model params
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]

# %% MODELS


class Model(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model."""

    KIND: str

    def get_params(self, deep: bool = True) -> Params:
        """Get the model params.

        Args:
            deep (bool, optional): ignored.

        Returns:
            Params: internal model parameters.
        """
        params: Params = {}
        for key, value in self.model_dump().items():
            if not key.startswith("_") and not key.isupper():
                params[key] = value
        return params

    def set_params(self, **params: ParamValue) -> Self:
        """Set the model params in place.

        Returns:
            T.Self: instance of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abc.abstractmethod
    def fit(self, inputs: schemas.ModelInputs, targets: schemas.Targets) -> Self:
        """Fit the model on the given inputs and targets.

        Args:
            inputs (schemas.ModelInputs): model training inputs.
            targets (schemas.Targets): model training targets.

        Returns:
            T.Self: instance of the model.
        """

    @abc.abstractmethod
    def predict(self, inputs: schemas.ModelInputs) -> schemas.Outputs:
        """Generate outputs with the model for the given inputs.

        Args:
            inputs (schemas.ModelInputs): model prediction inputs.

        Returns:
            schemas.Outputs: model prediction outputs.
        """

    def explain_model(self) -> schemas.FeatureImportances:
        """Explain the internal model structure.

        Returns:
            schemas.FeatureImportances: feature importances.
        """
        raise NotImplementedError()

    def explain_samples(self, inputs: schemas.ModelInputs) -> schemas.SHAPValues:
        """Explain model outputs on input samples.

        Returns:
            schemas.SHAPValues: SHAP values.
        """
        raise NotImplementedError()

    def get_internal_model(self) -> T.Any:
        """Return the internal model in the object.

        Raises:
            NotImplementedError: method not implemented.

        Returns:
            T.Any: any internal model (either empty or fitted).
        """
        raise NotImplementedError()


class FatigueRandomForestModel(Model):
    """Fatigue prediction model based on Random Forest.

    Defaults are set to the best parameters found during prototyping.
    """

    KIND: T.Literal["FatigueRandomForestModel"] = "FatigueRandomForestModel"

    # Hyperparameters (Optimized defaults from prototype.ipynb)
    n_estimators: int = 100
    max_depth: int = 12
    min_samples_split: int = 15
    min_samples_leaf: int = 4
    max_features: T.Union[str, int, float] = "sqrt"
    bootstrap: bool = True
    random_state: int = 42

    # Private Internal State
    _pipeline: T.Optional[pipeline.Pipeline] = None

    # Feature Configuration
    _features_in: list[str] = [
        "mean_hr_5min",
        "hr_volatility_5min",
        "hr_jumpiness_5min",
        "hr_mean_total",
        "hr_std_total",
        "stress_cv",
        "hours_awake",
        "cum_sleep_debt",
        "sleep_inertia_idx",
        "circadian_sin",
        "circadian_cos",
        "hr_zscore",
    ]

    # We explicitly exclude user_id to prevent overfitting to specific people
    _features_exclude: list[str] = ["user_id"]

    @override
    def fit(self, inputs: schemas.ModelInputs, targets: schemas.Targets) -> Self:
        # 1. Define Transformation Steps
        # We assume data is mostly clean, but we add a SimpleImputer just in case
        # NaN enters via hr_jumpiness_5min (which is nullable in schema).
        numerical_transformer = pipeline.Pipeline(
            steps=[
                ("imputer", impute.SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", preprocessing.RobustScaler()),
            ]
        )

        # ColumnTransformer: Selects only valid features, drops user_id
        preprocessor = compose.ColumnTransformer(
            transformers=[("num", numerical_transformer, self._features_in)],
            remainder="drop",  # Drops user_id and index
        )

        # 2. Define the Regressor
        # We use Regressor because the target is a continuous fatigue score (0.0 - 1.0)
        regressor = ensemble.RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,  # Use all cores
        )

        # 3. Build the Pipeline
        self._pipeline = pipeline.Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )

        # 4. Train
        # We fit using the column 'fatigue_score' from the Targets schema
        self._pipeline.fit(X=inputs, y=targets["fatigue_score"])

        return self

    @override
    def predict(self, inputs: schemas.ModelInputs) -> schemas.Outputs:
        model = self.get_internal_model()

        # Predict returns a numpy array
        predictions_array = model.predict(inputs)

        # Format as dataframe matching OutputsSchema
        outputs_df = pd.DataFrame(data={"prediction": predictions_array}, index=inputs.index)

        # Validate output schema
        return schemas.OutputsSchema.check(outputs_df)

    @override
    def explain_model(self) -> schemas.FeatureImportances:
        """Global Feature Importance (Gini Importance)."""
        model = self.get_internal_model()

        # Extract components
        regressor = model.named_steps["regressor"]

        # Get feature names (valid only if preprocessor passes names through)
        # Since we provided specific list `_features_in`, we can use that directly
        features = self._features_in
        importances = regressor.feature_importances_

        # Create DataFrame
        importances_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
            by="Importance", ascending=False
        )

        # Validate
        return schemas.FeatureImportancesSchema.check(importances_df)

    @override
    def explain_samples(self, inputs: schemas.ModelInputs) -> schemas.SHAPValues:
        """Local Explainability using SHAP."""
        model = self.get_internal_model()
        regressor = model.named_steps["regressor"]
        preprocessor = model.named_steps["preprocessor"]

        # 1. Transform inputs to the format the regressor sees (numpy array)
        X_transformed = preprocessor.transform(inputs)

        # 2. Initialize SHAP TreeExplainer
        # Note: TreeExplainer is efficient for Random Forests
        explainer = shap.TreeExplainer(model=regressor)

        # 3. Calculate SHAP values
        # check_additivity=False handles potential minor precision errors in complex pipelines
        shap_values_matrix = explainer.shap_values(X_transformed, check_additivity=False)

        # 4. Format Output
        shap_values_df = pd.DataFrame(
            data=shap_values_matrix, columns=self._features_in, index=inputs.index
        )

        return schemas.SHAPValuesSchema.check(shap_values_df)

    @override
    def get_internal_model(self) -> pipeline.Pipeline:
        if self._pipeline is None:
            raise ValueError("Model is not fitted yet! Call .fit() first.")
        return self._pipeline


ModelKind = FatigueRandomForestModel
