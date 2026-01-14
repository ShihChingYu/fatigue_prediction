# %% IMPORTS

import typing as T

import pytest
from typing_extensions import Self

from fatigue.core import models, schemas

# %% MODELS


def test_model(inputs_samples: schemas.ModelInputs) -> None:
    """Test the Abstract Base Class logic (params, abstract methods)."""

    # given
    class MyModel(models.Model):
        KIND: T.Literal["MyModel"] = "MyModel"

        # public
        a: int = 1
        b: int = 2
        # private
        _c: int = 3

        # We use ModelInputs here because that is what your base Model expects
        def fit(self, inputs: schemas.ModelInputs, targets: schemas.Targets) -> Self:
            return self

        def predict(self, inputs: schemas.ModelInputs) -> schemas.Outputs:
            return schemas.Outputs()

    # when
    model = MyModel(a=10)
    params_init = model.get_params()
    params_set_params = model.set_params(b=20).get_params()

    # We verify that abstract methods raise NotImplementedError by default
    with pytest.raises(NotImplementedError) as explain_model_error:
        model.explain_model()
    with pytest.raises(NotImplementedError) as explain_samples_error:
        model.explain_samples(inputs=inputs_samples)
    with pytest.raises(NotImplementedError) as get_internal_model_error:
        model.get_internal_model()

    # then
    assert params_init == {
        "a": 10,
        "b": 2,
    }, "Model should have the given params after init!"

    assert params_set_params == {
        "a": 10,
        "b": 20,
    }, "Model should have the given params after set_params!"

    assert isinstance(explain_model_error.value, NotImplementedError), (
        "Model should raise NotImplementedError for explain_model()!"
    )
    assert isinstance(explain_samples_error.value, NotImplementedError), (
        "Model should raise NotImplementedError for explain_samples()!"
    )
    assert isinstance(get_internal_model_error.value, NotImplementedError), (
        "Model should raise NotImplementedError for get_internal_model()!"
    )


def test_fatigue_random_forest_model(
    train_test_sets: tuple[
        schemas.ModelInputs, schemas.Targets, schemas.ModelInputs, schemas.Targets
    ],
) -> None:
    """Test the concrete Random Forest implementation."""
    # given
    # We use small params to make the test run fast and deterministic
    params = {"max_depth": 3, "n_estimators": 5, "random_state": 42}
    inputs_train, targets_train, inputs_test, _ = train_test_sets

    model = models.FatigueRandomForestModel().set_params(**params)

    # when
    # 1. Test error before fitting (Should fail because it's empty)
    with pytest.raises(ValueError) as not_fitted_error:
        model.get_internal_model()

    # 2. Fit and Predict
    model.fit(inputs=inputs_train, targets=targets_train)
    outputs = model.predict(inputs=inputs_test)

    # 3. Explain (Feature Importance & SHAP)
    shap_values = model.explain_samples(inputs=inputs_test)
    feature_importances = model.explain_model()

    # then
    # - error check
    assert not_fitted_error.match("Model is not fitted yet!"), (
        "Model should raise an error when not fitted!"
    )

    # - model params check
    current_params = model.get_params()
    assert current_params["max_depth"] == params["max_depth"]
    assert current_params["n_estimators"] == params["n_estimators"]

    assert model.get_internal_model() is not None, "Internal model should be fitted!"

    # - outputs check
    assert outputs.ndim == 2, "Outputs should be a dataframe!"
    assert len(outputs) == len(inputs_test), "Outputs should match input length"

    # - shap values check
    assert len(shap_values.index) == len(inputs_test.index), (
        "SHAP values should be the same length as inputs!"
    )
    # The SHAP columns must match the model's active features (excluding user_id)
    assert len(shap_values.columns) == len(model._features_in), (
        "SHAP values should match the number of active features used by the model!"
    )

    # - feature importances check
    # Random Forest importances sum to ~1.0
    assert feature_importances["Importance"].sum() == pytest.approx(1.0, abs=0.01), (
        "Feature importances should add up to approx 1.0!"
    )

    # Ensure 'user_id' was dropped and is NOT in importances
    assert "user_id" not in feature_importances["Feature"].values, (
        "user_id should be dropped and not appear in feature importance!"
    )
