"""Define a job for explaining the model structure and decisions."""

# %% IMPORTS

import typing as T

import mlflow
import pandas as pd
import pydantic as pdt
import shap

from fatigue.core import schemas
from fatigue.io import datasets, registries, services
from fatigue.jobs import base

# %% JOBS


class ExplanationsJob(base.Job):
    """Generate explanations from the model and a data sample.

    Parameters:
        inputs_samples (datasets.ReaderKind): reader for the samples data.
        models_explanations (datasets.WriterKind): writer for models explanation.
        samples_explanations (datasets.WriterKind): writer for samples explanation.
        alias_or_version (str | int): alias or version for the  model.
        loader (registries.LoaderKind): registry loader for the model.
    """

    KIND: T.Literal["ExplanationsJob"] = "ExplanationsJob"

    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Model Explanability (SHAP)"
    )

    # Samples
    inputs_samples: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Explanations
    models_explanations: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    samples_explanations: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    # Model
    alias_or_version: T.Union[str, int] = "latest"
    # Loader
    loader: registries.LoaderKind = pdt.Field(registries.CustomLoader(), discriminator="KIND")

    def run(self) -> base.Locals:
        # services
        logger = self.logger_service.logger()
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info("With logger: {}", logger)

            raw_samples = self.inputs_samples.read()

            # Log Data Lineage
            mlflow.log_input(
                self.inputs_samples.lineage(name="explanation_samples", data=raw_samples),
                context="explanation",
            )

            # Validate Schema
            inputs_validated = schemas.ModelInputsSchema.check(raw_samples)
            logger.info(f"Loaded {len(inputs_validated)} samples for explanation.")

            # Drop user_id
            # SHAP and the Model Pipeline only want numeric features.
            if "user_id" in inputs_validated.columns:
                inputs_samples = inputs_validated.drop(columns=["user_id"])
            else:
                inputs_samples = inputs_validated

            # 2. Load Model
            model_uri = registries.uri_for_model_alias_or_version(
                name=self.mlflow_service.registry_name,
                alias_or_version=str(self.alias_or_version),
            )

            # [TRACKING] Log Model Lineage
            mlflow.log_param("model_uri", model_uri)
            logger.info(f"Loading model from: {model_uri}")

            # [FIX] Load as PyFunc -> Unwrap -> Get Pipeline
            # We cannot use load_sklearn_model because it is wrapped in your Custom Class.
            loaded_pyfunc = mlflow.pyfunc.load_model(model_uri)

            # unwrapping the 'FatigueWrapper' to get the real sklearn pipeline
            pipeline = loaded_pyfunc.unwrap_python_model().model

            # Split Pipeline: Preprocessor vs. Predictor
            predictor = pipeline[-1]
            preprocessor = pipeline[:-1]

            logger.info(f"Predictor: {type(predictor).__name__}")

            # Transform Data
            # We must scale the data BEFORE giving it to SHAP,
            # because the RF internally sees scaled numbers.
            X_transformed = preprocessor.transform(inputs_samples)

            # Get feature names (if possible) for better plots
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = inputs_samples.columns

            # Global Explanations (Feature Importance)
            logger.info("Generating Global Feature Importance...")

            if hasattr(predictor, "feature_importances_"):
                importances = predictor.feature_importances_
                models_explanations = pd.DataFrame(
                    {"feature": feature_names, "importance": importances}
                ).sort_values(by="importance", ascending=False)

                # Write to parquet
                self.models_explanations.write(data=models_explanations)

                # Log HTML to Azure
                html_table = models_explanations.to_html(index=False, classes="table")
                mlflow.log_text(html_table, "global_feature_importance.html")
            else:
                logger.warning("Model does not support native feature importance.")

            # 6. Local Explanations (SHAP)
            logger.info("Generating SHAP Values...")
            explainer = shap.TreeExplainer(predictor)
            shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                shap_vals_array = shap_values[1]
            else:
                shap_vals_array = shap_values

            # Convert to DataFrame
            samples_explanations = pd.DataFrame(shap_vals_array, columns=feature_names)

            # Save
            self.samples_explanations.write(data=samples_explanations)

            # Log a SHAP Summary Plot to Azure
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_vals_array, X_transformed, feature_names=feature_names, show=False
            )
            plt.tight_layout()
            plt.savefig("shap_summary.png")
            mlflow.log_artifact("shap_summary.png")
            plt.close()

            logger.success("Explanations generated and saved.")

        return locals()
