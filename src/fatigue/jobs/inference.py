"""Define a job for generating batch predictions from a registered model."""

import typing as T

import mlflow
import numpy as np
import pandas as pd
import pandera as pa
import pydantic as pdt

from fatigue.core import schemas
from fatigue.io import datasets, registries, services
from fatigue.jobs import base


class InferenceJob(base.Job):
    """Generate batch fatigue predictions from a registered model."""

    KIND: T.Literal["InferenceJob"] = "InferenceJob"

    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Inference Analysis", description="Batch prediction on new monitoring data."
    )

    fatigue_threshold: float = 0.26

    # Inputs
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    ids: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Outputs
    outputs: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    # Model settings
    alias_or_version: T.Union[str, int] = "latest"
    loader: registries.LoaderKind = pdt.Field(registries.CustomLoader(), discriminator="KIND")

    # Speed control: Limit the number of rows to predict
    limit: T.Optional[int] = 5000

    def run(self) -> base.Locals:
        logger = self.logger_service.logger()

        # Start the MLflow Run
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            # 1. Read Inputs
            logger.info("Reading features and IDs...")
            features_df = self.inputs.read()
            ids_df = self.ids.read()

            # 2. Subset data for speed
            if self.limit:
                logger.info(f"Limiting prediction to first {self.limit} rows.")
                features_df = features_df.head(self.limit)
                ids_df = ids_df.head(self.limit)

            mlflow.log_input(
                self.inputs.lineage(name="inference_features", data=features_df),
                context="inference",
            )

            # Schema Validation
            validation_df = pd.concat([features_df, ids_df[["user_id"]]], axis=1)

            # Ensure columns match Schema exactly
            schemas.ModelInputsSchema.check(validation_df)

            # Prepare for Model
            X_pred = features_df

            X_pred_typed = T.cast(pa.typing.DataFrame[schemas.ModelInputsSchema], X_pred)

            # Load Model
            model_uri = registries.uri_for_model_alias_or_version(
                name=self.mlflow_service.registry_name,
                alias_or_version=str(self.alias_or_version),
            )
            model = self.loader.load(uri=model_uri)
            logger.info(f"Loading model from: {model_uri}")
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("fatigue_threshold", self.fatigue_threshold)

            # 6. Prediction
            logger.info(f"Generating predictions for {len(X_pred)} timestamps...")

            # Handle Pipeline Output Formats
            outputs = model.predict(inputs=X_pred_typed)

            if hasattr(outputs, "values"):
                risk_scores = outputs.values.ravel()
            elif isinstance(outputs, dict) and "prediction" in outputs:
                risk_scores = outputs["prediction"].values.ravel()
            else:
                risk_scores = np.array(outputs).ravel()

            # 7. Apply Thresholds
            binary_alerts = (risk_scores >= self.fatigue_threshold).astype(int)

            # 8. Calculate Stats
            total_count = len(binary_alerts)
            fatigue_count = int(binary_alerts.sum())
            awake_count = total_count - fatigue_count
            fatigue_pct = (fatigue_count / total_count) * 100
            awake_pct = (awake_count / total_count) * 100

            logger.info("Inference Summary:")
            logger.info(f" - Total Samples: {total_count}")
            logger.info(f" - Awake Cases:   {awake_count} ({awake_pct:.1f}%)")
            logger.info(f" - Fatigue Cases: {fatigue_count} ({fatigue_pct:.1f}%)")

            mlflow.log_metrics(
                {
                    "inf_total_samples": total_count,
                    "inf_fatigue_count": fatigue_count,
                    "inf_awake_count": awake_count,
                    "inf_mean_probability": float(risk_scores.mean()),
                }
            )

            # 9. Combine Results (Stitch IDs back to Predictions)
            results = ids_df.copy()
            results["fatigue_probability"] = risk_scores
            results["fatigue_event_alert"] = binary_alerts
            results["status"] = results["fatigue_event_alert"].map({0: "Awake", 1: "Fatigued"})

            # 10. Write
            self.outputs.write(data=results)
            logger.success("Inference results saved.")

        return locals()
