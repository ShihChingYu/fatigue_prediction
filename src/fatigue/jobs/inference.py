"""Define a job for generating batch predictions from a registered model."""

import typing as T

import mlflow
import pydantic as pdt

from fatigue.core import schemas
from fatigue.io import datasets, registries
from fatigue.jobs import base


class InferenceJob(base.Job):
    """Generate batch fatigue predictions from a registered model."""

    KIND: T.Literal["InferenceJob"] = "InferenceJob"

    fatigue_threshold: float = 0.26

    # Inputs
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Outputs
    outputs: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    # Model settings
    alias_or_version: T.Union[str, int] = "latest"
    loader: registries.LoaderKind = pdt.Field(registries.CustomLoader(), discriminator="KIND")

    # Speed control: Limit the number of rows to predict
    limit: T.Optional[int] = 1000

    def run(self) -> base.Locals:
        logger = self.logger_service.logger()

        # 1. Read Inputs
        logger.info("Reading continuous features...")
        data = self.inputs.read()

        # 2. Subset data for speed
        if self.limit:
            logger.info(f"Limiting prediction to first {self.limit} rows.")
            data = data.head(self.limit)

        # Filter the data to only include those columns
        feature_cols = list(schemas.ModelInputsSchema.to_schema().columns.keys())
        model_ready_data = data[feature_cols]

        # 3. Check again - Mean should now be near 0
        inputs = schemas.ModelInputsSchema.check(model_ready_data)

        # 3. Load Model from Registry
        model_uri = registries.uri_for_model_alias_or_version(
            name=self.mlflow_service.registry_name,
            alias_or_version=self.alias_or_version,
        )
        model = self.loader.load(uri=model_uri)
        logger.info(f"Loading model from: {model_uri}")

        # 4 Prediction STAGE 1: Calculate Continuous Scores (Probabilities/Risk)
        logger.info(f"Generating predictions for {len(inputs)} timestamps...")
        outputs = model.predict(inputs=inputs)
        risk_scores = outputs["prediction"]

        # 5 Prediction STAGE 2: Apply Decision Threshold
        binary_alerts = (risk_scores >= self.fatigue_threshold).astype(
            int
        )  # fatigue_threshold = 0.23

        # 5.1. Calculate counts
        total_count = len(binary_alerts)
        fatigue_count = int(binary_alerts.sum())
        awake_count = total_count - fatigue_count

        # 5.2. Calculate percentages for better insight
        fatigue_pct = (fatigue_count / total_count) * 100
        awake_pct = (awake_count / total_count) * 100

        # 5.3. Log to terminal/logger
        logger.info("Inference Summary:")
        logger.info(f" - Total Samples: {total_count}")
        logger.info(f" - Awake Cases:   {awake_count} ({awake_pct:.1f}%)")
        logger.info(f" - Fatigue Cases: {fatigue_count} ({fatigue_pct:.1f}%)")

        # 5.6. Log to MLflow
        mlflow.log_metrics(
            {
                "inf_total_samples": total_count,
                "inf_fatigue_count": fatigue_count,
                "inf_awake_count": awake_count,
                "inf_mean_probability": float(risk_scores.mean()),  # Check the average risk
                "inf_max_probability": float(risk_scores.max()),  # Check the highest risk
                "inf_min_probability": float(risk_scores.min()),  # Check the lowest risk
            }
        )

        # 6. Combine results for final output
        results = data[["HRTIME", "user_id"]].copy()
        results["fatigue_probability"] = risk_scores
        results["fatigue_event_alert"] = binary_alerts
        results["status"] = results["fatigue_event_alert"].map({0: "Awake", 1: "Fatigued"})

        # 7. Write and Notify
        self.outputs.write(data=results)
        logger.success(f"Inference complete. Alerts generated: {binary_alerts.sum()}")

        return locals()
