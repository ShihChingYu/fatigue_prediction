"""Define a job for evaluating registered models with test data."""

import typing as T

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pydantic as pdt
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    precision_recall_curve,
)

from fatigue.core import schemas
from fatigue.io import datasets, registries, services
from fatigue.jobs import base


class EvaluationsJob(base.Job):
    """Deep evaluation of a registered model using test data."""

    KIND: T.Literal["EvaluationsJob"] = "EvaluationsJob"
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Evaluations"
    )

    # Data - Test Set
    inputs_test: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    targets_test: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    # Model to evaluate
    alias_or_version: T.Union[str, int] = "latest"
    loader: registries.LoaderKind = pdt.Field(registries.CustomLoader(), discriminator="KIND")
    fatigue_threshold: float = 0.26

    def run(self) -> base.Locals:
        logger = self.logger_service.logger()

        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            # 1. Load Model from Registry
            model_uri = registries.uri_for_model_alias_or_version(
                name=self.mlflow_service.registry_name,
                alias_or_version=self.alias_or_version,
            )
            model = self.loader.load(uri=model_uri)
            logger.info(f"Evaluating Model: {model_uri}")

            # --- MODEL LINEAGE ---
            logger.info(f"Loading Model from: {model_uri}")
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("eval_threshold", self.fatigue_threshold)
            model = self.loader.load(uri=model_uri)

            # 2. Read Test Data
            inputs_test = schemas.ModelInputsSchema.check(self.inputs_test.read())
            targets_test = schemas.TargetsSchema.check(self.targets_test.read())
            y_true = targets_test["fatigue_score"]

            # Log the data that "tested" the model
            mlflow.log_input(
                self.inputs_test.lineage(name="inputs_test", data=inputs_test),
                context="testing",
            )

            # Pre-processing: Drop user_id if present
            if "user_id" in inputs_test.columns:
                inputs_test = inputs_test.drop(columns=["user_id"])

            # 3. Predict
            outputs = model.predict(inputs=inputs_test)
            if isinstance(outputs, dict) and "prediction" in outputs:
                y_pred = outputs["prediction"].values.ravel()
            else:
                # returns numpy array
                y_pred = np.array(outputs).ravel()

            # Handle Array vs DataFrame output
            # Pipelines return numpy arrays
            if hasattr(outputs, "values"):
                y_pred = outputs.values.ravel()
            elif isinstance(outputs, dict) and "prediction" in outputs:
                y_pred = outputs["prediction"].values.ravel()
            else:
                y_pred = np.array(outputs).ravel()

            # 4. Deep Metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            y_true_bin = (y_true >= self.fatigue_threshold).astype(int)
            y_pred_bin = (y_pred >= self.fatigue_threshold).astype(int)

            # Generate the STRING version for the text file artifact
            report_str = classification_report(
                y_true_bin, y_pred_bin, target_names=["Awake", "Fatigued"]
            )
            mlflow.log_text(report_str, "test_performance_report.txt")
            logger.info(f"Test Report:\n{report_str}")

            # Generate the DICT version for logging individual metrics
            report_dict = classification_report(
                y_true_bin, y_pred_bin, target_names=["Awake", "Fatigued"], output_dict=True
            )

            # 5. PR-Curve
            precisions, recalls, thresholds = precision_recall_curve(y_true_bin, y_pred)
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
            plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
            plt.axvline(
                x=self.fatigue_threshold,
                color="r",
                linestyle=":",
                label=f"Thresh {self.fatigue_threshold}",
            )
            plt.title("Test Set PR-Curve")
            plt.legend()
            plt.savefig("test_pr_curve.png")
            mlflow.log_artifact("test_pr_curve.png")

            mlflow.log_metrics(
                {
                    # Regression
                    "test_rmse": rmse,
                    # Global Classification
                    "test_accuracy": report_dict["accuracy"],
                    "test_macro_f1": report_dict["macro avg"]["f1-score"],
                    # Fatigued Class (The most important)
                    "test_fatigued_precision": report_dict["Fatigued"]["precision"],
                    "test_fatigued_recall": report_dict["Fatigued"]["recall"],
                    "test_fatigued_f1": report_dict["Fatigued"]["f1-score"],
                    # Awake Class (False Alarm monitoring)
                    "test_awake_precision": report_dict["Awake"]["precision"],
                    "test_awake_recall": report_dict["Awake"]["recall"],
                    "test_awake_f1": report_dict["Awake"]["f1-score"],
                }
            )

        return locals()
