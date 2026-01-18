"""Define a job for finding the best hyperparameters using Optuna."""

# %% IMPORTS

import typing as T

import mlflow
import mlflow.data
import numpy as np
import optuna
import pandera as pa
import pydantic as pdt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error, recall_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Internal imports
from fatigue.core import schemas
from fatigue.io import datasets, services
from fatigue.jobs import base

# %% JOBS


class TuningJob(base.Job):
    """Find the best hyperparameters for the Fatigue Model using Optuna.

    Optimization Goal: Minimize RMSE.
    Monitoring Goal: Track Recall at specific threshold.
    """

    KIND: T.Literal["TuningJob"] = "TuningJob"

    # Run Config
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(name="Tuning")

    # Data
    inputs: datasets.ReaderKind = pdt.Field(
        default_factory=lambda: datasets.ParquetReader(path="data/processed/inputs_train.parquet"),
        discriminator="KIND",
    )
    targets: datasets.ReaderKind = pdt.Field(
        default_factory=lambda: datasets.ParquetReader(path="data/processed/targets_train.parquet"),
        discriminator="KIND",
    )

    # Tuning Configuration
    n_trials: int = 50  # How many combinations to try
    n_splits: int = 5  # 5-Fold Cross Validation (Requested)
    fatigue_threshold: float = 0.26  # Threshold to monitor Recall

    def run(self) -> base.Locals:
        # 1. Setup Services
        logger = self.logger_service.logger()
        client = self.mlflow_service.client()

        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info(f"Starting Tuning Run: {run.info.run_id}")

            # 2. Read Data
            inputs = schemas.ModelInputsSchema.check(self.inputs.read())
            targets = schemas.TargetsSchema.check(self.targets.read())
            y = targets["fatigue_score"]

            inputs_sorted = inputs.sort_values(by=["user_id"]).reset_index(drop=True)
            inputs = T.cast(pa.typing.DataFrame[schemas.ModelInputsSchema], inputs_sorted)

            # --- DATA LINEAGE ---
            logger.info("Hashing dataset for Lineage...")

            lineage_df = inputs.assign(fatigue_score=y)
            logger.debug(f"Lineage Columns: {lineage_df.columns.tolist()}")

            # NOTE: If this crashes with "Out of Memory", change inputs to inputs.head(100)
            dataset = mlflow.data.from_pandas(  # type: ignore
                inputs,
                source=self.inputs.path,
                name="tuning_data",
            )
            mlflow.log_input(dataset, context="tuning")
            logger.info("Data Lineage established.")

            # --- MODEL LINEAGE (ALGORITHM TAGGING) ---
            mlflow.set_tag("model_family", "RandomForest")
            mlflow.set_tag("library", "scikit-learn")
            mlflow.set_tag("task", "regression")
            # -----------------------------------------------

            if "user_id" in inputs.columns:
                groups = inputs["user_id"]
                X = inputs.drop(columns=["user_id"])
            else:
                logger.warning("No user_id found. Standard CV used.")
                groups = None
                X = inputs

            # 3. Define Objective
            def objective(trial: optuna.Trial) -> float:
                # A. Define Search Space (Based on your training.yaml + range)
                params = {
                    "model__n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "model__max_depth": trial.suggest_int("max_depth", 5, 25),
                    "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "model__max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", 1.0]
                    ),
                    "model__bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                }

                # B. Initialize Pipeline (Impute -> Scale -> Model)
                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),  # Fill NaNs
                        ("scaler", StandardScaler()),  # Scale features
                        ("model", RandomForestRegressor(random_state=42)),  # Predict
                    ]
                )
                # Apply the params from Optuna to the pipeline
                pipeline.set_params(**params)

                # 2. Define Custom Scorers
                def rmse_func(y_t, y_p):
                    return np.sqrt(mean_squared_error(y_t, y_p))

                def custom_recall(y_t, y_p):
                    return recall_score(
                        (y_t >= self.fatigue_threshold).astype(int),
                        (y_p >= self.fatigue_threshold).astype(int),
                        zero_division=0,
                    )

                scorers = {
                    "rmse": make_scorer(rmse_func, greater_is_better=False),
                    "recall": make_scorer(custom_recall),
                }

                # D. Cross Validation (GroupKFold)
                cv = GroupKFold(n_splits=self.n_splits)

                # cross_validate gives us both metrics
                scores = cross_validate(
                    pipeline, X, y, groups=groups, cv=cv, scoring=scorers, n_jobs=-1
                )

                # E. Aggregate Scores
                avg_rmse = -np.mean(scores["test_rmse"])
                avg_recall = np.mean(scores["test_recall"])

                # F. Log this trial's details to MLflow (as nested runs or metrics)
                trial.set_user_attr("recall", avg_recall)

                return avg_rmse

            # 4. Run Optimization
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=self.n_trials)

            # 5. Log Best Results
            best_params = study.best_params
            best_rmse = study.best_value
            best_recall = study.best_trial.user_attrs["recall"]

            logger.success(f"Tuning Complete. Best RMSE: {best_rmse:.4f}")
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_rmse", best_rmse)
            mlflow.log_metric("best_cv_recall", best_recall)

            # 5. Log Optuna Visualization Artifacts
            try:
                # Importance Plot: Which parameter actually moved the needle?
                fig_imp = optuna.visualization.plot_param_importances(study)
                fig_imp.write_image("param_importances.png")
                mlflow.log_artifact("param_importances.png")

                # Optimization History: Did it converge?
                fig_hist = optuna.visualization.plot_optimization_history(study)
                fig_hist.write_image("optimization_history.png")
                mlflow.log_artifact("optimization_history.png")

                logger.info("Optuna visualization plots logged to MLflow.")
            except Exception as e:
                logger.warning(f"Could not log Optuna plots: {e}. (Is 'kaleido' installed?)")

            self.alerts_service.notify(
                title="Tuning Finished",
                message=f"Best RMSE: {best_rmse:.4f} (Recall: {best_recall:.4f})",
            )

        return locals()
