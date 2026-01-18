"""Define a job for training and registering a single AI/ML model."""

# %% IMPORTS

import typing as T
from typing import cast

import mlflow
import pandas as pd
import pandera as pa
import pydantic as pdt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Internal imports
from fatigue.core import models, schemas
from fatigue.io import datasets, registries, services
from fatigue.jobs import base
from fatigue.utils import signers

# %% JOBS


class TrainingJob(base.Job):
    """Train and register a single AI/ML model.

    Logic:
    1. Read Pre-Split Data (Train/Test Parquet files).
    2. Train a Regressor (RandomForest).
    3. Evaluate Regression Metrics (RMSE).
    4. Apply a Threshold to convert Scores -> Labels (Awake vs Fatigued).
    5. Register the Model in MLflow Model Registry.

    """

    KIND: T.Literal["TrainingJob"] = "TrainingJob"

    # Run Configuration
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(name="Training")

    # --- DATA READERS ---

    # 1. Training Data
    inputs_train: datasets.ReaderKind = pdt.Field(
        default_factory=lambda: datasets.ParquetReader(path="data/processed/inputs_train.parquet"),
        discriminator="KIND",
    )
    targets_train: datasets.ReaderKind = pdt.Field(
        default_factory=lambda: datasets.ParquetReader(path="data/processed/targets_train.parquet"),
        discriminator="KIND",
    )

    # --- Hyperparameters ---
    n_estimators: int = 105
    max_depth: int = 7
    min_samples_split: int = 18
    min_samples_leaf: int = 2
    bootstrap: bool = True
    max_features: T.Union[str, int, float] = "log2"

    # --- MODEL ---
    model: models.FatigueRandomForestModel = pdt.Field(
        default_factory=lambda: models.FatigueRandomForestModel(), discriminator="KIND"
    )

    # --- REGISTRY ---
    saver: registries.SaverKind = pdt.Field(registries.CustomSaver(), discriminator="KIND")
    signer: signers.SignerKind = pdt.Field(signers.InferSigner(), discriminator="KIND")
    registry: registries.RegisterKind = pdt.Field(registries.MlflowRegister(), discriminator="KIND")

    def run(self) -> base.Locals:
        # 1. Setup
        logger = self.logger_service.logger()
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info(f"Starting Training Run: {run.info.run_id}")

            # 2. Read Training Data
            logger.info("Reading Training Data...")

            inputs_train_raw = schemas.ModelInputsSchema.check(self.inputs_train.read())
            inputs_train = cast(pa.typing.DataFrame[schemas.ModelInputsSchema], inputs_train_raw)

            targets_train_raw = schemas.TargetsSchema.check(self.targets_train.read())
            targets_train = cast(pa.typing.DataFrame[schemas.TargetsSchema], targets_train_raw)

            # Drop user_id before training
            if "user_id" in inputs_train.columns:
                logger.info("Dropping 'user_id' for training...")
                inputs_train = inputs_train.drop(columns=["user_id"])

            logger.debug(f"Train Shape: {inputs_train.shape}")

            # Log Lineage
            mlflow.log_input(
                self.inputs_train.lineage(name="inputs_train", data=inputs_train),
                context="training",
            )

            # 3. AUTOLOGGING
            mlflow.sklearn.autolog(
                log_models=False, log_input_examples=False, log_model_signatures=False
            )
            logger.info("Autologging enabled (Metrics/Params: ON, Model Artifacts: OFF)")

            # 4. Construct Pipeline
            # [CRITICAL] This must match the TuningJob logic exactly!
            # If we don't use this pipeline, the model will crash on NaNs or behave differently.
            logger.info("Constructing Training Pipeline (Impute -> Scale -> RF)...")

            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),  # Handle missing data
                    ("scaler", StandardScaler()),  # Normalize features
                    (
                        "model",
                        RandomForestRegressor(  # The actual model
                            n_estimators=self.n_estimators,
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            bootstrap=self.bootstrap,
                            max_features=self.max_features,
                            random_state=42,
                        ),
                    ),
                ]
            )
            y_train = targets_train["fatigue_score"]
            # 4. Train (Regression)
            logger.info("Fitting Random Forest Regressor...")
            pipeline.fit(inputs_train, y_train)

            # Create a variable for the sample and CAST it immediately
            sample_inputs = inputs_train.head(5)
            sample_output = pipeline.predict(sample_inputs)

            # 3. Cast the output too
            sample_output_df = pd.DataFrame(sample_output, columns=["prediction"])

            # 4. Use the CASTED variables for signing and saving
            signature = self.signer.sign(
                inputs=T.cast(T.Any, sample_inputs), outputs=T.cast(T.Any, sample_output_df)
            )

            sample_inputs_typed = T.cast(
                pa.typing.DataFrame[schemas.ModelInputsSchema], sample_inputs
            )

            model_info = self.saver.save(
                model=pipeline,
                signature=signature,
                input_example=sample_inputs_typed,
            )
            if model_info is None:
                raise RuntimeError("Saver returned None for model info")

            model_version = self.registry.register(
                name=self.mlflow_service.registry_name, model_uri=model_info.model_uri
            )
            if model_version is None:
                raise RuntimeError("Registry returned None for model version")

            logger.success(f"Model Registered: {model_version.name} v{model_version.version}")

            self.alerts_service.notify(
                title="Training Complete",
                message=f"Model {model_version.name} v{model_version.version} registered.",
            )

        return locals()
