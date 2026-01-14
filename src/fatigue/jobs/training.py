"""Define a job for training and registering a single AI/ML model."""

# %% IMPORTS

import typing as T
from typing import cast

import mlflow
import pandera as pa
import pydantic as pdt

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
    n_estimators: int = 82
    max_depth: int = 12
    min_samples_split: int = 17
    min_samples_leaf: int = 4
    bootstrap: bool = True
    max_features: T.Union[str, int, float] = "sqrt"

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
            inputs_train = schemas.ModelInputsSchema.check(self.inputs_train.read())
            targets_train = schemas.TargetsSchema.check(self.targets_train.read())
            logger.debug(f"Train Shape: {inputs_train.shape}")

            # Cast inputs to satisfy Pandera strict typing
            inputs_train_raw = schemas.ModelInputsSchema.check(self.inputs_train.read())
            inputs_train = cast(
                pa.typing.DataFrame[schemas.ModelInputsSchema], inputs_train.head(5)
            )

            targets_train_raw = schemas.TargetsSchema.check(self.targets_train.read())
            targets_train = cast(pa.typing.DataFrame[schemas.TargetsSchema], targets_train_raw)

            logger.debug(f"Train Shape: {inputs_train.shape}")

            # Log Lineage
            mlflow.log_input(
                self.inputs_train.lineage(name="inputs_train", data=inputs_train),
                context="training",
            )

            # 4. Train (Regression)
            logger.info("Fitting Random Forest Regressor...")

            self.model.n_estimators = self.n_estimators
            self.model.max_depth = self.max_depth
            self.model.min_samples_split = self.min_samples_split
            self.model.min_samples_leaf = self.min_samples_leaf
            self.model.bootstrap = self.bootstrap
            self.model.max_features = self.max_features

            logger.info(
                f"Hyperparameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}"
            )
            self.model.fit(inputs=inputs_train, targets=targets_train)

            # 1. Create a variable for the sample and CAST it immediately
            sample_inputs = cast(
                pa.typing.DataFrame[schemas.ModelInputsSchema], inputs_train.head(5)
            )

            # 2. Use that variable for prediction
            sample_output_raw = self.model.predict(inputs=sample_inputs)

            # 3. Cast the output too
            sample_output = cast(pa.typing.DataFrame[schemas.OutputsSchema], sample_output_raw)

            # 4. Use the CASTED variables for signing and saving
            signature = self.signer.sign(
                inputs=T.cast(T.Any, sample_inputs), outputs=T.cast(T.Any, sample_output)
            )

            model_info = self.saver.save(
                model=self.model,
                signature=signature,
                input_example=sample_inputs,
            )
            model_version = self.registry.register(
                name=self.mlflow_service.registry_name, model_uri=model_info.model_uri
            )

            logger.success(f"Model Registered: {model_version.name} v{model_version.version}")

            self.alerts_service.notify(
                title="Training Complete",
                message=f"Model {model_version.name} v{model_version.version} registered.",
            )

        return locals()
