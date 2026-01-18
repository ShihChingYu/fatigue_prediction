"""Define a job for promoting a registered model version with an alias."""

# %% IMPORTS

import typing as T

from fatigue.io import services
from fatigue.jobs import base

# %% JOBS


class PromotionJob(base.Job):
    """Define a job for promoting a registered model version with an alias.

    https://mlflow.org/docs/latest/model-registry.html#concepts

    Parameters:
        alias (str): the mlflow alias to transition the registered model version.
        version (int | None): the model version to transition (use None for latest).
    """

    KIND: T.Literal["PromotionJob"] = "PromotionJob"

    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Promotion"
    )

    stage: str = "Production"
    version: T.Optional[int] = None

    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)

        with self.mlflow_service.run_context(self.run_config) as run:
            # - mlflow
            client = self.mlflow_service.client()
            logger.info("With client: {}", client)
            name = self.mlflow_service.registry_name

            # version
            if self.version is None:  # use the latest model version
                logger.info("Searching for latest version of model '{}'...", name)
                version = client.search_model_versions(
                    f"name='{name}'", max_results=1, order_by=["version_number DESC"]
                )[0].version
            else:
                version = str(self.version)

            logger.info("From version: {}", version)

            # Transition to Production
            client.transition_model_version_stage(
                name=name, version=version, stage=self.stage, archive_existing_versions=True
            )

            # Log to Azure Audit Trail
            self.mlflow_service.client().log_param(run.info.run_id, "promoted_version", version)
            self.mlflow_service.client().log_param(run.info.run_id, "promoted_alias", self.stage)

            # notify
            self.alerts_service.notify(
                title="Promotion Job Finished",
                message=f"Model: {name}, Version: {version} -> {self.stage}",
            )

            logger.success(f"Successfully promoted version {version} to {self.stage}.")
        return locals()
