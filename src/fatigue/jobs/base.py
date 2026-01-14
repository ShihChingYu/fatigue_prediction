"""Base for high-level project jobs."""

# %% IMPORTS

import abc
import types as TS
import typing as T

import pydantic as pdt

# ADAPTED: Import from your project structure
from fatigue.io import services

# %% TYPES

# Local job variables
Locals = T.Dict[str, T.Any]

# %% JOBS


class Job(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a job.

    Use a job to execute runs in context.
    e.g., to define common services like logger

    Parameters:
        logger_service (services.LoggerService): manage the logger system.
        alerts_service (services.AlertsService): manage the alerts system.
        mlflow_service (services.MlflowService): manage the mlflow system.
    """

    KIND: str

    logger_service: services.LoggerService = services.LoggerService()
    alerts_service: services.AlertsService = services.AlertsService()
    mlflow_service: services.MlflowService = services.MlflowService()

    def __enter__(self) -> "Job":
        """Enter the job context.

        Returns:
            Job: return the current object.
        """
        self.logger_service.start()
        logger = self.logger_service.logger()
        logger.debug("[START] Logger service: {}", self.logger_service)

        logger.debug("[START] Alerts service: {}", self.alerts_service)
        self.alerts_service.start()

        logger.debug("[START] Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.start()

        return self

    def __exit__(
        self,
        exc_type: T.Optional[T.Type[BaseException]],
        exc_value: T.Optional[BaseException],
        exc_traceback: T.Optional[TS.TracebackType],
    ) -> T.Literal[False]:
        """Exit the job context.

        Returns:
            T.Literal[False]: always propagate exceptions.
        """
        logger = self.logger_service.logger()

        logger.debug("[STOP] Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.stop()

        logger.debug("[STOP] Alerts service: {}", self.alerts_service)
        self.alerts_service.stop()

        logger.debug("[STOP] Logger service: {}", self.logger_service)
        self.logger_service.stop()

        return False  # re-raise any exceptions that occurred

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """
