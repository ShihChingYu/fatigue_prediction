"""Manage global context during execution."""

# %% IMPORTS

from __future__ import annotations

import abc
import contextlib as ctx
import os
import sys
import typing as T

import loguru
import mlflow
import mlflow.tracking as mt
import pydantic as pdt
from plyer import notification

# %% SERVICES


class Service(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a global service.

    Use services to manage global contexts.
    e.g., logger object, mlflow client, spark context, ...
    """

    @abc.abstractmethod
    def start(self) -> None:
        """Start the service."""

    def stop(self) -> None:
        """Stop the service."""
        # does nothing by default


class LoggerService(Service):
    """Service for logging messages.

    https://loguru.readthedocs.io/en/stable/api/logger.html
    """

    sink: str = "stderr"
    level: str = "DEBUG"
    format: str = (
        "<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green>"
        "<level>[{level}]</level>"
        "<cyan>[{name}:{function}:{line}]</cyan>"
        " <level>{message}</level>"
    )
    colorize: bool = True
    serialize: bool = False
    backtrace: bool = True
    diagnose: bool = False
    catch: bool = True

    def start(self) -> None:
        loguru.logger.remove()
        config = self.model_dump()
        # use standard sinks or keep the original
        sinks = {"stderr": sys.stderr, "stdout": sys.stdout}

        # safely swap string "stderr" for actual sys.stderr object
        if config["sink"] in sinks:
            config["sink"] = sinks[config["sink"]]

        loguru.logger.add(**config)

    def logger(self) -> loguru.Logger:
        """Return the main logger."""
        return loguru.logger


class AlertsService(Service):
    """Service for sending notifications.

    In production, use with Slack, Discord, or emails.
    """

    enable: bool = True
    app_name: str = "Fatigue"
    timeout: T.Optional[int] = None

    def start(self) -> None:
        pass

    def notify(self, title: str, message: str) -> None:
        """Send a notification to the system."""
        if self.enable:
            try:
                notification.notify(
                    title=title,
                    message=message,
                    app_name=self.app_name,
                    timeout=self.timeout,
                )
            except (NotImplementedError, ImportError):
                # Fallback if plyer is not fully supported on the OS
                self._print(title=title, message=message)
        else:
            self._print(title=title, message=message)

    def _print(self, title: str, message: str) -> None:
        """Print a notification to the system."""
        print(f"[{self.app_name}] {title}: {message}")


class MlflowService(Service):
    """Service for Mlflow tracking and registry."""

    class RunConfig(pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
        """Run configuration for Mlflow tracking."""

        name: str
        description: T.Optional[str] = None
        tags: T.Optional[T.Dict[str, T.Any]] = None
        log_system_metrics: T.Optional[bool] = True

    # server uri
    tracking_uri: str = "./mlruns"
    registry_uri: str = "./mlruns"

    # ADAPTED: Project Name defaults
    experiment_name: str = "fatigue"
    registry_name: str = "fatigue"

    # autolog
    autolog_disable: bool = False
    autolog_disable_for_unsupported_versions: bool = False
    autolog_exclusive: bool = False
    autolog_log_input_examples: bool = True
    autolog_log_model_signatures: bool = True
    autolog_log_models: bool = False
    autolog_log_datasets: bool = False
    autolog_silent: bool = False

    def start(self) -> None:
        if mlflow.active_run():
            return

        final_tracking = os.getenv("MLFLOW_TRACKING_URI") or self.tracking_uri
        final_registry = os.getenv("MLFLOW_REGISTRY_URI") or self.registry_uri

        if "azureml" in final_tracking and final_registry == "./mlruns":
            final_registry = final_tracking

        # server uri
        mlflow.set_tracking_uri(uri=self.tracking_uri)
        mlflow.set_registry_uri(uri=self.registry_uri)
        # experiment
        mlflow.set_experiment(experiment_name=self.experiment_name)
        # autolog
        mlflow.autolog(
            disable=self.autolog_disable,
            disable_for_unsupported_versions=self.autolog_disable_for_unsupported_versions,
            exclusive=self.autolog_exclusive,
            log_input_examples=self.autolog_log_input_examples,
            log_model_signatures=self.autolog_log_model_signatures,
            log_datasets=self.autolog_log_datasets,
            silent=self.autolog_silent,
        )

    @ctx.contextmanager
    def run_context(self, run_config: RunConfig) -> T.Generator[mlflow.ActiveRun, None, None]:
        """Yield an active Mlflow run and exit it afterwards."""
        with mlflow.start_run(
            run_name=run_config.name,
            tags=run_config.tags,
            description=run_config.description,
            log_system_metrics=run_config.log_system_metrics,
        ) as run:
            yield run

    def client(self) -> mt.MlflowClient:
        """Return a new Mlflow client."""
        final_tracking = os.getenv("MLFLOW_TRACKING_URI") or self.tracking_uri
        final_registry = os.getenv("MLFLOW_REGISTRY_URI") or self.registry_uri

        if "azureml" in final_tracking and final_registry == "./mlruns":
            final_registry = final_tracking

        return mt.MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
