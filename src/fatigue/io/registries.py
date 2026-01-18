"""Savers, loaders, and registers for model registries."""

# %% IMPORTS

import abc
import typing as T

import mlflow
import pydantic as pdt
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PyFuncModel, PythonModel, PythonModelContext

from fatigue.core import models, schemas

# %% TYPES

Alias = mlflow.entities.model_registry.ModelVersion

# %% HELPERS


def uri_for_model_alias(name: str, alias: str) -> str:
    """Create a model URI from a model name and an alias."""
    return f"models:/{name}@{alias}"


def uri_for_model_version(name: str, version: int) -> str:
    """Create a model URI from a model name and a version."""
    return f"models:/{name}/{version}"


def uri_for_model_alias_or_version(name: str, alias_or_version: T.Union[str, int]) -> str:
    """Create a model URi from a model name and an alias or version."""
    # 1. Handle "latest" explicitly
    if str(alias_or_version).lower() == "latest":
        client = mlflow.MlflowClient()
        # Get the latest version (regardless of stage)
        latest_versions = client.get_latest_versions(name, stages=None)
        if not latest_versions:
            raise RuntimeError(f"No versions found for model '{name}'")
        # Sort by version number to ensure we get the absolute newest
        latest_version = sorted(latest_versions, key=lambda x: int(x.version))[-1].version
        return f"models:/{name}/{latest_version}"

    # 2. Handle specific version numbers (e.g., 7)
    if isinstance(alias_or_version, int) or str(alias_or_version).isdigit():
        return f"models:/{name}/{alias_or_version}"

    # 3. Handle actual aliases (e.g., "Champion", "Challenger")
    return f"models:/{name}@{alias_or_version}"


# %% SAVERS


class Saver(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for saving models in registry.

    Parameters:
        path (str): model path inside the Mlflow store.
    """

    KIND: str
    path: str = "model"

    @abc.abstractmethod
    def save(
        self,
        model: models.Model,
        signature: ModelSignature,
        input_example: schemas.Inputs,
    ) -> ModelInfo:
        """Save a model in the model registry."""


class CustomSaver(Saver):
    """Saver for project models using the Mlflow PyFunc module."""

    KIND: T.Literal["CustomSaver"] = "CustomSaver"

    class Adapter(PythonModel):  # type: ignore[misc]
        """Adapt a custom model to the Mlflow PyFunc flavor."""

        def __init__(self, model: models.Model):
            self.model = model

        def predict(
            self,
            context: PythonModelContext,
            model_input: schemas.Inputs,
            # ADAPTED (Py3.9): Use T.Union for optional dict
            params: T.Union[T.Dict[str, T.Any], None] = None,
        ) -> schemas.Outputs:
            """Generate predictions with a custom model."""
            return self.model.predict(model_input)

    def save(
        self,
        model: models.Model,
        signature: ModelSignature,
        input_example: schemas.Inputs,
    ) -> ModelInfo:
        adapter = CustomSaver.Adapter(model=model)
        return mlflow.pyfunc.log_model(
            python_model=adapter,
            signature=signature,
            artifact_path=self.path,
            input_example=input_example,
        )


class BuiltinSaver(Saver):
    """Saver for built-in models using an Mlflow flavor module."""

    KIND: T.Literal["BuiltinSaver"] = "BuiltinSaver"
    flavor: str

    def save(
        self,
        model: models.Model,
        signature: ModelSignature,
        input_example: schemas.Inputs,
    ) -> ModelInfo:
        builtin_model = model.get_internal_model()
        module = getattr(mlflow, self.flavor)
        return module.log_model(
            builtin_model,
            artifact_path=self.path,
            signature=signature,
            input_example=input_example,
        )


SaverKind = T.Union[CustomSaver, BuiltinSaver]


# %% LOADERS


class Loader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for loading models from registry."""

    KIND: str

    class Adapter(abc.ABC):
        """Adapt any model for the project inference."""

        @abc.abstractmethod
        def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
            """Generate predictions with the internal model."""

    @abc.abstractmethod
    def load(self, uri: str) -> "Loader.Adapter":
        """Load a model from the model registry."""


class CustomLoader(Loader):
    """Loader for custom models using the Mlflow PyFunc module."""

    KIND: T.Literal["CustomLoader"] = "CustomLoader"

    class Adapter(Loader.Adapter):
        def __init__(self, model: PyFuncModel) -> None:
            self.model = model

        def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
            outputs = self.model.predict(data=inputs)
            return T.cast(schemas.Outputs, outputs)

    def load(self, uri: str) -> "CustomLoader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = CustomLoader.Adapter(model=model)
        return adapter


class BuiltinLoader(Loader):
    """Loader for built-in models using the Mlflow PyFunc module."""

    KIND: T.Literal["BuiltinLoader"] = "BuiltinLoader"

    class Adapter(Loader.Adapter):
        def __init__(self, model: PyFuncModel) -> None:
            self.model = model

        def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
            # Assuming schemas.OutputsSchema exists in your project structure
            columns = list(schemas.OutputsSchema.to_schema().columns)  # type: ignore
            outputs = self.model.predict(data=inputs)
            return schemas.Outputs(outputs, columns=columns, index=inputs.index)

    def load(self, uri: str) -> "BuiltinLoader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = BuiltinLoader.Adapter(model=model)
        return adapter


LoaderKind = T.Union[CustomLoader, BuiltinLoader]


# %% REGISTERS


class Register(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for registering models to a location."""

    KIND: str
    tags: dict[str, T.Any] = {}

    @abc.abstractmethod
    def register(self, name: str, model_uri: str) -> ModelVersion:
        """Register a model given its name and URI."""


class MlflowRegister(Register):
    """Register for models in the Mlflow Model Registry."""

    KIND: T.Literal["MlflowRegister"] = "MlflowRegister"

    def register(self, name: str, model_uri: str) -> ModelVersion:
        return mlflow.register_model(name=name, model_uri=model_uri, tags=self.tags)


# ADAPTED (Py3.9): Use T.Union for consistency
RegisterKind = T.Union[MlflowRegister, MlflowRegister]
