"""Generate signatures for AI/ML models."""

# %% IMPORTS

import abc
import typing as T

import mlflow
import pydantic as pdt
from mlflow.models.signature import ModelSignature

# %% TYPES


# %% SIGNERS


class Signer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for generating model signatures.

    Allow switching between model signing strategies.
    e.g., automatic inference, manual model signature, ...

    https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
    """

    KIND: str

    @abc.abstractmethod
    def sign(self, inputs: T.Any, outputs: T.Any) -> ModelSignature:
        """Generate a model signature from its inputs/outputs.

        Args:
            inputs (schemas.Inputs): inputs data.
            outputs (schemas.Outputs): outputs data.

        Returns:
            Signature: signature of the model.
        """


class InferSigner(Signer):
    """Generate model signatures from inputs/outputs data."""

    KIND: T.Literal["InferSigner"] = "InferSigner"

    def sign(self, inputs: T.Any, outputs: T.Any) -> ModelSignature:
        return mlflow.models.infer_signature(model_input=inputs, model_output=outputs)


SignerKind = T.Union[InferSigner]
