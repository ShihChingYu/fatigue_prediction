"""Parse, merge, and convert config objects."""

# %% IMPORTS

import os
import typing as T
from pathlib import Path

import omegaconf as oc

# %% TYPES

# Defines a union type for Config objects
Config = T.Union[oc.ListConfig, oc.DictConfig]

# %% PARSERS


def parse_file(path: T.Union[str, Path]) -> Config:
    """Parse a config file from a path.

    Args:
        path (str | Path): Path to local config file (yaml).

    Returns:
        Config: Representation of the config file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    return oc.OmegaConf.load(path)


def parse_string(string: str) -> Config:
    """Parse the given config string (useful for CLI overrides).

    Args:
        string (str): Content of config string (e.g., "training.epochs=10").

    Returns:
        Config: Representation of the config string.
    """
    return oc.OmegaConf.create(string)


# %% MERGERS


def merge_configs(configs: T.Sequence[Config]) -> Config:
    """Merge a list of config objects into a single config.

    Later configs in the list overwrite earlier ones.

    Args:
        configs (T.Sequence[Config]): List of config objects.

    Returns:
        Config: Representation of the merged config objects.
    """
    return oc.OmegaConf.merge(*configs)


# %% CONVERTERS


def to_object(config: Config, resolve: bool = True) -> T.Any:
    """Convert a config object to a native Python object (dict or list).

    Args:
        config (Config): Representation of the config.
        resolve (bool): Whether to resolve variable interpolations (e.g., ${paths.data}).
                        Defaults to True.

    Returns:
        object: Conversion of the config to a python object (dict/list).
    """
    return oc.OmegaConf.to_container(config, resolve=resolve)


def to_yaml(config: Config, resolve: bool = True) -> str:
    """Convert a config object to a YAML string.

    Useful for logging the exact configuration used for a job.

    Args:
        config (Config): The config object.
        resolve (bool): Whether to resolve variable interpolations.

    Returns:
        str: The YAML string representation.
    """
    return oc.OmegaConf.to_yaml(config, resolve=resolve)


# %% ORCHESTRATION


def load_config(
    path: T.Union[str, Path], overrides: T.Union[T.Sequence[str], None] = None
) -> Config:
    """High-level utility to load a config file and apply CLI overrides.

    This is the main function your jobs will call.

    Args:
        path (str | Path): Path to the base config file.
        overrides (T.Sequence[str] | None): List of dot-notation overrides
                                            (e.g. ["training.lr=0.01"]).

    Returns:
        Config: The final merged configuration.
    """
    # 1. Load base file
    base_config = parse_file(path)

    # 2. Parse and merge overrides if they exist
    if overrides:
        override_configs = [parse_string(s) for s in overrides]
        return merge_configs([base_config, *override_configs])

    return base_config
