"""Scripts for the CLI application."""

# ruff: noqa: E402

# %% WARNINGS

import warnings

# disable annoying mlflow warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

# %% IMPORTS

import argparse
import json
import sys
import typing as T
import yaml  # Assuming you use PyYAML

# Internal Imports
from fatigue import settings

# %% PARSERS

parser = argparse.ArgumentParser(description="Run a Fatigue Prediction Job.")
parser.add_argument("files", nargs="*", help="Config files for the job (local path only).")
parser.add_argument(
    "-e", "--extras", nargs="*", default=[], help="Config strings (JSON) for the job."
)
parser.add_argument("-s", "--schema", action="store_true", help="Print settings schema and exit.")

# %% SCRIPTS


def main(argv: T.Optional[T.List[str]] = None) -> int:
    """Main script for the application."""

    args = parser.parse_args(argv)

    # 1. Print Schema (Helper for debugging)
    if args.schema:
        schema = settings.MainSettings.model_json_schema()
        json.dump(schema, sys.stdout, indent=4)
        return 0

    # 2. Parse Config Files
    # We load every file provided in the command line
    loaded_configs = []
    for file_path in args.files:
        with open(file_path, "r") as f:
            # We assume your configs are YAML
            loaded_configs.append(yaml.safe_load(f))

    # 3. Parse Extras (Command line overrides)
    # Example: -e '{"job": {"n_estimators": 50}}'
    for string in args.extras:
        loaded_configs.append(json.loads(string))

    if not loaded_configs:
        raise RuntimeError("No configs provided. Please pass a .yaml file.")

    # 4. Merge Configs
    # We start with the first file and update it with subsequent files
    final_config = loaded_configs[0]
    for override in loaded_configs[1:]:
        _deep_merge(final_config, override)

    # 5. Validate & Run
    # Validate the dictionary against your Pydantic Schema
    setting = settings.MainSettings.model_validate(final_config)

    print(f"Starting Job: {setting.job.KIND}...")

    # Run the job inside its context manager (starts/stops MLflow)
    with setting.job as runner:
        runner.run()

    print("Job finished successfully.")
    return 0


# %% HELPERS


def _deep_merge(base: dict, update: dict) -> None:
    """Recursive merge for config dictionaries."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


if __name__ == "__main__":
    main()
