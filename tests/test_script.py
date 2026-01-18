"""Integration tests for the CLI entry point (scripts.py)."""

# %% IMPORTS

import json
from pathlib import Path

import pytest
from _pytest import capture as pc

# Change 'fatigue' to your actual package name if different
from fatigue import scripts

# %% FIXTURES


@pytest.fixture
def confs_path() -> Path:
    """Return the path to the test configurations."""
    # This assumes you put your test-specific YAMLs in tests/confs/
    return Path("tests/confs")


# %% TESTS: CLI UTILITIES


def test_schema(capsys: pc.CaptureFixture[str]) -> None:
    """Test that 'python scripts.py --schema' dumps a valid JSON."""
    args = ["prog", "--schema"]

    try:
        scripts.main(args)
    except SystemExit:
        pass  # Argparse often exits after help/schema, which is fine.

    captured = capsys.readouterr()
    assert captured.err == "", "Captured error should be empty!"
    assert json.loads(captured.out), "Output should be valid JSON!"


def test_main__no_configs() -> None:
    """Test that running without arguments raises an error."""
    argv: list[str] = []

    # Depending on how your scripts.py is written, it might raise RuntimeError or SystemExit
    with pytest.raises((RuntimeError, ValueError, SystemExit)):
        scripts.main(argv)


# %% TESTS: JOB EXECUTION (The "Mega Test")


@pytest.mark.parametrize(
    "config_file",
    [
        # LIST ALL YOUR JOBS HERE
        "tuning.yaml",
        # "training.yaml",
        # "evaluation.yaml",
        # "inference.yaml",
        # "monitoring.yaml",
    ],
)
def test_integration_run(confs_path: Path, config_file: str) -> None:
    """
    Run a full job using the configuration file.

    This simulates: 'python src/fatigue/scripts.py tests/confs/{config_file}'
    """

    # 1. Setup Path
    config_path = confs_path / config_file

    # 2. Prepare Arguments
    argv = [str(config_path)]

    print(f"\n--- Testing Job: {config_file} ---")

    # 3. Run the Job
    try:
        # scripts.main() usually returns the 'locals()' dict of the job's run method
        result = scripts.main(argv)
    except Exception as e:
        pytest.fail(f"Job {config_file} failed with error: {e}")

    # 4. Basic Assertions
    assert isinstance(result, dict), "Job should return a dictionary of results (locals)"

    assert any(key in result for key in ["run", "run_config", "run_id"]), (
        f"Job {config_file} finished but returned incomplete results."
    )
