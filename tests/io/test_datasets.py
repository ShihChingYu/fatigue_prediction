# %% IMPORTS
import os
import typing as T

import pandas as pd
import pytest

from fatigue.core import schemas
from fatigue.io import datasets

# %% FIXTURES


@pytest.fixture
def inputs_path(tmp_path, inputs_samples: pd.DataFrame) -> str:
    """Create a temporary parquet file for reading tests."""
    path = tmp_path / "inputs.parquet"
    inputs_samples.to_parquet(path)
    return str(path)


@pytest.fixture
def tmp_outputs_path(tmp_path) -> str:
    """Create a temporary path for writing tests."""
    return str(tmp_path / "outputs.parquet")


# %% READERS


@pytest.mark.parametrize("limit", [None, 10])
def test_parquet_reader(limit: T.Optional[int], inputs_path: str) -> None:
    # given
    reader = datasets.ParquetReader(path=inputs_path, limit=limit)

    # when
    data = reader.read()
    lineage = reader.lineage(name="inputs", data=data)

    # then
    # - data
    assert data.ndim == 2, "Data should be a dataframe!"

    if limit is not None:
        assert len(data) == limit, "Data should have the limit size!"

    # - lineage
    assert lineage.name == "inputs", "Lineage name should be inputs!"

    # Check if source.uri matches. Note: MLflow might prepend 'file://' depending on version
    assert str(lineage.source.uri).endswith("inputs.parquet"), (
        "Lineage source uri should point to the inputs path!"
    )

    assert lineage.schema is not None, "Lineage schema should not be None"

    # Verify schema columns match data columns
    assert set(lineage.schema.input_names()) == set(data.columns), (
        "Lineage schema names should be the data columns!"
    )

    assert lineage.profile["num_rows"] == len(data), (
        "Lineage profile should contain the data row count!"
    )


# %% WRITERS


def test_parquet_writer(targets_samples: schemas.Targets, tmp_outputs_path: str) -> None:
    # given
    writer = datasets.ParquetWriter(path=tmp_outputs_path)

    # when
    writer.write(data=targets_samples)

    # then
    assert os.path.exists(tmp_outputs_path), "Data should be written!"

    # Verify content
    written_data = pd.read_parquet(tmp_outputs_path)
    assert len(written_data) == len(targets_samples)
