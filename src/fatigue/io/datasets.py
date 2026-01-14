"""Read/Write datasets from/to external sources/destinations."""

# %% IMPORTS

import abc
import typing as T

import mlflow.data.pandas_dataset as lineage
import pandas as pd
import pydantic as pdt
from typing_extensions import TypeAlias, override

# %% TYPINGS

Lineage: TypeAlias = lineage.PandasDataset

# %% READERS


class Reader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset reader.

    Use a reader to load a dataset in memory.
    e.g., to read file, database, cloud storage, ...

    Parameters:
        limit (int, optional): maximum number of rows to read. Defaults to None.
    """

    KIND: str

    limit: T.Optional[int] = None

    @abc.abstractmethod
    def read(self) -> pd.DataFrame:
        """Read a dataframe from a dataset.

        Returns:
            pd.DataFrame: dataframe representation.
        """

    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: T.Optional[str] = None,
        predictions: T.Optional[str] = None,
    ) -> Lineage:
        """Generate lineage information.

        Args:
            name (str): dataset name.
            data (pd.DataFrame): reader dataframe.
            targets (str | None): name of the target column.
            predictions (str | None): name of the prediction column.

        Returns:
            Lineage: lineage information.
        """


class CSVReader(Reader):
    """Read a dataframe from a CSV file (Useful for Raw Data).

    Parameters:
        path (str): local path to the dataset.
        sep (str): delimiter to use. Defaults to ",".
        header (int | str | None): Row number(s) to use as the column names. Defaults to 0.
    """

    KIND: T.Literal["CSVReader"] = "CSVReader"

    path: str
    sep: str = ","
    header: T.Union[int, str, None] = 0

    @override
    def read(self) -> pd.DataFrame:
        # Standard pandas CSV read
        data = pd.read_csv(self.path, sep=self.sep, header=self.header)
        if self.limit is not None:
            data = data.head(self.limit)
        return data

    @override
    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: T.Optional[str] = None,
        predictions: T.Optional[str] = None,
    ) -> Lineage:
        return lineage.from_pandas(
            df=data,
            name=name,
            source=self.path,
            targets=targets,
            predictions=predictions,
        )


class ParquetReader(Reader):
    """Read a dataframe from a parquet file (Useful for Processed Data).

    Parameters:
        path (str): local path to the dataset.
        backend (str): data type backend to use. Defaults to "pyarrow".
    """

    KIND: T.Literal["ParquetReader"] = "ParquetReader"

    path: str
    backend: T.Literal["pyarrow", "numpy_nullable"] = "pyarrow"

    @override
    def read(self) -> pd.DataFrame:
        # can't limit rows at read time easily with pandas read_parquet,
        # so we read then slice.
        data = pd.read_parquet(self.path, dtype_backend=self.backend)
        if self.limit is not None:
            data = data.head(self.limit)
        return data

    @override
    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: T.Optional[str] = None,
        predictions: T.Optional[str] = None,
    ) -> Lineage:
        return lineage.from_pandas(
            df=data,
            name=name,
            source=self.path,
            targets=targets,
            predictions=predictions,
        )


ReaderKind = T.Union[CSVReader, ParquetReader]

# %% WRITERS


class Writer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset writer.

    Use a writer to save a dataset from memory.
    e.g., to write file, database, cloud storage, ...
    """

    KIND: str

    @abc.abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """Write a dataframe to a dataset.

        Args:
            data (pd.DataFrame): dataframe representation.
        """


class ParquetWriter(Writer):
    """Write a dataframe to a parquet file.

    Parameters:
        path (str): local or S3 path to the dataset.
    """

    KIND: T.Literal["ParquetWriter"] = "ParquetWriter"

    path: str

    @override
    def write(self, data: pd.DataFrame) -> None:
        # Saving processed files (Train/Test splits)
        # index=False is usually preferred unless the index contains vital info
        data.to_parquet(self.path, index=True)


WriterKind = ParquetWriter
