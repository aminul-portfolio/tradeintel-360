import csv
import os
import re
from zipfile import BadZipFile

import pandas as pd


class IngestionError(Exception):
    """Base exception for user-safe trading-file ingestion failures."""


class UnsupportedFileType(IngestionError):
    """Raised when the uploaded file extension is not supported."""


class EmptyDataset(IngestionError):
    """Raised when the uploaded file contains no usable rows."""


class MissingProfitColumn(IngestionError):
    """Raised when the uploaded file has no Profit column."""


class NoParsableProfit(IngestionError):
    """Raised when the Profit column contains no numeric values."""


class UnreadableFile(IngestionError):
    """Raised when a supported file cannot be read safely."""


_UNNAMED_EXPORT_COLUMN = re.compile(r"^Unnamed:\s*\d+$")

_DATE_COLUMNS = ["Open Time", "Open", "Date"]

_NUMERIC_COLUMNS = [
    "Size",
    "Profit",
    "Commission",
    "Commissions",
    "Swap",
    "Balance",
    "Pips",
    "Volume",
]

_CSV_ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "ISO-8859-1",
]


def _read_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV using a supported encoding and detected delimiter."""
    encoding = None
    sample = None

    for candidate in _CSV_ENCODINGS:
        try:
            with open(file_path, "r", encoding=candidate) as file_obj:
                sample = file_obj.read(2048)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise UnreadableFile(
                "Unable to read the uploaded CSV file."
            ) from exc

    if encoding is None or sample is None:
        raise UnreadableFile(
            "Unable to read the CSV file with supported encodings."
        )

    try:
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    try:
        return pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=delimiter,
            on_bad_lines="skip",
        )
    except pd.errors.EmptyDataError as exc:
        raise EmptyDataset(
            "The uploaded trading file contains no data."
        ) from exc
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        raise UnreadableFile(
            "Unable to read the uploaded CSV file."
        ) from exc


def _read_excel(file_path: str) -> pd.DataFrame:
    """Read the first worksheet from an XLSX trading file."""
    try:
        return pd.read_excel(
            file_path,
            sheet_name=0,
        )
    except (OSError, ValueError, ImportError, BadZipFile) as exc:
        raise UnreadableFile(
            "Unable to read the uploaded Excel file."
        ) from exc

def _normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, broker-neutral DataFrame normalisation."""
    df = df.copy()

    df.dropna(how="all", inplace=True)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    export_index_columns = [
        column
        for column in df.columns
        if _UNNAMED_EXPORT_COLUMN.fullmatch(column)
    ]
    if export_index_columns:
        df.drop(
            columns=export_index_columns,
            inplace=True,
        )

    if "Commissions" in df.columns and "Commission" not in df.columns:
        df.rename(
            columns={"Commissions": "Commission"},
            inplace=True,
        )

    for date_column in _DATE_COLUMNS:
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(
                df[date_column],
                errors="coerce",
            )
            break

    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the minimum deterministic trade-analysis contract."""
    if df.empty:
        raise EmptyDataset(
            "The uploaded trading file contains no usable data."
        )

    if "Profit" not in df.columns:
        raise MissingProfitColumn(
            "The uploaded trading file must contain a Profit column."
        )

    if df["Profit"].notna().sum() == 0:
        raise NoParsableProfit(
            "The Profit column does not contain any numeric values."
        )


def clean_ftmo_csv(file_path: str) -> pd.DataFrame:
    """Read, normalise, and validate a supported trading-history file."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".csv":
        df = _read_csv(file_path)
    elif extension == ".xlsx":
        df = _read_excel(file_path)
    else:
        raise UnsupportedFileType(
            "Unsupported file format. Please upload a CSV or XLSX file."
        )

    df = _normalise_dataframe(df)
    _validate_dataframe(df)

    return df
