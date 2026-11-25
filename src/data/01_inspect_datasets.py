from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REPORT_DIR = PROJECT_ROOT / "reports" / "data_inspection"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Columns that may function as identifiers.
IDENTIFIER_KEYWORDS = [
    "id",
    "uuid",
    "identifier"
]

# Columns that may contain dates or timestamps.
DATE_KEYWORDS = [
    "date",
    "time",
    "timestamp",
    "birth",
    "death",
    "start",
    "stop"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_identifier_column(column_name):
    """
    Identify columns that may contain record identifiers.
    """

    name = column_name.lower()

    return (
        name == "id"
        or name.endswith("_id")
        or name.endswith("uuid")
        or name.endswith("identifier")
    )


def is_date_column(column_name):
    """
    Identify columns that may contain dates or timestamps.
    """

    name = column_name.lower()

    return any(keyword in name for keyword in DATE_KEYWORDS)


def safe_filename(filename):
    """
    Convert a filename into a safe report filename.
    """

    return filename.replace(".csv", "").replace(" ", "_")


# ============================================================
# MAIN INSPECTION
# ============================================================

def inspect_dataset(file_path):

    print("\n" + "=" * 70)
    print(f"INSPECTING: {file_path.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    dataset_name = safe_filename(file_path.name)

    row_count = len(df)
    column_count = len(df.columns)

    print(f"Rows: {row_count:,}")
    print(f"Columns: {column_count:,}")

    # --------------------------------------------------------
    # 1. Dataset structure
    # --------------------------------------------------------

    structure = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "non_null_count": df.notna().sum().values,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().values * 100
        ).round(2)
    })

    structure.to_csv(
        REPORT_DIR / f"{dataset_name}_structure.csv",
        index=False
    )

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # 2. Missing values
    # --------------------------------------------------------

    missing_summary = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing_summary.columns = [
        "column",
        "missing_count"
    ]

    missing_summary["missing_percentage"] = (
        missing_summary["missing_count"]
        / row_count
        * 100
    ).round(2)

    missing_summary = missing_summary.sort_values(
        by="missing_count",
        ascending=False
    )

    missing_summary.to_csv(
        REPORT_DIR / f"{dataset_name}_missing_values.csv",
        index=False
    )

    print("\nColumns with missing values:")

    missing_columns = missing_summary[
        missing_summary["missing_count"] > 0
    ]

    if missing_columns.empty:
        print("  No missing values detected.")
    else:
        print(
            missing_columns[
                ["column", "missing_count", "missing_percentage"]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # 3. Duplicate records
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nExact duplicate rows:")
    print(f"  {duplicate_count:,}")

    duplicate_summary = pd.DataFrame({
        "dataset": [file_path.name],
        "row_count": [row_count],
        "exact_duplicate_rows": [duplicate_count]
    })

    duplicate_summary.to_csv(
        REPORT_DIR / f"{dataset_name}_duplicates.csv",
        index=False
    )

    # --------------------------------------------------------
    # 4. Identifier columns
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in df.columns
        if is_identifier_column(column)
    ]

    identifier_results = []

    print("\nPotential identifier columns:")

    if not identifier_columns:
        print("  No obvious identifier columns detected.")

    for column in identifier_columns:

        unique_count = df[column].nunique(
            dropna=True
        )

        missing_count = df[column].isna().sum()

        duplicate_values = (
            df[column].duplicated()
            & df[column].notna()
        ).sum()

        identifier_results.append({
            "column": column,
            "unique_values": unique_count,
            "missing_values": missing_count,
            "duplicate_values": duplicate_values,
            "unique_across_non_missing_rows": (
                unique_count
                == df[column].notna().sum()
            )
        })

        print(
            f"  {column}: "
            f"{unique_count:,} unique values, "
            f"{missing_count:,} missing"
        )

    if identifier_results:

        pd.DataFrame(
            identifier_results
        ).to_csv(
            REPORT_DIR / f"{dataset_name}_identifiers.csv",
            index=False
        )

    # --------------------------------------------------------
    # 5. Date and timestamp inspection
    # --------------------------------------------------------

    date_columns = [
        column
        for column in df.columns
        if is_date_column(column)
    ]

    date_results = []

    print("\nPotential date/time columns:")

    for column in date_columns:

        parsed_dates = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_dates = parsed_dates.dropna()

        invalid_count = (
            df[column].notna().sum()
            - len(valid_dates)
        )

        if len(valid_dates) > 0:

            minimum_date = valid_dates.min()
            maximum_date = valid_dates.max()

        else:

            minimum_date = None
            maximum_date = None

        date_results.append({
            "column": column,
            "valid_dates": len(valid_dates),
            "invalid_dates": invalid_count,
            "minimum_date": minimum_date,
            "maximum_date": maximum_date
        })

        print(
            f"  {column}: "
            f"{len(valid_dates):,} valid dates, "
            f"{invalid_count:,} invalid dates"
        )

        print(
            f"    Range: "
            f"{minimum_date} to {maximum_date}"
        )

    if date_results:

        pd.DataFrame(
            date_results
        ).to_csv(
            REPORT_DIR / f"{dataset_name}_dates.csv",
            index=False
        )

    # --------------------------------------------------------
    # 6. Numeric column summaries
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:

        numeric_summary = df[
            numeric_columns
        ].describe().T

        numeric_summary.to_csv(
            REPORT_DIR / f"{dataset_name}_numeric_summary.csv"
        )

        print("\nNumeric columns:")
        print(list(numeric_columns))

    # --------------------------------------------------------
    # 7. Categorical column summaries
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    categorical_results = []

    for column in categorical_columns:

        categorical_results.append({
            "column": column,
            "unique_values": df[column].nunique(
                dropna=True
            ),
            "missing_values": df[column].isna().sum()
        })

    if categorical_results:

        pd.DataFrame(
            categorical_results
        ).to_csv(
            REPORT_DIR / f"{dataset_name}_categorical_summary.csv",
            index=False
        )

    # --------------------------------------------------------
    # 8. Memory usage
    # --------------------------------------------------------

    memory_usage_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    print("\nMemory usage:")
    print(f"  {memory_usage_mb:.2f} MB")

    # --------------------------------------------------------
    # 9. Final dataset summary
    # --------------------------------------------------------

    final_summary = pd.DataFrame({
        "dataset": [file_path.name],
        "rows": [row_count],
        "columns": [column_count],
        "exact_duplicate_rows": [duplicate_count],
        "missing_cells": [df.isna().sum().sum()],
        "memory_usage_mb": [round(memory_usage_mb, 2)]
    })

    final_summary.to_csv(
        REPORT_DIR / f"{dataset_name}_summary.csv",
        index=False
    )

    print("\nInspection completed.")

    # Release memory
    del df


# ============================================================
# EXECUTION
# ============================================================

def main():

    csv_files = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    if not csv_files:

        print("No CSV files found in:")
        print(RAW_DATA_DIR)

        return

    print("=" * 70)
    print("HEALTHCARE DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDatasets found: {len(csv_files)}")

    for file_path in csv_files:

        inspect_dataset(file_path)

    print("\n" + "=" * 70)
    print("ALL DATASETS INSPECTED")
    print("=" * 70)

    print("\nReports saved to:")
    print(REPORT_DIR)


if __name__ == "__main__":
    main()
