from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "encounters.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "target_investigation"
FIGURE_DIR = PROJECT_ROOT / "figures" / "eda"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

REPEAT_WINDOWS_DAYS = [7, 30, 90]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading encounters dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# PREPARE DATES
# ============================================================

df["START"] = pd.to_datetime(
    df["START"],
    errors="coerce"
)

df["STOP"] = pd.to_datetime(
    df["STOP"],
    errors="coerce"
)

df = df.dropna(
    subset=["PATIENT", "START"]
).copy()

df = df.sort_values(
    ["PATIENT", "START", "STOP"]
).reset_index(drop=True)


# ============================================================
# FIND THE NEXT ENCOUNTER FOR EACH PATIENT
# ============================================================

print("\nIdentifying each patient's next encounter...")

df["NEXT_ENCOUNTER_START"] = (
    df.groupby("PATIENT")["START"]
    .shift(-1)
)

df["DAYS_TO_NEXT_ENCOUNTER"] = (
    df["NEXT_ENCOUNTER_START"] - df["START"]
).dt.total_seconds() / (60 * 60 * 24)


# ============================================================
# DEFINE OBSERVATION PERIOD
# ============================================================

# The latest encounter date determines the end of the
# observation period available in this dataset.

dataset_end_date = df["START"].max()

print(f"Dataset start date: {df['START'].min()}")
print(f"Dataset end date:   {dataset_end_date}")


# ============================================================
# INVESTIGATE REPEAT ENCOUNTER TARGETS
# ============================================================

summary_rows = []

class_summary_rows = []

for window in REPEAT_WINDOWS_DAYS:

    print("\n" + "=" * 60)
    print(f"REPEAT ENCOUNTER WITHIN {window} DAYS")
    print("=" * 60)

    # An encounter is eligible only if the dataset contains
    # the full follow-up period after that encounter.
    #
    # This avoids incorrectly labeling encounters near the
    # end of the dataset as non-repeat encounters.

    df[f"ELIGIBLE_{window}_DAYS"] = (
        df["START"] + pd.Timedelta(days=window)
        <= dataset_end_date
    )

    eligible_mask = df[f"ELIGIBLE_{window}_DAYS"]

    eligible_df = df.loc[eligible_mask].copy()

    # A repeat encounter must occur after the index encounter
    # and within the specified number of days.

    eligible_df[f"REPEAT_WITHIN_{window}_DAYS"] = (
        eligible_df["DAYS_TO_NEXT_ENCOUNTER"].notna()
        &
        (eligible_df["DAYS_TO_NEXT_ENCOUNTER"] > 0)
        &
        (
            eligible_df["DAYS_TO_NEXT_ENCOUNTER"]
            <= window
        )
    ).astype(int)

    total_eligible = len(eligible_df)

    positive_cases = int(
        eligible_df[
            f"REPEAT_WITHIN_{window}_DAYS"
        ].sum()
    )

    negative_cases = total_eligible - positive_cases

    positive_percentage = (
        positive_cases / total_eligible * 100
        if total_eligible > 0
        else 0
    )

    negative_percentage = (
        negative_cases / total_eligible * 100
        if total_eligible > 0
        else 0
    )

    excluded_records = len(df) - total_eligible

    summary_rows.append(
        {
            "repeat_window_days": window,
            "total_encounters": len(df),
            "eligible_encounters": total_eligible,
            "excluded_due_to_incomplete_followup": (
                excluded_records
            ),
            "repeat_encounters": positive_cases,
            "non_repeat_encounters": negative_cases,
            "repeat_percentage": positive_percentage,
            "non_repeat_percentage": negative_percentage,
        }
    )

    print(f"Total encounters: {len(df):,}")
    print(f"Eligible encounters: {total_eligible:,}")
    print(f"Excluded due to incomplete follow-up: {excluded_records:,}")
    print(f"Repeat encounters: {positive_cases:,}")
    print(f"Non-repeat encounters: {negative_cases:,}")
    print(f"Repeat percentage: {positive_percentage:.2f}%")
    print(f"Non-repeat percentage: {negative_percentage:.2f}%")

    # --------------------------------------------------------
    # REPEAT ENCOUNTER RATE BY ENCOUNTER CLASS
    # --------------------------------------------------------

    class_summary = (
        eligible_df
        .groupby("ENCOUNTERCLASS")
        [f"REPEAT_WITHIN_{window}_DAYS"]
        .agg(
            encounter_count="count",
            repeat_encounters="sum",
            repeat_rate="mean"
        )
        .reset_index()
    )

    class_summary["repeat_percentage"] = (
        class_summary["repeat_rate"] * 100
    )

    class_summary["repeat_window_days"] = window

    class_summary_rows.append(class_summary)

    print("\nRepeat encounter rate by encounter class:")

    print(
        class_summary[
            [
                "ENCOUNTERCLASS",
                "encounter_count",
                "repeat_encounters",
                "repeat_percentage"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE OVERALL FEASIBILITY SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_path = (
    REPORT_DIR
    / "repeat_encounter_feasibility_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 60)
print("OVERALL REPEAT ENCOUNTER FEASIBILITY")
print("=" * 60)

print(summary_df.to_string(index=False))


# ============================================================
# SAVE CLASS-LEVEL SUMMARY
# ============================================================

class_summary_df = pd.concat(
    class_summary_rows,
    ignore_index=True
)

class_summary_path = (
    REPORT_DIR
    / "repeat_encounter_by_class.csv"
)

class_summary_df.to_csv(
    class_summary_path,
    index=False
)


# ============================================================
# INVESTIGATE TIME BETWEEN ENCOUNTERS
# ============================================================

print("\n" + "=" * 60)
print("TIME BETWEEN SUCCESSIVE ENCOUNTERS")
print("=" * 60)

repeat_intervals = df.loc[
    df["DAYS_TO_NEXT_ENCOUNTER"].notna()
    &
    (df["DAYS_TO_NEXT_ENCOUNTER"] > 0),
    "DAYS_TO_NEXT_ENCOUNTER"
]

if len(repeat_intervals) > 0:

    interval_statistics = repeat_intervals.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )

    print(interval_statistics)

    interval_summary_df = (
        interval_statistics
        .to_frame(name="days_between_encounters")
    )

    interval_summary_path = (
        REPORT_DIR
        / "time_between_encounters_summary.csv"
    )

    interval_summary_df.to_csv(
        interval_summary_path
    )

else:

    print("No successive encounters were found.")


# ============================================================
# CREATE REPEAT ENCOUNTER RATE FIGURE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    summary_df["repeat_window_days"].astype(str),
    summary_df["repeat_percentage"]
)

plt.title(
    "Repeat Encounter Rate by Follow-up Window"
)

plt.xlabel(
    "Follow-up Window (Days)"
)

plt.ylabel(
    "Repeat Encounter Rate (%)"
)

plt.tight_layout()

figure_path = (
    FIGURE_DIR
    / "repeat_encounter_rate_by_window.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("REPEAT ENCOUNTER INVESTIGATION COMPLETED")
print("=" * 60)

print("\nReports saved to:")
print(REPORT_DIR)

print("\nGenerated reports:")
print(" - repeat_encounter_feasibility_summary.csv")
print(" - repeat_encounter_by_class.csv")
print(" - time_between_encounters_summary.csv")

print("\nFigure saved to:")
print(figure_path)

print("\nNo original dataset was modified.")