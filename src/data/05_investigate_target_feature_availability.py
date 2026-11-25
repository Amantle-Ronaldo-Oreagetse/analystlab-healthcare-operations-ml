from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

ENCOUNTERS_PATH = RAW_DIR / "encounters.csv"
PATIENTS_PATH = RAW_DIR / "patients.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "target_investigation"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "eda"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET_WINDOW_DAYS = 7


# ============================================================
# LOAD ENCOUNTERS
# ============================================================

print("Loading encounters dataset...")

encounters = pd.read_csv(
    ENCOUNTERS_PATH
)

print(f"Encounter dataset shape: {encounters.shape}")


# ============================================================
# PREPARE ENCOUNTER DATES
# ============================================================

encounters["START"] = pd.to_datetime(
    encounters["START"],
    errors="coerce"
)

encounters["STOP"] = pd.to_datetime(
    encounters["STOP"],
    errors="coerce"
)

encounters = encounters.dropna(
    subset=["PATIENT", "START", "STOP"]
).copy()

encounters = encounters.sort_values(
    ["PATIENT", "START", "STOP"]
).reset_index(drop=True)


# ============================================================
# DEFINE DATASET OBSERVATION END
# ============================================================

dataset_end_date = encounters["START"].max()

print(
    f"Dataset observation end: "
    f"{dataset_end_date}"
)


# ============================================================
# INVESTIGATION A:
# COMPARE TWO TARGET DEFINITIONS
# ============================================================

print("\n" + "=" * 65)
print("TARGET DEFINITION COMPARISON")
print("=" * 65)


# ------------------------------------------------------------
# TARGET A: START-TO-START
# ------------------------------------------------------------
#
# This replicates the previous investigation:
#
# Another encounter starts within 7 days after the current
# encounter starts.
#
# This is not necessarily the best operational definition
# if the prediction is intended to occur at encounter end.
# ------------------------------------------------------------

encounters["NEXT_START"] = (
    encounters.groupby("PATIENT")["START"]
    .shift(-1)
)

encounters["DAYS_TO_NEXT_START"] = (
    encounters["NEXT_START"]
    - encounters["START"]
).dt.total_seconds() / (60 * 60 * 24)


encounters["START_TO_START_ELIGIBLE"] = (
    encounters["START"]
    + pd.Timedelta(days=TARGET_WINDOW_DAYS)
    <= dataset_end_date
)


encounters["START_TO_START_TARGET"] = (
    encounters["DAYS_TO_NEXT_START"].notna()
    &
    (
        encounters["DAYS_TO_NEXT_START"] > 0
    )
    &
    (
        encounters["DAYS_TO_NEXT_START"]
        <= TARGET_WINDOW_DAYS
    )
).astype(int)


# ------------------------------------------------------------
# TARGET B: STOP-TO-NEXT-START
# ------------------------------------------------------------
#
# This is our proposed operational target:
#
# At the end of the current encounter, will another encounter
# START within the next 7 days?
#
# We search for the first encounter start strictly after
# the current encounter's STOP.
#
# This avoids counting another encounter that started
# before the current encounter had ended.
# ------------------------------------------------------------

# Find the next encounter that starts after the current encounter ends

encounters["NEXT_START_AFTER_STOP"] = pd.Series(
    pd.NaT,
    index=encounters.index,
    dtype="datetime64[ns, UTC]"
)

for patient_id, patient_group in encounters.groupby("PATIENT", sort=False):

    patient_indices = patient_group.index

    # Explicitly convert timestamps to nanosecond resolution
    patient_starts_ns = patient_group["START"].to_numpy(
        dtype="datetime64[ns]"
    )

    patient_stops_ns = patient_group["STOP"].to_numpy(
        dtype="datetime64[ns]"
    )

    # Find the first encounter starting after each encounter's STOP
    positions = np.searchsorted(
        patient_starts_ns,
        patient_stops_ns,
        side="right"
    )

    valid_positions = positions < len(patient_starts_ns)

    next_start_ns = np.full(
        len(patient_group),
        np.datetime64("NaT"),
        dtype="datetime64[ns]"
    )

    next_start_ns[valid_positions] = (
        patient_starts_ns[positions[valid_positions]]
    )

    next_start_timestamps = pd.to_datetime(
        next_start_ns,
        utc=True
    )

    encounters.loc[
        patient_indices,
        "NEXT_START_AFTER_STOP"
    ] = next_start_timestamps


encounters["DAYS_TO_NEXT_AFTER_STOP"] = (
    encounters["NEXT_START_AFTER_STOP"]
    - encounters["STOP"]
).dt.total_seconds() / (60 * 60 * 24)


# ============================================================
# DIAGNOSTIC: INVESTIGATE STOP-TO-NEXT-START INTERVALS
# ============================================================

print("\n" + "=" * 60)
print("DIAGNOSTICS: NEXT ENCOUNTER AFTER CURRENT ENCOUNTER ENDS")
print("=" * 60)

print(
    "Missing next-start timestamps:",
    encounters["NEXT_START_AFTER_STOP"].isna().sum()
)

print(
    "Missing days-to-next-start values:",
    encounters["DAYS_TO_NEXT_AFTER_STOP"].isna().sum()
)

print("\nDays to next encounter after STOP:")
print(
    encounters["DAYS_TO_NEXT_AFTER_STOP"].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# Count encounters followed by another encounter within each window
within_7_days = (
    (encounters["DAYS_TO_NEXT_AFTER_STOP"] > 0)
    & (encounters["DAYS_TO_NEXT_AFTER_STOP"] <= 7)
)

within_30_days = (
    (encounters["DAYS_TO_NEXT_AFTER_STOP"] > 0)
    & (encounters["DAYS_TO_NEXT_AFTER_STOP"] <= 30)
)

print("\nEncounters followed by another encounter:")
print("Within 7 days after STOP:", within_7_days.sum())
print("Within 30 days after STOP:", within_30_days.sum())

# Inspect examples where the original start-to-start target was positive
sample = encounters.loc[
    (encounters["DAYS_TO_NEXT_START"] > 0)
    & (encounters["DAYS_TO_NEXT_START"] <= 7),
    [
        "START",
        "STOP",
        "NEXT_START",
        "DAYS_TO_NEXT_START",
        "NEXT_START_AFTER_STOP",
        "DAYS_TO_NEXT_AFTER_STOP",
    ],
].head(15)

print("\nSample of encounters with a positive start-to-start 7-day target:")
print(sample.to_string(index=False))

# ------------------------------------------------------------
# FULL FOLLOW-UP ELIGIBILITY
# ------------------------------------------------------------
#
# An encounter is eligible only if the dataset contains
# the complete 7-day follow-up period after STOP.
#
# Otherwise, we cannot reliably label it as a negative case.
# ------------------------------------------------------------

encounters["STOP_TO_START_ELIGIBLE"] = (
    encounters["STOP"]
    + pd.Timedelta(days=TARGET_WINDOW_DAYS)
    <= dataset_end_date
)


encounters["STOP_TO_START_TARGET"] = (
    encounters["DAYS_TO_NEXT_AFTER_STOP"].notna()
    &
    (
        encounters["DAYS_TO_NEXT_AFTER_STOP"] > 0
    )
    &
    (
        encounters["DAYS_TO_NEXT_AFTER_STOP"]
        <= TARGET_WINDOW_DAYS
    )
).astype(int)


# ============================================================
# SUMMARIZE TARGET DEFINITIONS
# ============================================================

target_summary_rows = []


target_definitions = [
    {
        "target_name": "Start-to-start",
        "eligible_column": "START_TO_START_ELIGIBLE",
        "target_column": "START_TO_START_TARGET",
    },
    {
        "target_name": "Stop-to-next-start",
        "eligible_column": "STOP_TO_START_ELIGIBLE",
        "target_column": "STOP_TO_START_TARGET",
    },
]


for definition in target_definitions:

    eligible_mask = encounters[
        definition["eligible_column"]
    ]

    eligible_data = encounters.loc[
        eligible_mask
    ]

    positive_cases = int(
        eligible_data[
            definition["target_column"]
        ].sum()
    )

    total_eligible = len(eligible_data)

    negative_cases = (
        total_eligible - positive_cases
    )

    positive_rate = (
        positive_cases / total_eligible * 100
        if total_eligible > 0
        else 0
    )

    target_summary_rows.append(
        {
            "target_definition": (
                definition["target_name"]
            ),
            "eligible_encounters": (
                total_eligible
            ),
            "excluded_encounters": (
                len(encounters)
                - total_eligible
            ),
            "positive_cases": (
                positive_cases
            ),
            "negative_cases": (
                negative_cases
            ),
            "positive_rate_percent": (
                positive_rate
            ),
        }
    )


target_summary = pd.DataFrame(
    target_summary_rows
)

target_summary_path = (
    REPORT_DIR
    / "repeat_7day_target_definition_comparison.csv"
)

target_summary.to_csv(
    target_summary_path,
    index=False
)


print("\nTarget definition comparison:")

print(
    target_summary.to_string(
        index=False
    )
)


# ============================================================
# SUMMARIZE PROPOSED TARGET BY ENCOUNTER CLASS
# ============================================================

print("\n" + "=" * 65)
print("PROPOSED TARGET BY ENCOUNTER CLASS")
print("=" * 65)


eligible_encounters = encounters.loc[
    encounters["STOP_TO_START_ELIGIBLE"]
].copy()


class_summary = (
    eligible_encounters
    .groupby("ENCOUNTERCLASS")
    ["STOP_TO_START_TARGET"]
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


class_summary_path = (
    REPORT_DIR
    / "repeat_7day_target_by_class_at_stop.csv"
)

class_summary.to_csv(
    class_summary_path,
    index=False
)


print(
    class_summary[
        [
            "ENCOUNTERCLASS",
            "encounter_count",
            "repeat_encounters",
            "repeat_percentage",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# INVESTIGATION B:
# BUILD CANDIDATE FEATURES AVAILABLE BY ENCOUNTER END
# ============================================================

print("\n" + "=" * 65)
print("CANDIDATE FEATURE AVAILABILITY")
print("=" * 65)


# ------------------------------------------------------------
# PATIENT DEMOGRAPHICS
# ------------------------------------------------------------

print("\nLoading patient demographics...")

patients = pd.read_csv(
    PATIENTS_PATH,
    usecols=[
        "Id",
        "BIRTHDATE",
        "GENDER",
        "RACE",
        "ETHNICITY",
    ]
)


patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"],
    errors="coerce",
    utc=True
)


# Merge only the selected demographic columns.
# Names, addresses, SSNs, and other direct identifiers
# are not loaded into this investigation.

encounters = encounters.merge(
    patients,
    left_on="PATIENT",
    right_on="Id",
    how="left",
    validate="many_to_one"
)


# ------------------------------------------------------------
# AGE AT INDEX ENCOUNTER
# ------------------------------------------------------------

encounters["AGE_AT_ENCOUNTER"] = (
    (
        encounters["START"]
        - encounters["BIRTHDATE"]
    ).dt.days
    / 365.25
)


# Remove impossible negative ages from the candidate feature.
encounters.loc[
    encounters["AGE_AT_ENCOUNTER"] < 0,
    "AGE_AT_ENCOUNTER"
] = np.nan


# ------------------------------------------------------------
# HISTORICAL FEATURES
# ------------------------------------------------------------
#
# These features use only encounters that started before
# the current index encounter START.
#
# We do not use future encounters to construct history.
# ------------------------------------------------------------

print("\nConstructing patient history features...")

encounters["PRIOR_ENCOUNTER_COUNT"] = (
    encounters.groupby("PATIENT")
    .cumcount()
)


encounters["PREVIOUS_ENCOUNTER_START"] = (
    encounters.groupby("PATIENT")["START"]
    .shift(1)
)


encounters["DAYS_SINCE_PREVIOUS_ENCOUNTER"] = (
    encounters["START"]
    - encounters["PREVIOUS_ENCOUNTER_START"]
).dt.total_seconds() / (60 * 60 * 24)


# Count encounters in the 30 days before the index encounter.
# The current encounter is excluded.

prior_30_day_counts = []
prior_90_day_counts = []

for patient_id, patient_group in encounters.groupby(
    "PATIENT",
    sort=False
):

    patient_starts = (
        patient_group["START"]
        .to_numpy()
    )

    counts_30 = []
    counts_90 = []

    for index_start in patient_starts:

        cutoff_30 = (
            index_start
            - np.timedelta64(30, "D")
        )

        cutoff_90 = (
            index_start
            - np.timedelta64(90, "D")
        )

        # Count starts in [cutoff, index_start).
        # The current encounter is excluded.

        left_30 = np.searchsorted(
            patient_starts,
            cutoff_30,
            side="left"
        )

        right_30 = np.searchsorted(
            patient_starts,
            index_start,
            side="left"
        )

        left_90 = np.searchsorted(
            patient_starts,
            cutoff_90,
            side="left"
        )

        right_90 = np.searchsorted(
            patient_starts,
            index_start,
            side="left"
        )

        counts_30.append(
            right_30 - left_30
        )

        counts_90.append(
            right_90 - left_90
        )

    prior_30_day_counts.extend(
        counts_30
    )

    prior_90_day_counts.extend(
        counts_90
    )


encounters["PRIOR_ENCOUNTERS_30_DAYS"] = (
    prior_30_day_counts
)

encounters["PRIOR_ENCOUNTERS_90_DAYS"] = (
    prior_90_day_counts
)


# ------------------------------------------------------------
# ENCOUNTER DURATION
# ------------------------------------------------------------
#
# Duration is known at encounter end.
# However, previous investigation found extreme durations.
# We will flag it as a candidate requiring data-quality review.
# ------------------------------------------------------------

encounters["DURATION_HOURS"] = (
    encounters["STOP"]
    - encounters["START"]
).dt.total_seconds() / 3600


# ============================================================
# FEATURE AVAILABILITY AUDIT
# ============================================================

feature_audit_rows = [
    {
        "feature": "ENCOUNTERCLASS",
        "source": "encounters.csv",
        "availability": "Candidate",
        "reason": (
            "Encounter type is likely known during "
            "the encounter. Confirm operational timing."
        ),
    },
    {
        "feature": "START calendar features",
        "source": "encounters.csv",
        "availability": "Candidate",
        "reason": (
            "Encounter start date and time are known "
            "at encounter end."
        ),
    },
    {
        "feature": "AGE_AT_ENCOUNTER",
        "source": "patients.csv",
        "availability": "Candidate",
        "reason": (
            "Age can be calculated at index encounter "
            "date. Do not use current age."
        ),
    },
    {
        "feature": "GENDER",
        "source": "patients.csv",
        "availability": "Candidate with review",
        "reason": (
            "Potential demographic feature. Confirm "
            "recorded value was available at prediction time."
        ),
    },
    {
        "feature": "RACE",
        "source": "patients.csv",
        "availability": "Candidate with review",
        "reason": (
            "Potential demographic feature. Assess "
            "responsible use and subgroup performance."
        ),
    },
    {
        "feature": "ETHNICITY",
        "source": "patients.csv",
        "availability": "Candidate with review",
        "reason": (
            "Potential demographic feature. Assess "
            "responsible use and subgroup performance."
        ),
    },
    {
        "feature": "PRIOR_ENCOUNTER_COUNT",
        "source": "encounters.csv",
        "availability": "Candidate",
        "reason": (
            "Calculated using only encounters before "
            "the index encounter."
        ),
    },
    {
        "feature": "DAYS_SINCE_PREVIOUS_ENCOUNTER",
        "source": "encounters.csv",
        "availability": "Candidate",
        "reason": (
            "Calculated from a previous encounter only."
        ),
    },
    {
        "feature": "PRIOR_ENCOUNTERS_30_DAYS",
        "source": "encounters.csv",
        "availability": "Candidate",
        "reason": (
            "Counts prior encounters in the 30 days "
            "before the index encounter."
        ),
    },
    {
        "feature": "PRIOR_ENCOUNTERS_90_DAYS",
        "source": "encounters.csv",
        "availability": "Candidate",
        "reason": (
            "Counts prior encounters in the 90 days "
            "before the index encounter."
        ),
    },
    {
        "feature": "DURATION_HOURS",
        "source": "encounters.csv",
        "availability": "Candidate with quality review",
        "reason": (
            "Known at encounter end, but extreme "
            "duration values require investigation."
        ),
    },
    {
        "feature": "ORGANIZATION",
        "source": "encounters.csv",
        "availability": "Conditional",
        "reason": (
            "Likely known during registration, but "
            "confirm the value is available by prediction time."
        ),
    },
    {
        "feature": "PAYER",
        "source": "encounters.csv",
        "availability": "Conditional",
        "reason": (
            "May be known during registration. "
            "Confirm availability and assess generalization."
        ),
    },
    {
        "feature": "CODE",
        "source": "encounters.csv",
        "availability": "Conditional",
        "reason": (
            "Encounter codes may be assigned or finalized "
            "after the encounter."
        ),
    },
    {
        "feature": "DESCRIPTION",
        "source": "encounters.csv",
        "availability": "Conditional",
        "reason": (
            "May be derived from encounter coding. "
            "Confirm it is available by encounter end."
        ),
    },
    {
        "feature": "BASE_ENCOUNTER_COST",
        "source": "encounters.csv",
        "availability": "Conditional",
        "reason": (
            "Confirm whether this cost is known at "
            "the prediction point."
        ),
    },
    {
        "feature": "TOTAL_CLAIM_COST",
        "source": "encounters.csv",
        "availability": "Exclude initially",
        "reason": (
            "Claim cost may be finalized after encounter "
            "end and may contain post-encounter information."
        ),
    },
    {
        "feature": "PAYER_COVERAGE",
        "source": "encounters.csv",
        "availability": "Exclude initially",
        "reason": (
            "May be determined or updated after the "
            "encounter. Availability is uncertain."
        ),
    },
    {
        "feature": "REASONCODE",
        "source": "encounters.csv",
        "availability": "Exclude initially",
        "reason": (
            "May be assigned after the encounter and "
            "could reveal outcome-related information."
        ),
    },
    {
        "feature": "REASONDESCRIPTION",
        "source": "encounters.csv",
        "availability": "Exclude initially",
        "reason": (
            "May be assigned after the encounter and "
            "could reveal outcome-related information."
        ),
    },
    {
        "feature": "PROCEDURES",
        "source": "procedures.csv",
        "availability": "Exclude initially",
        "reason": (
            "Procedure timestamps do not establish when "
            "the information was entered or available."
        ),
    },
    {
        "feature": "DEATHDATE",
        "source": "patients.csv",
        "availability": "Exclude",
        "reason": (
            "Could contain future information relative "
            "to the index encounter."
        ),
    },
    {
        "feature": "PATIENT identifier",
        "source": "encounters.csv",
        "availability": "Exclude from predictors",
        "reason": (
            "Identifier is used for grouping and history "
            "construction, not as a model feature."
        ),
    },
    {
        "feature": "NEXT_START_AFTER_STOP",
        "source": "encounters.csv",
        "availability": "Target construction only",
        "reason": (
            "Future encounter timing is used only to "
            "construct the target, never as a feature."
        ),
    },
]


feature_audit = pd.DataFrame(
    feature_audit_rows
)


feature_audit_path = (
    REPORT_DIR
    / "repeat_7day_feature_availability_audit.csv"
)

feature_audit.to_csv(
    feature_audit_path,
    index=False
)


print("\nFeature availability audit:")

print(
    feature_audit[
        [
            "feature",
            "availability",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FEATURE MISSINGNESS REPORT
# ============================================================

print("\n" + "=" * 65)
print("CANDIDATE FEATURE MISSINGNESS")
print("=" * 65)


candidate_features = [
    "ENCOUNTERCLASS",
    "ORGANIZATION",
    "PAYER",
    "CODE",
    "DESCRIPTION",
    "BASE_ENCOUNTER_COST",
    "AGE_AT_ENCOUNTER",
    "GENDER",
    "RACE",
    "ETHNICITY",
    "PRIOR_ENCOUNTER_COUNT",
    "DAYS_SINCE_PREVIOUS_ENCOUNTER",
    "PRIOR_ENCOUNTERS_30_DAYS",
    "PRIOR_ENCOUNTERS_90_DAYS",
    "DURATION_HOURS",
]


missingness_rows = []

for feature in candidate_features:

    missing_count = int(
        encounters[feature].isna().sum()
    )

    total_count = len(encounters)

    missing_percentage = (
        missing_count / total_count * 100
    )

    missingness_rows.append(
        {
            "feature": feature,
            "total_encounters": total_count,
            "missing_count": missing_count,
            "missing_percentage": (
                missing_percentage
            ),
        }
    )


missingness_df = pd.DataFrame(
    missingness_rows
)


missingness_path = (
    REPORT_DIR
    / "repeat_7day_candidate_feature_missingness.csv"
)

missingness_df.to_csv(
    missingness_path,
    index=False
)


print(
    missingness_df.to_string(
        index=False
    )
)


# ============================================================
# CREATE TARGET COMPARISON FIGURE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    target_summary["target_definition"],
    target_summary["positive_rate_percent"]
)

plt.title(
    "7-Day Repeat Encounter Target Definition Comparison"
)

plt.xlabel(
    "Target Definition"
)

plt.ylabel(
    "Positive Rate (%)"
)

plt.tight_layout()


figure_path = (
    FIGURE_DIR
    / "repeat_7day_target_definition_comparison.png"
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

print("\n" + "=" * 65)
print("TARGET AND FEATURE AVAILABILITY INVESTIGATION COMPLETED")
print("=" * 65)

print("\nReports saved to:")
print(REPORT_DIR)

print("\nGenerated reports:")
print(
    " - repeat_7day_target_definition_comparison.csv"
)
print(
    " - repeat_7day_target_by_class_at_stop.csv"
)
print(
    " - repeat_7day_feature_availability_audit.csv"
)
print(
    " - repeat_7day_candidate_feature_missingness.csv"
)

print("\nFigure saved to:")
print(figure_path)

print("\nNo original dataset was modified.")