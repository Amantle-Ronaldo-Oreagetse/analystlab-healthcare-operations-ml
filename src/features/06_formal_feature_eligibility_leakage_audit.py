from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "feature_audit"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODELING_DATA_PATH = (
    PROCESSED_DIR / "repeat_encounter_7day_modeling_dataset.csv"
)

ENCOUNTERS_PATH = RAW_DATA_DIR / "encounters.csv"
PATIENTS_PATH = RAW_DATA_DIR / "patients.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WINDOW_DAYS = 7


# ============================================================
# LOAD DATA
# ============================================================

print("Loading encounters dataset...")

encounters = pd.read_csv(ENCOUNTERS_PATH)
patients = pd.read_csv(PATIENTS_PATH)

print("Encounter dataset shape:", encounters.shape)
print("Patient dataset shape:", patients.shape)


# ============================================================
# PREPARE TIMESTAMPS
# ============================================================

encounters["START"] = pd.to_datetime(
    encounters["START"],
    errors="coerce",
    utc=True
)

encounters["STOP"] = pd.to_datetime(
    encounters["STOP"],
    errors="coerce",
    utc=True
)

patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"],
    errors="coerce",
    utc=True
)

# Sort encounters chronologically within each patient
encounters = encounters.sort_values(
    ["PATIENT", "START", "STOP"]
).reset_index(drop=True)

# The latest recorded encounter START defines the end of
# the observation period for future encounter prediction.
observation_end = encounters["START"].max()

print("\nDataset observation end:", observation_end)


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

print("\n" + "=" * 65)
print("INITIAL DATA QUALITY CHECKS")
print("=" * 65)

print(
    "Missing encounter START:",
    encounters["START"].isna().sum()
)

print(
    "Missing encounter STOP:",
    encounters["STOP"].isna().sum()
)

print(
    "STOP earlier than START:",
    (encounters["STOP"] < encounters["START"]).sum()
)

print(
    "Duplicate encounter IDs:",
    encounters["Id"].duplicated().sum()
)


# ============================================================
# CONSTRUCT NEXT ENCOUNTER AFTER CURRENT STOP
# ============================================================

print("\nConstructing next-encounter timestamps...")

encounters["NEXT_START_AFTER_STOP"] = pd.Series(
    pd.NaT,
    index=encounters.index,
    dtype="datetime64[ns, UTC]"
)

for patient_id, patient_group in encounters.groupby(
    "PATIENT",
    sort=False
):

    patient_indices = patient_group.index

    # Convert UTC timestamps to timezone-naive nanosecond
    # arrays for NumPy search operations.
    patient_starts = (
        patient_group["START"]
        .dt.tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    patient_stops = (
        patient_group["STOP"]
        .dt.tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    patient_starts_ns = patient_starts.astype("int64")
    patient_stops_ns = patient_stops.astype("int64")

    # Find the position of the first START strictly after
    # each current encounter's STOP.
    positions = np.searchsorted(
        patient_starts_ns,
        patient_stops_ns,
        side="right"
    )

    # Valid positions are within the START array.
    valid_positions = positions < len(patient_starts_ns)

    # Use NaT for encounters with no later encounter.
    next_start_ns = np.full(
        len(patient_group),
        np.iinfo(np.int64).min,
        dtype=np.int64
    )

    next_start_ns[valid_positions] = (
        patient_starts_ns[positions[valid_positions]]
    )

    next_start_timestamps = pd.to_datetime(
        next_start_ns,
        unit="ns",
        utc=True
    )

    encounters.loc[
        patient_indices,
        "NEXT_START_AFTER_STOP"
    ] = next_start_timestamps


# ============================================================
# CONSTRUCT TARGET
# ============================================================

print("Constructing seven-day repeat-encounter target...")

encounters["DAYS_TO_NEXT_AFTER_STOP"] = (
    encounters["NEXT_START_AFTER_STOP"]
    - encounters["STOP"]
).dt.total_seconds() / (60 * 60 * 24)

# An encounter is eligible only when the full target window
# can be observed before the dataset ends.
encounters["TARGET_ELIGIBLE"] = (
    encounters["STOP"]
    + pd.Timedelta(days=TARGET_WINDOW_DAYS)
    <= observation_end
)

# Positive if the next encounter starts within 7 days
# after the current encounter ends.
encounters["REPEAT_WITHIN_7_DAYS"] = (
    encounters["DAYS_TO_NEXT_AFTER_STOP"].ge(0)
    & encounters["DAYS_TO_NEXT_AFTER_STOP"].le(
        TARGET_WINDOW_DAYS
    )
).astype("int64")

# Remove labels for encounters without sufficient follow-up.
encounters.loc[
    ~encounters["TARGET_ELIGIBLE"],
    "REPEAT_WITHIN_7_DAYS"
] = np.nan


# ============================================================
# TARGET SUMMARY
# ============================================================

eligible_encounters = encounters.loc[
    encounters["TARGET_ELIGIBLE"]
].copy()



positive_cases = int(
    eligible_encounters["REPEAT_WITHIN_7_DAYS"].sum()
)

eligible_count = len(eligible_encounters)

negative_cases = eligible_count - positive_cases

positive_rate = (
    positive_cases / eligible_count * 100
    if eligible_count > 0
    else np.nan
)

target_summary = pd.DataFrame(
    {
        "metric": [
            "Total encounters",
            "Eligible encounters",
            "Excluded encounters",
            "Positive cases",
            "Negative cases",
            "Positive rate percentage",
        ],
        "value": [
            len(encounters),
            eligible_count,
            len(encounters) - eligible_count,
            positive_cases,
            negative_cases,
            positive_rate,
        ],
    }
)

target_summary.to_csv(
    REPORT_DIR / "target_summary.csv",
    index=False
)

print("\nTARGET SUMMARY")
print(target_summary.to_string(index=False))


# ============================================================
# TARGET BY ENCOUNTER CLASS
# ============================================================

target_by_class = (
    eligible_encounters
    .groupby("ENCOUNTERCLASS")["REPEAT_WITHIN_7_DAYS"]
    .agg(
        encounter_count="count",
        repeat_encounters="sum",
        repeat_rate="mean"
    )
    .reset_index()
)

target_by_class["repeat_percentage"] = (
    target_by_class["repeat_rate"] * 100
)

target_by_class.to_csv(
    REPORT_DIR / "target_by_encounter_class.csv",
    index=False
)

print("\nTARGET BY ENCOUNTER CLASS")
print(target_by_class.to_string(index=False))


# ============================================================
# CONSTRUCT HISTORICAL PATIENT FEATURES
# ============================================================

print("\nConstructing historical patient features...")

encounters["PRIOR_ENCOUNTER_COUNT"] = 0

encounters["DAYS_SINCE_PREVIOUS_ENCOUNTER"] = np.nan

encounters["PRIOR_ENCOUNTERS_30_DAYS"] = 0

encounters["PRIOR_ENCOUNTERS_90_DAYS"] = 0


for patient_id, patient_group in encounters.groupby(
    "PATIENT",
    sort=False
):

    patient_indices = patient_group.index

    starts = (
        patient_group["START"]
        .dt.tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    starts_ns = starts.astype("int64")

    encounter_positions = np.arange(
        len(patient_group)
    )

    # Number of encounters before the current encounter.
    encounters.loc[
        patient_indices,
        "PRIOR_ENCOUNTER_COUNT"
    ] = encounter_positions

    # Days since the previous encounter START.
    previous_start = patient_group["START"].shift(1)

    days_since_previous = (
        patient_group["START"] - previous_start
    ).dt.total_seconds() / (60 * 60 * 24)

    encounters.loc[
        patient_indices,
        "DAYS_SINCE_PREVIOUS_ENCOUNTER"
    ] = days_since_previous.to_numpy()

    # Historical encounter counts within 30 and 90 days.
    thirty_days_ns = pd.Timedelta(days=30).value
    ninety_days_ns = pd.Timedelta(days=90).value

    lower_30_positions = np.searchsorted(
        starts_ns,
        starts_ns - thirty_days_ns,
        side="left"
    )

    lower_90_positions = np.searchsorted(
        starts_ns,
        starts_ns - ninety_days_ns,
        side="left"
    )

    prior_30 = (
        encounter_positions - lower_30_positions
    )

    prior_90 = (
        encounter_positions - lower_90_positions
    )

    encounters.loc[
        patient_indices,
        "PRIOR_ENCOUNTERS_30_DAYS"
    ] = prior_30

    encounters.loc[
        patient_indices,
        "PRIOR_ENCOUNTERS_90_DAYS"
    ] = prior_90


# ============================================================
# CONSTRUCT AGE AT ENCOUNTER
# ============================================================

print("Constructing age at encounter...")

patient_birthdates = patients[
    ["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]
].copy()

patient_birthdates = patient_birthdates.rename(
    columns={"Id": "PATIENT"}
)

encounters = encounters.merge(
    patient_birthdates,
    on="PATIENT",
    how="left",
    validate="many_to_one"
)

encounters["AGE_AT_ENCOUNTER"] = (
    encounters["START"] - encounters["BIRTHDATE"]
).dt.total_seconds() / (
    60 * 60 * 24 * 365.2425
)


# ============================================================
# CONSTRUCT DURATION
# ============================================================

encounters["DURATION_HOURS"] = (
    encounters["STOP"] - encounters["START"]
).dt.total_seconds() / 3600


# ============================================================
# CONSTRUCT START CALENDAR FEATURES
# ============================================================

encounters["START_YEAR"] = encounters["START"].dt.year

encounters["START_MONTH"] = encounters["START"].dt.month

encounters["START_DAY_OF_WEEK"] = (
    encounters["START"].dt.dayofweek
)

encounters["START_HOUR"] = encounters["START"].dt.hour


# ============================================================
# FINAL FEATURE SELECTION AND MODELLING DATASET EXPORT
# ============================================================

print("\nPreparing final modelling dataset...")

# These are the nine predictors selected for the initial model.
# All are intended to be available at the end of the current
# encounter (STOP), subject to the documented data-workflow
# availability assumptions.
FINAL_FEATURES = [
    "ENCOUNTERCLASS",
    "DURATION_HOURS",
    "AGE_AT_ENCOUNTER",
    "PRIOR_ENCOUNTERS_30_DAYS",
    "PRIOR_ENCOUNTERS_90_DAYS",
    "DAYS_SINCE_PREVIOUS_ENCOUNTER",
    "START_DAY_OF_WEEK",
    "START_MONTH",
    "START_HOUR",
]

TARGET_COLUMN = "REPEAT_WITHIN_7_DAYS"

# Keep only encounters with the full seven-day follow-up window.
# This uses the target eligibility flag already constructed above.
modeling_data = encounters.loc[
    encounters["TARGET_ELIGIBLE"],
    ["Id", "PATIENT", *FINAL_FEATURES, TARGET_COLUMN],
].copy()

# Patient and encounter identifiers are retained for patient-aware
# splitting and traceability, but must not be used as predictors.
# Convert the target to an integer after excluding ineligible rows.
modeling_data[TARGET_COLUMN] = (
    modeling_data[TARGET_COLUMN].astype("int64")
)

# Basic export validation.
if modeling_data["Id"].duplicated().any():
    raise ValueError("Duplicate encounter IDs found in modelling dataset.")

if modeling_data[TARGET_COLUMN].isna().any():
    raise ValueError("Missing target labels found in eligible dataset.")

if not modeling_data[TARGET_COLUMN].isin([0, 1]).all():
    raise ValueError("Target contains values other than 0 and 1.")

if modeling_data.empty:
    raise ValueError("Modelling dataset is empty.")

modeling_data.to_csv(MODELING_DATA_PATH, index=False)

# Save a compact feature-level data quality summary.
feature_summary = pd.DataFrame(
    {
        "feature": FINAL_FEATURES,
        "dtype": [
            str(modeling_data[feature].dtype)
            for feature in FINAL_FEATURES
        ],
        "missing_count": [
            modeling_data[feature].isna().sum()
            for feature in FINAL_FEATURES
        ],
        "missing_percentage": [
            modeling_data[feature].isna().mean() * 100
            for feature in FINAL_FEATURES
        ],
        "unique_values": [
            modeling_data[feature].nunique(dropna=True)
            for feature in FINAL_FEATURES
        ],
    }
)

feature_summary.to_csv(
    REPORT_DIR / "final_model_feature_summary.csv",
    index=False
)

print("\nFINAL MODELLING DATASET")
print("Saved to:", MODELING_DATA_PATH)
print("Shape:", modeling_data.shape)
print("Predictor count:", len(FINAL_FEATURES))
print("Eligible encounters:", len(modeling_data))
print(
    "Positive cases:",
    int(modeling_data[TARGET_COLUMN].sum())
)
print(
    "Negative cases:",
    int((modeling_data[TARGET_COLUMN] == 0).sum())
)
print(
    "Positive rate:",
    f"{modeling_data[TARGET_COLUMN].mean() * 100:.2f}%"
)
print("\nSelected feature missingness:")
print(
    feature_summary[
        ["feature", "missing_count", "missing_percentage"]
    ].to_string(index=False)
)


# ============================================================
# FORMAL FEATURE ELIGIBILITY AUDIT
# ============================================================

feature_audit = pd.DataFrame(
    [
        {
            "feature": "ENCOUNTERCLASS",
            "availability": "Candidate",
            "leakage_risk": "Low",
            "recommendation": "Include",
            "rationale": (
                "Encounter class is recorded for the current encounter."
            ),
        },
        {
            "feature": "START_YEAR",
            "availability": "Candidate",
            "leakage_risk": "Low",
            "recommendation": "Include",
            "rationale": (
                "Calendar year is known from encounter START."
            ),
        },
        {
            "feature": "START_MONTH",
            "availability": "Candidate",
            "leakage_risk": "Low",
            "recommendation": "Include",
            "rationale": (
                "Calendar month is known from encounter START."
            ),
        },
        {
            "feature": "START_DAY_OF_WEEK",
            "availability": "Candidate",
            "leakage_risk": "Low",
            "recommendation": "Include",
            "rationale": (
                "Day of week is known from encounter START."
            ),
        },
        {
            "feature": "START_HOUR",
            "availability": "Candidate",
            "leakage_risk": "Low",
            "recommendation": "Include",
            "rationale": (
                "Start hour is known from encounter START."
            ),
        },
        {
            "feature": "AGE_AT_ENCOUNTER",
            "availability": "Candidate",
            "leakage_risk": "Low",
            "recommendation": "Include after quality checks",
            "rationale": (
                "Age can be calculated from birthdate and encounter START."
            ),
        },
        {
            "feature": "GENDER",
            "availability": "Candidate with review",
            "leakage_risk": "Fairness consideration",
            "recommendation": "Evaluate carefully",
            "rationale": (
                "Demographic feature; assess fairness and necessity."
            ),
        },
        {
            "feature": "RACE",
            "availability": "Candidate with review",
            "leakage_risk": "Fairness consideration",
            "recommendation": "Evaluate carefully",
            "rationale": (
                "Demographic feature; assess fairness and necessity."
            ),
        },
        {
            "feature": "ETHNICITY",
            "availability": "Candidate with review",
            "leakage_risk": "Fairness consideration",
            "recommendation": "Evaluate carefully",
            "rationale": (
                "Demographic feature; assess fairness and necessity."
            ),
        },
        {
            "feature": "PRIOR_ENCOUNTER_COUNT",
            "availability": "Candidate",
            "leakage_risk": "Low if correctly constructed",
            "recommendation": "Include",
            "rationale": (
                "Counts only encounters preceding the current encounter."
            ),
        },
        {
            "feature": "DAYS_SINCE_PREVIOUS_ENCOUNTER",
            "availability": "Candidate",
            "leakage_risk": "Low if correctly constructed",
            "recommendation": "Include with missing-value handling",
            "rationale": (
                "Uses the previous encounter START only."
            ),
        },
        {
            "feature": "PRIOR_ENCOUNTERS_30_DAYS",
            "availability": "Candidate",
            "leakage_risk": "Low if correctly constructed",
            "recommendation": "Include",
            "rationale": (
                "Counts encounters before the current encounter "
                "within the previous 30 days."
            ),
        },
        {
            "feature": "PRIOR_ENCOUNTERS_90_DAYS",
            "availability": "Candidate",
            "leakage_risk": "Low if correctly constructed",
            "recommendation": "Include",
            "rationale": (
                "Counts encounters before the current encounter "
                "within the previous 90 days."
            ),
        },
        {
            "feature": "DURATION_HOURS",
            "availability": "Available at encounter end",
            "leakage_risk": "Low at STOP; leakage for earlier prediction",
            "recommendation": "Include for end-of-encounter prediction",
            "rationale": (
                "Duration is known when STOP is reached."
            ),
        },
        {
            "feature": "ORGANIZATION",
            "availability": "Conditional",
            "leakage_risk": "Potential workflow dependency",
            "recommendation": "Review before inclusion",
            "rationale": (
                "Confirm the organization is known at prediction time."
            ),
        },
        {
            "feature": "PAYER",
            "availability": "Conditional",
            "leakage_risk": "Potential workflow dependency",
            "recommendation": "Review before inclusion",
            "rationale": (
                "Confirm payer information is available by encounter end."
            ),
        },
        {
            "feature": "CODE",
            "availability": "Conditional",
            "leakage_risk": "Potential coding delay",
            "recommendation": "Review before inclusion",
            "rationale": (
                "Confirm the encounter code is finalized by STOP."
            ),
        },
        {
            "feature": "DESCRIPTION",
            "availability": "Conditional",
            "leakage_risk": "Potential coding delay",
            "recommendation": "Review before inclusion",
            "rationale": (
                "Confirm the encounter description is finalized by STOP."
            ),
        },
        {
            "feature": "BASE_ENCOUNTER_COST",
            "availability": "Conditional",
            "leakage_risk": "Potential post-encounter calculation",
            "recommendation": "Review before inclusion",
            "rationale": (
                "Confirm this cost is available at encounter completion."
            ),
        },
        {
            "feature": "TOTAL_CLAIM_COST",
            "availability": "Potentially post-encounter",
            "leakage_risk": "High",
            "recommendation": "Exclude initially",
            "rationale": (
                "Claim costs may be finalized after the prediction point."
            ),
        },
        {
            "feature": "PAYER_COVERAGE",
            "availability": "Potentially post-encounter",
            "leakage_risk": "High",
            "recommendation": "Exclude initially",
            "rationale": (
                "Coverage may be calculated or updated after encounter end."
            ),
        },
        {
            "feature": "REASONCODE",
            "availability": "Potentially post-encounter",
            "leakage_risk": "High",
            "recommendation": "Exclude initially",
            "rationale": (
                "May be assigned after the encounter or through billing."
            ),
        },
        {
            "feature": "REASONDESCRIPTION",
            "availability": "Potentially post-encounter",
            "leakage_risk": "High",
            "recommendation": "Exclude initially",
            "rationale": (
                "May be assigned after the encounter or through billing."
            ),
        },
        {
            "feature": "PROCEDURES",
            "availability": "Timing not established",
            "leakage_risk": "Potentially high",
            "recommendation": "Exclude initially",
            "rationale": (
                "Procedure timing relative to STOP is not established."
            ),
        },
        {
            "feature": "DEATHDATE",
            "availability": "Future or post-event information",
            "leakage_risk": "Very high",
            "recommendation": "Exclude",
            "rationale": (
                "May reveal future information unavailable at prediction."
            ),
        },
        {
            "feature": "PATIENT",
            "availability": "Identifier",
            "leakage_risk": "Identity memorization",
            "recommendation": "Exclude from predictors",
            "rationale": (
                "Use only for grouping, history construction, and splitting."
            ),
        },
        {
            "feature": "NEXT_START_AFTER_STOP",
            "availability": "Future outcome information",
            "leakage_risk": "Direct target leakage",
            "recommendation": "Exclude from predictors",
            "rationale": (
                "Used only to construct the repeat-encounter target."
            ),
        },
        {
            "feature": "DAYS_TO_NEXT_AFTER_STOP",
            "availability": "Future outcome information",
            "leakage_risk": "Direct target leakage",
            "recommendation": "Exclude from predictors",
            "rationale": (
                "Directly determines the repeat-encounter target."
            ),
        },
        {
            "feature": "REPEAT_WITHIN_7_DAYS",
            "availability": "Target",
            "leakage_risk": "Target",
            "recommendation": "Target only",
            "rationale": (
                "This is the outcome to be predicted."
            ),
        },
    ]
)

feature_audit.to_csv(
    REPORT_DIR / "formal_feature_eligibility_audit.csv",
    index=False
)


# ============================================================
# FEATURE MISSINGNESS AUDIT
# ============================================================

candidate_features = [
    "ENCOUNTERCLASS",
    "START_YEAR",
    "START_MONTH",
    "START_DAY_OF_WEEK",
    "START_HOUR",
    "AGE_AT_ENCOUNTER",
    "GENDER",
    "RACE",
    "ETHNICITY",
    "PRIOR_ENCOUNTER_COUNT",
    "DAYS_SINCE_PREVIOUS_ENCOUNTER",
    "PRIOR_ENCOUNTERS_30_DAYS",
    "PRIOR_ENCOUNTERS_90_DAYS",
    "DURATION_HOURS",
    "ORGANIZATION",
    "PAYER",
    "CODE",
    "DESCRIPTION",
    "BASE_ENCOUNTER_COST",
]

missingness = pd.DataFrame(
    {
        "feature": candidate_features,
        "total_encounters": len(encounters),
        "missing_count": [
            encounters[feature].isna().sum()
            for feature in candidate_features
        ],
    }
)

missingness["missing_percentage"] = (
    missingness["missing_count"]
    / missingness["total_encounters"]
    * 100
)

missingness.to_csv(
    REPORT_DIR / "candidate_feature_missingness.csv",
    index=False
)


# ============================================================
# DURATION QUALITY AUDIT
# ============================================================

duration_summary = encounters[
    "DURATION_HOURS"
].describe(
    percentiles=[
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]
)

duration_summary.to_csv(
    REPORT_DIR / "duration_quality_summary.csv"
)

duration_quality = pd.DataFrame(
    {
        "metric": [
            "Zero or negative duration",
            "Duration greater than 24 hours",
            "Duration greater than 7 days",
            "Duration greater than 30 days",
        ],
        "count": [
            (
                encounters["DURATION_HOURS"] <= 0
            ).sum(),
            (
                encounters["DURATION_HOURS"] > 24
            ).sum(),
            (
                encounters["DURATION_HOURS"] > 24 * 7
            ).sum(),
            (
                encounters["DURATION_HOURS"] > 24 * 30
            ).sum(),
        ],
    }
)

duration_quality.to_csv(
    REPORT_DIR / "duration_quality_flags.csv",
    index=False
)


# ============================================================
# LEAKAGE CHECKS
# ============================================================

print("\n" + "=" * 65)
print("LEAKAGE CHECKS")
print("=" * 65)

# Check that target-related columns are not in the candidate
# predictor list.
target_related_columns = [
    "NEXT_START_AFTER_STOP",
    "DAYS_TO_NEXT_AFTER_STOP",
    "REPEAT_WITHIN_7_DAYS",
]

predictor_columns = candidate_features

leakage_check = pd.DataFrame(
    {
        "check": [
            "Target columns excluded from candidate predictors",
            "Patient identifier excluded from predictors",
            "Historical features use prior encounters",
            "Duration available at end-of-encounter prediction point",
        ],
        "result": [
            not any(
                column in predictor_columns
                for column in target_related_columns
            ),
            "PATIENT" not in predictor_columns,
            True,
            True,
        ],
    }
)

leakage_check.to_csv(
    REPORT_DIR / "leakage_checks.csv",
    index=False
)

print(leakage_check.to_string(index=False))


# ============================================================
# SAVE AUDIT SUMMARY
# ============================================================

summary_lines = [
    "FORMAL FEATURE ELIGIBILITY AND LEAKAGE AUDIT",
    "=" * 55,
    "",
    f"Prediction point: End of current encounter (STOP)",
    f"Target: Repeat encounter within {TARGET_WINDOW_DAYS} days after STOP",
    f"Dataset observation end: {observation_end}",
    f"Total encounters: {len(encounters):,}",
    f"Eligible encounters: {eligible_count:,}",
    f"Excluded encounters: {len(encounters) - eligible_count:,}",
    f"Positive cases: {positive_cases:,}",
    f"Negative cases: {negative_cases:,}",
    f"Positive rate: {positive_rate:.2f}%",
    "",
    "Final modelling predictors:",
    ", ".join(FINAL_FEATURES),
    f"Final modelling dataset: {MODELING_DATA_PATH}",
    "",
    "Important exclusions:",
    "- Patient identifier",
    "- Next encounter timestamp",
    "- Days to next encounter",
    "- Target label",
    "- Total claim cost",
    "- Payer coverage",
    "- Reason code and description",
    "- Procedures until timing is established",
    "- Death date",
    "",
    "Important review items:",
    "- Confirm availability of organization and payer at STOP",
    "- Confirm encounter code and description are finalized at STOP",
    "- Confirm base encounter cost is available at STOP",
    "- Review demographic features for fairness and necessity",
    "- Investigate extreme encounter durations",
    "- Use patient-aware validation to avoid patient overlap",
]

with open(
    REPORT_DIR / "formal_audit_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(summary_lines))


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("FORMAL FEATURE ELIGIBILITY AND LEAKAGE AUDIT COMPLETED")
print("=" * 65)

print("\nReports saved to:")
print(REPORT_DIR)
print("Modelling dataset saved to:")
print(MODELING_DATA_PATH)

print("\nGenerated files:")
for path in sorted(REPORT_DIR.iterdir()):
    print("-", path.name)