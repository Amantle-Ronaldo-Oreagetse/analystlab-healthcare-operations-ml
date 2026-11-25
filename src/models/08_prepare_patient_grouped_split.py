from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "repeat_encounter_7day_modeling_dataset.csv"
)

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"

REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"

TRAIN_PATH = TRAIN_DIR / "train_dataset.csv"
TEST_PATH = TEST_DIR / "test_dataset.csv"

SPLIT_SUMMARY_PATH = REPORT_DIR / "patient_grouped_split_summary.csv"
CLASS_DISTRIBUTION_PATH = REPORT_DIR / "split_class_distribution.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "REPEAT_WITHIN_7_DAYS"

GROUP_COLUMN = "PATIENT"

IDENTIFIER_COLUMNS = [
    "Id",
    "PATIENT",
]

RANDOM_STATE = 42
NUMBER_OF_SPLITS = 5

# We use the first fold:
# approximately 80% training and 20% testing.


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading modelling dataset...")

data = pd.read_csv(MODELING_DATA_PATH)

print(f"Dataset shape: {data.shape}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = IDENTIFIER_COLUMNS + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


if data[GROUP_COLUMN].isna().any():
    raise ValueError(
        "Patient identifiers contain missing values."
    )


if data[TARGET].isna().any():
    raise ValueError(
        "Target contains missing values."
    )


if not set(data[TARGET].unique()).issubset({0, 1}):
    raise ValueError(
        "Target must contain only 0 and 1."
    )


if data["Id"].duplicated().any():
    raise ValueError(
        "Duplicate encounter identifiers detected."
    )


# ============================================================
# SEPARATE FEATURES, TARGET, AND PATIENT GROUPS
# ============================================================

# The identifiers are not predictors.
# They are retained separately for grouping and traceability.

feature_columns = [
    column
    for column in data.columns
    if column not in IDENTIFIER_COLUMNS + [TARGET]
]

X = data[feature_columns]

y = data[TARGET].astype(int)

groups = data[GROUP_COLUMN]


print("\nFEATURES USED FOR SPLITTING")

for feature in feature_columns:
    print(f"- {feature}")

print(f"\nNumber of predictors: {len(feature_columns)}")
print(f"Number of unique patients: {groups.nunique()}")


# ============================================================
# PATIENT-GROUPED, STRATIFIED SPLIT
# ============================================================

print("\nCreating patient-grouped train/test split...")

splitter = StratifiedGroupKFold(
    n_splits=NUMBER_OF_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(
        X=X,
        y=y,
        groups=groups,
    )
)


# ============================================================
# CREATE TRAINING AND TESTING DATASETS
# ============================================================

train_data = data.iloc[train_indices].copy()

test_data = data.iloc[test_indices].copy()


# ============================================================
# VALIDATE PATIENT SEPARATION
# ============================================================

train_patients = set(train_data[GROUP_COLUMN])

test_patients = set(test_data[GROUP_COLUMN])

overlapping_patients = train_patients.intersection(
    test_patients
)

if overlapping_patients:
    raise ValueError(
        "Patient leakage detected: "
        f"{len(overlapping_patients)} patients appear "
        "in both training and testing sets."
    )


# ============================================================
# VALIDATE ENCOUNTER SEPARATION
# ============================================================

train_encounters = set(train_data["Id"])

test_encounters = set(test_data["Id"])

overlapping_encounters = train_encounters.intersection(
    test_encounters
)

if overlapping_encounters:
    raise ValueError(
        "Encounter leakage detected."
    )


# ============================================================
# SAVE TRAINING AND TESTING DATASETS
# ============================================================

train_data.to_csv(
    TRAIN_PATH,
    index=False,
)

test_data.to_csv(
    TEST_PATH,
    index=False,
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

total_encounters = len(data)

train_encounters_count = len(train_data)

test_encounters_count = len(test_data)

total_patients = groups.nunique()

train_patient_count = train_data[GROUP_COLUMN].nunique()

test_patient_count = test_data[GROUP_COLUMN].nunique()


split_summary = pd.DataFrame(
    {
        "dataset": [
            "Full dataset",
            "Training set",
            "Testing set",
        ],
        "encounter_count": [
            total_encounters,
            train_encounters_count,
            test_encounters_count,
        ],
        "encounter_percentage": [
            100.0,
            train_encounters_count / total_encounters * 100,
            test_encounters_count / total_encounters * 100,
        ],
        "patient_count": [
            total_patients,
            train_patient_count,
            test_patient_count,
        ],
        "positive_cases": [
            int(y.sum()),
            int(train_data[TARGET].sum()),
            int(test_data[TARGET].sum()),
        ],
        "positive_rate_percentage": [
            y.mean() * 100,
            train_data[TARGET].mean() * 100,
            test_data[TARGET].mean() * 100,
        ],
    }
)

split_summary.to_csv(
    SPLIT_SUMMARY_PATH,
    index=False,
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_distribution = (
    pd.concat(
        [
            train_data[TARGET].value_counts(normalize=True)
            .rename("training_proportion"),

            test_data[TARGET].value_counts(normalize=True)
            .rename("testing_proportion"),
        ],
        axis=1,
    )
    .fillna(0)
    .reset_index()
)

class_distribution.columns = [
    "target_value",
    "training_proportion",
    "testing_proportion",
]

class_distribution["training_percentage"] = (
    class_distribution["training_proportion"] * 100
)

class_distribution["testing_percentage"] = (
    class_distribution["testing_proportion"] * 100
)

class_distribution.to_csv(
    CLASS_DISTRIBUTION_PATH,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("PATIENT-GROUPED TRAIN/TEST SPLIT COMPLETED")
print("=" * 65)

print("\nSPLIT SUMMARY")

print(
    split_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    )
)

print("\nCLASS DISTRIBUTION")

print(
    class_distribution.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nLEAKAGE CHECKS")

print(
    f"Patients appearing in both sets: "
    f"{len(overlapping_patients)}"
)

print(
    f"Encounters appearing in both sets: "
    f"{len(overlapping_encounters)}"
)

print("\nSaved files:")

print(f"Training dataset: {TRAIN_PATH}")
print(f"Testing dataset:  {TEST_PATH}")

print(f"Split summary: {SPLIT_SUMMARY_PATH}")
print(f"Class distribution: {CLASS_DISTRIBUTION_PATH}")