from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "train_dataset.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "test_dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "repeat_encounter_7day"
)

CV_RESULTS_PATH = REPORT_DIR / "model_cross_validation_results.csv"

CV_SUMMARY_PATH = REPORT_DIR / "model_cross_validation_summary.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "REPEAT_WITHIN_7_DAYS"

GROUP_COLUMN = "PATIENT"

IDENTIFIER_COLUMNS = [
    "Id",
    "PATIENT",
]

CATEGORICAL_FEATURES = [
    "ENCOUNTERCLASS",
]

RANDOM_STATE = 42

NUMBER_OF_SPLITS = 5


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...")

train_data = pd.read_csv(TRAIN_PATH)

print(f"Training dataset shape: {train_data.shape}")


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = (
    IDENTIFIER_COLUMNS
    + [TARGET]
    + CATEGORICAL_FEATURES
)

missing_columns = [
    column
    for column in required_columns
    if column not in train_data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if train_data[TARGET].isna().any():
    raise ValueError(
        "Target contains missing values."
    )

if train_data[GROUP_COLUMN].isna().any():
    raise ValueError(
        "Patient identifiers contain missing values."
    )


# ============================================================
# SEPARATE FEATURES, TARGET, AND GROUPS
# ============================================================

feature_columns = [
    column
    for column in train_data.columns
    if column not in IDENTIFIER_COLUMNS + [TARGET]
]

X_train = train_data[feature_columns]

y_train = train_data[TARGET].astype(int)

groups = train_data[GROUP_COLUMN]


# ============================================================
# IDENTIFY NUMERICAL FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    column
    for column in feature_columns
    if column not in CATEGORICAL_FEATURES
]


print("\nFEATURES")

print(f"Categorical features: {CATEGORICAL_FEATURES}")

print(f"Numerical features: {NUMERICAL_FEATURES}")

print(f"Total predictors: {len(feature_columns)}")

print(f"Training encounters: {len(X_train)}")

print(f"Training patients: {groups.nunique()}")

print(f"Training positive rate: {y_train.mean():.4f}")


# ============================================================
# PREPROCESSING
# ============================================================

# Numerical features:
# 1. Replace missing values using the training median.
# 2. Standardize numerical values.

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# Categorical features:
# 1. Replace missing values using the most frequent category.
# 2. Convert categories into numerical indicator columns.

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# CREATE PATIENT-GROUPED CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=NUMBER_OF_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# DEFINE EVALUATION METRICS
# ============================================================

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# ============================================================
# CROSS-VALIDATION
# ============================================================

all_cv_results = []

summary_results = []

print("\n" + "=" * 70)
print("STARTING PATIENT-GROUPED CROSS-VALIDATION")
print("=" * 70)

for model_name, estimator in models.items():

    print(f"\nEvaluating: {model_name}")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    # --------------------------------------------------------
    # SAVE INDIVIDUAL FOLD RESULTS
    # --------------------------------------------------------

    for fold_number in range(NUMBER_OF_SPLITS):

        fold_result = {
            "model": model_name,
            "fold": fold_number + 1,
            "precision": cv_results[
                "test_precision"
            ][fold_number],
            "recall": cv_results[
                "test_recall"
            ][fold_number],
            "f1": cv_results[
                "test_f1"
            ][fold_number],
            "roc_auc": cv_results[
                "test_roc_auc"
            ][fold_number],
        }

        all_cv_results.append(
            fold_result
        )

    # --------------------------------------------------------
    # CALCULATE MEAN AND STANDARD DEVIATION
    # --------------------------------------------------------

    model_summary = {
        "model": model_name,
    }

    for metric in scoring:

        scores = cv_results[
            f"test_{metric}"
        ]

        model_summary[
            f"{metric}_mean"
        ] = scores.mean()

        model_summary[
            f"{metric}_std"
        ] = scores.std()

    summary_results.append(
        model_summary
    )

    print(
        f"Mean F1-score: "
        f"{model_summary['f1_mean']:.4f}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{model_summary['roc_auc_mean']:.4f}"
    )


# ============================================================
# SAVE CROSS-VALIDATION RESULTS
# ============================================================

cv_results_df = pd.DataFrame(
    all_cv_results
)

cv_results_df.to_csv(
    CV_RESULTS_PATH,
    index=False,
)


cv_summary_df = pd.DataFrame(
    summary_results
)

cv_summary_df = cv_summary_df.sort_values(
    by="f1_mean",
    ascending=False,
)

cv_summary_df.to_csv(
    CV_SUMMARY_PATH,
    index=False,
)


# ============================================================
# TRAIN FINAL VERSIONS ON THE FULL TRAINING SET
# ============================================================

print("\n" + "=" * 70)
print("FITTING FINAL MODEL VERSIONS ON THE TRAINING SET")
print("=" * 70)

for model_name, estimator in models.items():

    print(f"\nFitting: {model_name}")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".joblib"
    )

    model_path = MODEL_DIR / model_filename

    joblib.dump(
        pipeline,
        model_path,
    )

    print(f"Saved model: {model_path}")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING AND COMPARISON COMPLETED")
print("=" * 70)

print("\nCROSS-VALIDATION SUMMARY")

print(
    cv_summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved cross-validation results:")
print(CV_RESULTS_PATH)

print("\nSaved cross-validation summary:")
print(CV_SUMMARY_PATH)

print("\nSaved models:")
print(MODEL_DIR)