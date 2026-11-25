from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "repeat_encounter_7day"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
)

METRICS_PATH = REPORT_DIR / "test_set_model_metrics.csv"
CONFUSION_PATH = REPORT_DIR / "test_set_confusion_matrices.csv"
CLASSIFICATION_PATH = REPORT_DIR / "test_set_classification_reports.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "REPEAT_WITHIN_7_DAYS"

GROUP_COLUMN = "PATIENT"

IDENTIFIER_COLUMNS = [
    "Id",
    "PATIENT",
]

MODEL_FILES = {
    "Dummy Classifier": "dummy_classifier.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "RBF SVM": "rbf_svm.joblib",
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATASETS
# ============================================================

print("Loading training and testing datasets...")

train_data = pd.read_csv(TRAIN_PATH)

test_data = pd.read_csv(TEST_PATH)

print(f"Training dataset: {train_data.shape}")
print(f"Testing dataset:  {test_data.shape}")


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = IDENTIFIER_COLUMNS + [TARGET]

for column in required_columns:

    if column not in test_data.columns:
        raise ValueError(
            f"Required column missing from test data: {column}"
        )

if test_data[TARGET].isna().any():
    raise ValueError("Missing target values in test data.")

if not set(test_data[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

if test_data[GROUP_COLUMN].isna().any():
    raise ValueError("Missing patient identifiers in test data.")


# Confirm that training and testing patients are separate.

train_patients = set(train_data[GROUP_COLUMN])

test_patients = set(test_data[GROUP_COLUMN])

overlapping_patients = train_patients.intersection(
    test_patients
)

if overlapping_patients:
    raise ValueError(
        "Patient leakage detected between training and testing data."
    )

print("\nPatient separation verified: no overlapping patients.")


# ============================================================
# PREPARE TEST FEATURES AND TARGET
# ============================================================

feature_columns = [
    column
    for column in train_data.columns
    if column not in IDENTIFIER_COLUMNS + [TARGET]
]

X_test = test_data[feature_columns]

y_test = test_data[TARGET].astype(int)

print(f"\nNumber of test encounters: {len(X_test)}")
print(f"Number of predictors: {len(feature_columns)}")
print(f"Test positive cases: {y_test.sum()}")
print(f"Test negative cases: {(y_test == 0).sum()}")
print(f"Test positive rate: {y_test.mean():.4f}")


# ============================================================
# EVALUATE MODELS
# ============================================================

metrics_results = []

confusion_results = []

classification_results = []

print("\n" + "=" * 70)
print("EVALUATING FITTED MODELS ON THE UNTOUCHED TEST SET")
print("=" * 70)


for model_name, model_filename in MODEL_FILES.items():

    print(f"\nEvaluating: {model_name}")

    model_path = MODEL_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # Load fitted pipeline.
    model = joblib.load(model_path)

    # Generate class predictions.
    y_pred = model.predict(X_test)

    # Obtain continuous scores for ranking metrics.
    if hasattr(model, "predict_proba"):

        y_scores = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):

        y_scores = model.decision_function(X_test)

    else:
        raise ValueError(
            f"{model_name} does not provide prediction scores."
        )

    # --------------------------------------------------------
    # CLASSIFICATION METRICS
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_scores,
    )

    average_precision = average_precision_score(
        y_test,
        y_scores,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    metrics_results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
        }
    )

    confusion_results.append(
        {
            "model": model_name,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        }
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No repeat encounter",
            "Repeat encounter",
        ],
        output_dict=True,
        zero_division=0,
    )

    for class_name, values in report.items():

        if isinstance(values, dict):

            classification_results.append(
                {
                    "model": model_name,
                    "class": class_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1-score"],
                    "support": values["support"],
                }
            )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(metrics_results)

metrics_df = metrics_df.sort_values(
    by="f1_score",
    ascending=False,
)

confusion_df = pd.DataFrame(confusion_results)

classification_df = pd.DataFrame(
    classification_results
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)

confusion_df.to_csv(
    CONFUSION_PATH,
    index=False,
)

classification_df.to_csv(
    CLASSIFICATION_PATH,
    index=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SET MODEL COMPARISON")
print("=" * 70)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

print(
    confusion_df.to_string(
        index=False,
    )
)


print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)

print("\nGenerated reports:")

print(METRICS_PATH)
print(CONFUSION_PATH)
print(CLASSIFICATION_PATH)