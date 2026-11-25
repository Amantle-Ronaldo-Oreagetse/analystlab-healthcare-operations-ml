from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "train_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "repeat_encounter_7day"
    / "random_forest.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "evaluation"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
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

TARGET = "REPEAT_WITHIN_7_DAYS"

GROUP_COLUMN = "PATIENT"

N_SPLITS = 5
RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50


# Thresholds from 0.05 to 0.95 in increments of 0.01.
THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2,
)


# ============================================================
# 2. LOAD TRAINING DATA AND MODEL CONFIGURATION
# ============================================================

print("Loading training dataset...")
train_data = pd.read_csv(TRAIN_DATA_PATH)

print("Loading saved Random Forest pipeline...")
saved_model = joblib.load(MODEL_PATH)

X_train = train_data[FEATURES]
y_train = train_data[TARGET].astype(int)
groups = train_data[GROUP_COLUMN]

print("\nTraining dataset shape:", train_data.shape)
print("Number of predictors:", len(FEATURES))
print("Number of patients:", groups.nunique())

print("\nTarget distribution:")
print(y_train.value_counts().sort_index())

print(
    "\nPositive rate:",
    round(y_train.mean(), 4),
)


# ============================================================
# 3. CREATE PATIENT-GROUPED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("GENERATING OUT-OF-FOLD PREDICTIONS")
print("=" * 65)

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# Each training encounter will receive a probability
# from a model that did not train on that encounter's patient.

oof_probabilities = np.full(
    len(train_data),
    np.nan,
)

oof_fold = np.full(
    len(train_data),
    -1,
)


for fold_number, (train_indices, validation_indices) in enumerate(
    cv.split(X_train, y_train, groups),
    start=1,
):

    print(f"\nProcessing fold {fold_number}...")

    fold_train_groups = set(
        groups.iloc[train_indices]
    )

    fold_validation_groups = set(
        groups.iloc[validation_indices]
    )

    overlapping_patients = (
        fold_train_groups
        & fold_validation_groups
    )

    if overlapping_patients:
        raise RuntimeError(
            f"Patient leakage detected in fold {fold_number}."
        )

    # Clone the saved pipeline configuration.
    # This creates a fresh, unfitted model with the same
    # preprocessing and Random Forest parameters.

    fold_model = clone(saved_model)

    fold_model.fit(
        X_train.iloc[train_indices],
        y_train.iloc[train_indices],
    )

    fold_probabilities = fold_model.predict_proba(
        X_train.iloc[validation_indices]
    )[:, 1]

    oof_probabilities[validation_indices] = (
        fold_probabilities
    )

    oof_fold[validation_indices] = fold_number

    print(
        f"Fold {fold_number} completed. "
        f"Validation encounters: {len(validation_indices)}"
    )


# ============================================================
# 4. VALIDATE OUT-OF-FOLD PREDICTIONS
# ============================================================

if np.isnan(oof_probabilities).any():
    raise RuntimeError(
        "Some training encounters did not receive "
        "out-of-fold predictions."
    )

if (oof_fold == -1).any():
    raise RuntimeError(
        "Some training encounters were not assigned "
        "to a validation fold."
    )

print("\nAll training encounters received OOF predictions.")

print(
    "OOF ROC-AUC:",
    round(
        roc_auc_score(
            y_train,
            oof_probabilities,
        ),
        4,
    ),
)

print(
    "OOF Average Precision:",
    round(
        average_precision_score(
            y_train,
            oof_probabilities,
        ),
        4,
    ),
)


# ============================================================
# 5. SAVE OOF PREDICTIONS
# ============================================================

oof_results = pd.DataFrame(
    {
        "Id": train_data["Id"],
        "PATIENT": train_data["PATIENT"],
        "ACTUAL": y_train,
        "OOF_PROBABILITY": oof_probabilities,
        "OOF_FOLD": oof_fold,
    }
)

oof_path = (
    REPORT_DIR
    / "random_forest_oof_threshold_predictions.csv"
)

oof_results.to_csv(
    oof_path,
    index=False,
)

print("\nOOF predictions saved:")
print(oof_path)


# ============================================================
# 6. EVALUATE EACH CLASSIFICATION THRESHOLD
# ============================================================

print("\n" + "=" * 65)
print("EVALUATING CLASSIFICATION THRESHOLDS")
print("=" * 65)

threshold_results = []

for threshold in THRESHOLDS:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    accuracy = accuracy_score(
        y_train,
        predictions,
    )

    predicted_positive_rate = (
        predictions.mean()
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    threshold_results.append(
        {
            "THRESHOLD": threshold,
            "PRECISION": precision,
            "RECALL": recall,
            "F1": f1,
            "SPECIFICITY": specificity,
            "ACCURACY": accuracy,
            "PREDICTED_POSITIVE_RATE": predicted_positive_rate,
            "FALSE_POSITIVE_RATE": false_positive_rate,
            "FALSE_NEGATIVE_RATE": false_negative_rate,
            "TRUE_POSITIVES": tp,
            "TRUE_NEGATIVES": tn,
            "FALSE_POSITIVES": fp,
            "FALSE_NEGATIVES": fn,
        }
    )


threshold_metrics = pd.DataFrame(
    threshold_results
)


# ============================================================
# 7. SAVE THRESHOLD RESULTS
# ============================================================

threshold_metrics_path = (
    REPORT_DIR
    / "random_forest_threshold_metrics.csv"
)

threshold_metrics.to_csv(
    threshold_metrics_path,
    index=False,
)

print("\nThreshold metrics saved:")
print(threshold_metrics_path)


# ============================================================
# 8. DISPLAY SELECTED THRESHOLDS
# ============================================================

print("\n" + "=" * 65)
print("SELECTED THRESHOLD COMPARISON")
print("=" * 65)

selected_thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]

selected_results = threshold_metrics[
    threshold_metrics["THRESHOLD"].isin(
        selected_thresholds
    )
]

print(
    selected_results[
        [
            "THRESHOLD",
            "PRECISION",
            "RECALL",
            "F1",
            "SPECIFICITY",
            "PREDICTED_POSITIVE_RATE",
            "FALSE_POSITIVES",
            "FALSE_NEGATIVES",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 9. IDENTIFY THE THRESHOLD WITH THE HIGHEST F1
# ============================================================

best_f1_row = threshold_metrics.loc[
    threshold_metrics["F1"].idxmax()
]

print("\n" + "=" * 65)
print("HIGHEST F1 THRESHOLD")
print("=" * 65)

print(
    "Threshold:",
    best_f1_row["THRESHOLD"],
)

print(
    "Precision:",
    round(best_f1_row["PRECISION"], 4),
)

print(
    "Recall:",
    round(best_f1_row["RECALL"], 4),
)

print(
    "F1:",
    round(best_f1_row["F1"], 4),
)

print(
    "Specificity:",
    round(best_f1_row["SPECIFICITY"], 4),
)

print(
    "False positives:",
    int(best_f1_row["FALSE_POSITIVES"]),
)

print(
    "False negatives:",
    int(best_f1_row["FALSE_NEGATIVES"]),
)


# ============================================================
# 10. PLOT PRECISION, RECALL AND F1
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_metrics["THRESHOLD"],
    threshold_metrics["PRECISION"],
    label="Precision",
)

plt.plot(
    threshold_metrics["THRESHOLD"],
    threshold_metrics["RECALL"],
    label="Recall",
)

plt.plot(
    threshold_metrics["THRESHOLD"],
    threshold_metrics["F1"],
    label="F1",
)

plt.axvline(
    x=DEFAULT_THRESHOLD,
    linestyle="--",
    linewidth=1,
    label="Default threshold = 0.50",
)

plt.axvline(
    x=best_f1_row["THRESHOLD"],
    linestyle=":",
    linewidth=1,
    label="Highest F1 threshold",
)

plt.xlabel("Classification threshold")
plt.ylabel("Metric value")

plt.title(
    "Random Forest - Threshold Analysis "
    "(Patient-Grouped OOF Predictions)"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

figure_path = (
    FIGURE_DIR
    / "random_forest_threshold_analysis.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nThreshold analysis figure saved:")
print(figure_path)


# ============================================================
# 11. FINAL NOTES
# ============================================================

print("\n" + "=" * 65)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 65)

print("\nGenerated files:")
print("1. random_forest_oof_threshold_predictions.csv")
print("2. random_forest_threshold_metrics.csv")
print("3. random_forest_threshold_analysis.png")

print("\nImportant:")
print(
    "The threshold with the highest F1 is a candidate, "
    "not automatically the final threshold."
)

print(
    "Threshold selection should consider the operational "
    "trade-off between missed repeat encounters and "
    "unnecessary positive predictions."
)

print(
    "The test dataset was not used in this analysis."
)