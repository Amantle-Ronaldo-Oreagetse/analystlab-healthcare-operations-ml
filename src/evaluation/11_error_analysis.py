from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "test_dataset.csv"
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

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

THRESHOLD = 0.50


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

print("Loading test dataset...")
test_data = pd.read_csv(TEST_DATA_PATH)

print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)

print("\nDataset shape:", test_data.shape)
print("Number of test encounters:", len(test_data))


# ============================================================
# 3. PREPARE FEATURES AND GENERATE PREDICTIONS
# ============================================================

X_test = test_data[FEATURES]
y_test = test_data[TARGET].astype(int)

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= THRESHOLD).astype(int)


# ============================================================
# 4. CREATE PREDICTION-LEVEL ERROR CATEGORIES
# ============================================================

prediction_results = test_data[
    ["Id", "PATIENT", "ENCOUNTERCLASS"]
].copy()

prediction_results["ACTUAL"] = y_test.to_numpy()
prediction_results["PREDICTED"] = predictions
prediction_results["PREDICTED_PROBABILITY"] = probabilities

prediction_results["ERROR_CATEGORY"] = np.select(
    [
        (prediction_results["ACTUAL"] == 1)
        & (prediction_results["PREDICTED"] == 1),

        (prediction_results["ACTUAL"] == 0)
        & (prediction_results["PREDICTED"] == 0),

        (prediction_results["ACTUAL"] == 0)
        & (prediction_results["PREDICTED"] == 1),

        (prediction_results["ACTUAL"] == 1)
        & (prediction_results["PREDICTED"] == 0),
    ],
    [
        "True Positive",
        "True Negative",
        "False Positive",
        "False Negative",
    ],
    default="Unknown",
)


# Save individual predictions
prediction_results.to_csv(
    REPORT_DIR / "random_forest_test_predictions.csv",
    index=False,
)


# ============================================================
# 5. OVERALL ERROR SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("OVERALL ERROR ANALYSIS")
print("=" * 65)

error_counts = (
    prediction_results["ERROR_CATEGORY"]
    .value_counts()
    .reindex(
        [
            "True Positive",
            "True Negative",
            "False Positive",
            "False Negative",
        ],
        fill_value=0,
    )
)

print("\nPrediction outcome counts:")
print(error_counts)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
).ravel()

print("\nConfusion matrix:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

print("\nOverall error rates:")
print(f"False positive rate: {fp / (fp + tn):.4f}")
print(f"False negative rate: {fn / (fn + tp):.4f}")


# ============================================================
# 6. PERFORMANCE BY ENCOUNTER CLASS
# ============================================================

print("\n" + "=" * 65)
print("PERFORMANCE BY ENCOUNTER CLASS")
print("=" * 65)

class_results = []

for encounter_class, group in prediction_results.groupby(
    "ENCOUNTERCLASS"
):

    actual = group["ACTUAL"]
    predicted = group["PREDICTED"]
    probability = group["PREDICTED_PROBABILITY"]

    class_tn, class_fp, class_fn, class_tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        actual,
        predicted,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0,
    )

    specificity = (
        class_tn / (class_tn + class_fp)
        if (class_tn + class_fp) > 0
        else np.nan
    )

    # ROC-AUC and average precision require both target classes.
    if actual.nunique() == 2:
        roc_auc = roc_auc_score(actual, probability)
        average_precision = average_precision_score(
            actual,
            probability,
        )
    else:
        roc_auc = np.nan
        average_precision = np.nan

    class_results.append(
        {
            "ENCOUNTERCLASS": encounter_class,
            "TOTAL_ENCOUNTERS": len(group),
            "ACTUAL_POSITIVES": int(actual.sum()),
            "ACTUAL_NEGATIVES": int((actual == 0).sum()),
            "POSITIVE_RATE": actual.mean(),
            "PREDICTED_POSITIVES": int(predicted.sum()),
            "TRUE_POSITIVES": class_tp,
            "TRUE_NEGATIVES": class_tn,
            "FALSE_POSITIVES": class_fp,
            "FALSE_NEGATIVES": class_fn,
            "PRECISION": precision,
            "RECALL": recall,
            "SPECIFICITY": specificity,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "AVERAGE_PRECISION": average_precision,
        }
    )


class_metrics = pd.DataFrame(class_results)

class_metrics = class_metrics.sort_values(
    "TOTAL_ENCOUNTERS",
    ascending=False,
)

print(
    class_metrics.round(4).to_string(index=False)
)

class_metrics.to_csv(
    REPORT_DIR / "random_forest_performance_by_encounter_class.csv",
    index=False,
)


# ============================================================
# 7. FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("ERROR COUNTS BY ENCOUNTER CLASS")
print("=" * 65)

error_by_class = (
    prediction_results
    .groupby("ENCOUNTERCLASS")
    .agg(
        TOTAL_ENCOUNTERS=("ACTUAL", "size"),
        FALSE_POSITIVES=(
            "ERROR_CATEGORY",
            lambda x: (x == "False Positive").sum(),
        ),
        FALSE_NEGATIVES=(
            "ERROR_CATEGORY",
            lambda x: (x == "False Negative").sum(),
        ),
        TRUE_POSITIVES=(
            "ERROR_CATEGORY",
            lambda x: (x == "True Positive").sum(),
        ),
        TRUE_NEGATIVES=(
            "ERROR_CATEGORY",
            lambda x: (x == "True Negative").sum(),
        ),
    )
    .reset_index()
)

error_by_class["FALSE_POSITIVE_RATE"] = (
    error_by_class["FALSE_POSITIVES"]
    / (
        error_by_class["FALSE_POSITIVES"]
        + error_by_class["TRUE_NEGATIVES"]
    )
)

error_by_class["FALSE_NEGATIVE_RATE"] = (
    error_by_class["FALSE_NEGATIVES"]
    / (
        error_by_class["FALSE_NEGATIVES"]
        + error_by_class["TRUE_POSITIVES"]
    )
)

print(error_by_class.round(4).to_string(index=False))

error_by_class.to_csv(
    REPORT_DIR / "random_forest_errors_by_encounter_class.csv",
    index=False,
)


# ============================================================
# 8. EXAMINE FALSE NEGATIVES
# ============================================================

print("\n" + "=" * 65)
print("FALSE NEGATIVES: ACTUAL REPEAT ENCOUNTERS MISSED")
print("=" * 65)

false_negatives = prediction_results[
    prediction_results["ERROR_CATEGORY"] == "False Negative"
].copy()

false_negatives = false_negatives.sort_values(
    "PREDICTED_PROBABILITY",
    ascending=False,
)

print("Total false negatives:", len(false_negatives))

print("\nHighest-confidence false negatives:")
print(
    false_negatives[
        [
            "ENCOUNTERCLASS",
            "ACTUAL",
            "PREDICTED",
            "PREDICTED_PROBABILITY",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

false_negatives.to_csv(
    REPORT_DIR / "random_forest_false_negatives.csv",
    index=False,
)


# ============================================================
# 9. EXAMINE FALSE POSITIVES
# ============================================================

print("\n" + "=" * 65)
print("FALSE POSITIVES: PREDICTED REPEATS THAT DID NOT OCCUR")
print("=" * 65)

false_positives = prediction_results[
    prediction_results["ERROR_CATEGORY"] == "False Positive"
].copy()

false_positives = false_positives.sort_values(
    "PREDICTED_PROBABILITY",
    ascending=False,
)

print("Total false positives:", len(false_positives))

print("\nHighest-confidence false positives:")
print(
    false_positives[
        [
            "ENCOUNTERCLASS",
            "ACTUAL",
            "PREDICTED",
            "PREDICTED_PROBABILITY",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

false_positives.to_csv(
    REPORT_DIR / "random_forest_false_positives.csv",
    index=False,
)


# ============================================================
# 10. PREDICTION CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("PREDICTION CONFIDENCE ANALYSIS")
print("=" * 65)

confidence_summary = (
    prediction_results
    .groupby("ERROR_CATEGORY")
    ["PREDICTED_PROBABILITY"]
    .agg(
        COUNT="count",
        MEAN="mean",
        MEDIAN="median",
        MIN="min",
        MAX="max",
    )
    .reset_index()
)

print(confidence_summary.round(4).to_string(index=False))

confidence_summary.to_csv(
    REPORT_DIR / "random_forest_prediction_confidence.csv",
    index=False,
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("\nFiles saved in:")
print(REPORT_DIR)

print("\nGenerated reports:")
print("1. random_forest_test_predictions.csv")
print("2. random_forest_performance_by_encounter_class.csv")
print("3. random_forest_errors_by_encounter_class.csv")
print("4. random_forest_false_negatives.csv")
print("5. random_forest_false_positives.csv")
print("6. random_forest_prediction_confidence.csv")

print("\nInterpretation reminder:")
print(
    "False negatives are actual repeat encounters the model missed."
)
print(
    "False positives are predicted repeat encounters that did not occur."
)
print(
    "These results describe performance at the existing 0.50 threshold."
)