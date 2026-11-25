from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


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


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

print("Loading test dataset...")
test_data = pd.read_csv(TEST_DATA_PATH)

print("Loading fitted Random Forest pipeline...")
model = joblib.load(MODEL_PATH)

X_test = test_data[FEATURES]
y_test = test_data[TARGET].astype(int)

print("\nTest dataset shape:", test_data.shape)
print("Number of predictors:", len(FEATURES))


# ============================================================
# 3. BASELINE MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("BASELINE MODEL PERFORMANCE")
print("=" * 65)

test_probabilities = model.predict_proba(X_test)[:, 1]

baseline_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

print(f"Baseline test ROC-AUC: {baseline_roc_auc:.4f}")


# ============================================================
# 4. CALCULATE PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("CALCULATING PERMUTATION IMPORTANCE")
print("=" * 65)

print("This may take a few minutes...")

importance_result = permutation_importance(
    estimator=model,
    X=X_test,
    y=y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=1,
)


# ============================================================
# 5. CREATE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame(
    {
        "FEATURE": FEATURES,
        "IMPORTANCE_MEAN": importance_result.importances_mean,
        "IMPORTANCE_STD": importance_result.importances_std,
    }
)

importance_df = importance_df.sort_values(
    "IMPORTANCE_MEAN",
    ascending=False,
)

importance_df["IMPORTANCE_RANK"] = range(
    1,
    len(importance_df) + 1,
)

importance_df = importance_df[
    [
        "IMPORTANCE_RANK",
        "FEATURE",
        "IMPORTANCE_MEAN",
        "IMPORTANCE_STD",
    ]
]

print("\n" + "=" * 65)
print("PERMUTATION IMPORTANCE RESULTS")
print("=" * 65)

print(
    importance_df.round(5).to_string(index=False)
)


# ============================================================
# 6. SAVE IMPORTANCE RESULTS
# ============================================================

importance_path = (
    REPORT_DIR
    / "random_forest_permutation_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False,
)

print("\nImportance table saved:")
print(importance_path)


# ============================================================
# 7. VISUALIZE FEATURE IMPORTANCE
# ============================================================

plot_data = importance_df.sort_values(
    "IMPORTANCE_MEAN",
    ascending=True,
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_data["FEATURE"],
    plot_data["IMPORTANCE_MEAN"],
    xerr=plot_data["IMPORTANCE_STD"],
    capsize=3,
)

plt.axvline(
    x=0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel(
    "Decrease in ROC-AUC after permutation"
)

plt.ylabel("Predictor")

plt.title(
    "Random Forest - Permutation Feature Importance"
)

plt.tight_layout()

figure_path = (
    FIGURE_DIR
    / "random_forest_permutation_importance.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nFeature importance figure saved:")
print(figure_path)


# ============================================================
# 8. INTERPRETATION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("INTERPRETATION SUMMARY")
print("=" * 65)

top_features = importance_df.head(5)

print("\nTop five predictors:")

for _, row in top_features.iterrows():

    print(
        f"{row['IMPORTANCE_RANK']}. "
        f"{row['FEATURE']}: "
        f"{row['IMPORTANCE_MEAN']:.5f}"
    )


print("\nInterpretation:")

print(
    "A larger positive importance means that shuffling "
    "the predictor reduced the model's ROC-AUC more."
)

print(
    "An importance close to zero means that shuffling "
    "the predictor had little effect on ROC-AUC."
)

print(
    "A negative importance means that the model performed "
    "slightly better after the predictor was shuffled."
)

print("\nImportant limitation:")

print(
    "Permutation importance measures how much the fitted "
    "model relies on a predictor for its test-set performance. "
    "It does not establish causation."
)

print(
    "Correlated predictors may share importance, so the "
    "importance of one predictor can be understated when "
    "another predictor contains similar information."
)

print("\nFeature importance analysis completed.")