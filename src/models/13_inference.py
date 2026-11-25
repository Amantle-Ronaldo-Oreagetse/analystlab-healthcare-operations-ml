from pathlib import Path
import argparse

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "repeat_encounter_7day"
    / "random_forest.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "inference"

THRESHOLD = 0.50

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

NUMERIC_FEATURES = [
    "DURATION_HOURS",
    "AGE_AT_ENCOUNTER",
    "PRIOR_ENCOUNTERS_30_DAYS",
    "PRIOR_ENCOUNTERS_90_DAYS",
    "DAYS_SINCE_PREVIOUS_ENCOUNTER",
    "START_DAY_OF_WEEK",
    "START_MONTH",
    "START_HOUR",
]

VALID_ENCOUNTER_CLASSES = {
    "ambulatory",
    "emergency",
    "inpatient",
    "outpatient",
    "urgentcare",
    "wellness",
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def validate_input(data):

    if data.empty:
        raise ValueError("Input dataset contains no records.")

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required predictors: {missing_features}"
        )

    data = data[FEATURES].copy()

    # Validate encounter class.
    data["ENCOUNTERCLASS"] = (
        data["ENCOUNTERCLASS"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    invalid_classes = (
        data["ENCOUNTERCLASS"].isna()
        | ~data["ENCOUNTERCLASS"].isin(
            VALID_ENCOUNTER_CLASSES
        )
    )

    if invalid_classes.any():
        invalid_values = (
            data.loc[
                invalid_classes,
                "ENCOUNTERCLASS"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Invalid or missing encounter classes: {invalid_values}"
        )

    # Validate numeric predictors.
    for column in NUMERIC_FEATURES:

        original_values = data[column]

        converted_values = pd.to_numeric(
            original_values,
            errors="coerce"
        )

        invalid_values = (
            original_values.notna()
            & converted_values.isna()
        )

        if invalid_values.any():
            raise ValueError(
                f"Column '{column}' contains "
                "invalid non-numeric values."
            )

        data[column] = converted_values

    return data


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, data):

    probabilities = model.predict_proba(data)[:, 1]

    predicted_classes = (
        probabilities >= THRESHOLD
    ).astype(int)

    results = data.copy()

    results["PREDICTED_PROBABILITY"] = probabilities

    results["PREDICTED_CLASS"] = predicted_classes

    results["PREDICTION_LABEL"] = (
        results["PREDICTED_CLASS"]
        .map({
            0: "No subsequent encounter observed within 7 days",
            1: "Subsequent encounter predicted within 7 days",
        })
    )

    results["DECISION_THRESHOLD"] = THRESHOLD

    return results


# ============================================================
# DEMONSTRATION INPUT
# ============================================================

def create_demo_input():

    return pd.DataFrame([
        {
            "ENCOUNTERCLASS": "ambulatory",
            "DURATION_HOURS": 1.5,
            "AGE_AT_ENCOUNTER": 45,
            "PRIOR_ENCOUNTERS_30_DAYS": 2,
            "PRIOR_ENCOUNTERS_90_DAYS": 4,
            "DAYS_SINCE_PREVIOUS_ENCOUNTER": 12.0,
            "START_DAY_OF_WEEK": 2,
            "START_MONTH": 6,
            "START_HOUR": 10,
        }
    ])


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict the probability of a subsequent "
            "encounter within seven days after encounter end."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to an input CSV file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="inference_predictions.csv",
        help="Output CSV filename."
    )

    args = parser.parse_args()

    print("\nLoading saved Random Forest model...")

    model = load_model()

    if args.input:

        input_path = Path(args.input)

        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {input_path}"
            )

        print(f"Loading input data: {input_path}")

        input_data = pd.read_csv(input_path)

    else:

        print("No input CSV supplied.")
        print("Running the built-in demonstration.")

        input_data = create_demo_input()

    print(f"Input records: {len(input_data)}")

    validated_data = validate_input(input_data)

    results = generate_predictions(
        model,
        validated_data
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / args.output

    results.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)

    print(
        results[
            [
                "PREDICTED_PROBABILITY",
                "PREDICTED_CLASS",
                "PREDICTION_LABEL",
            ]
        ].to_string(index=False)
    )

    print(f"\nDecision threshold: {THRESHOLD}")

    print(f"Predictions saved to: {output_path}")

    print("\nInference completed successfully.")


if __name__ == "__main__":
    main()