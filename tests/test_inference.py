import sys
import unittest
import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# LOAD INFERENCE MODULE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INFERENCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "13_inference.py"
)

spec = importlib.util.spec_from_file_location(
    "inference_module",
    INFERENCE_PATH
)

inference = importlib.util.module_from_spec(spec)
sys.modules["inference_module"] = inference
spec.loader.exec_module(inference)


# ============================================================
# MOCK MODEL FOR TESTING
# ============================================================

class MockModel:

    def predict_proba(self, data):

        probabilities = np.array([
            [0.70, 0.30],
            [0.20, 0.80],
            [0.90, 0.10],
        ])

        return probabilities[:len(data)]


# ============================================================
# TEST CASES
# ============================================================

class TestInferenceValidation(unittest.TestCase):

    def setUp(self):

        self.valid_data = pd.DataFrame([
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

    def test_valid_input_is_accepted(self):

        validated = inference.validate_input(
            self.valid_data
        )

        self.assertEqual(len(validated), 1)

        self.assertEqual(
            list(validated.columns),
            inference.FEATURES
        )

    def test_empty_dataset_is_rejected(self):

        empty_data = pd.DataFrame(
            columns=inference.FEATURES
        )

        with self.assertRaises(ValueError):

            inference.validate_input(empty_data)

    def test_missing_feature_is_rejected(self):

        incomplete_data = self.valid_data.drop(
            columns=["AGE_AT_ENCOUNTER"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Missing required predictors"
        ):

            inference.validate_input(incomplete_data)

    def test_invalid_encounter_class_is_rejected(self):

        invalid_data = self.valid_data.copy()

        invalid_data.loc[0, "ENCOUNTERCLASS"] = "unknown"

        with self.assertRaisesRegex(
            ValueError,
            "Invalid or missing encounter classes"
        ):

            inference.validate_input(invalid_data)

    def test_malformed_numeric_value_is_rejected(self):

        invalid_data = self.valid_data.copy()

        invalid_data["AGE_AT_ENCOUNTER"] = (
        invalid_data["AGE_AT_ENCOUNTER"].astype("object"))

        invalid_data.loc[0, "AGE_AT_ENCOUNTER"] = "forty-five"

        with self.assertRaisesRegex(
            ValueError,
            "invalid non-numeric values"
        ):

            inference.validate_input(invalid_data)

    def test_missing_numeric_value_is_allowed(self):

        incomplete_data = self.valid_data.copy()

        incomplete_data.loc[
            0,
            "DAYS_SINCE_PREVIOUS_ENCOUNTER"
        ] = np.nan

        validated = inference.validate_input(
            incomplete_data
        )

        self.assertTrue(
            pd.isna(
                validated.loc[
                    0,
                    "DAYS_SINCE_PREVIOUS_ENCOUNTER"
                ]
            )
        )

    def test_prediction_output_is_valid(self):

        model = MockModel()

        validated = inference.validate_input(
            self.valid_data
        )

        results = inference.generate_predictions(
            model,
            validated
        )

        self.assertIn(
            "PREDICTED_PROBABILITY",
            results.columns
        )

        self.assertIn(
            "PREDICTED_CLASS",
            results.columns
        )

        self.assertIn(
            "PREDICTION_LABEL",
            results.columns
        )

        self.assertTrue(
            results["PREDICTED_PROBABILITY"].between(
                0,
                1
            ).all()
        )

        self.assertTrue(
            results["PREDICTED_CLASS"].isin(
                [0, 1]
            ).all()
        )


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":

    unittest.main(verbosity=2)
