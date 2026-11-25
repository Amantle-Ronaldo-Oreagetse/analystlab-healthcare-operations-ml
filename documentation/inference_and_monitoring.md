# Inference and Monitoring

## 1. Overview

This document describes the inference workflow for the Healthcare Analytics and AI Implementation project.

The project includes a saved Random Forest pipeline that estimates the probability of a subsequent healthcare encounter being recorded within seven days after the current encounter ends.

The inference workflow allows a user to:

- Load the saved model.
- Provide new encounter-level records.
- Validate input features.
- Generate predicted probabilities.
- Convert probabilities into binary predictions.
- Save prediction results to a CSV file.

The current implementation is a working machine learning prototype. It is not a deployed healthcare system and has not been clinically validated.

---

## 2. Saved Model

The selected Random Forest pipeline is stored at:

`models/repeat_encounter_7day/random_forest.joblib`

The saved pipeline includes:

- Numerical preprocessing.
- Categorical preprocessing.
- The fitted Random Forest classifier.

The model was selected based on its overall performance during cross-validation and held-out test evaluation.

The model uses nine predictors:

1. `ENCOUNTERCLASS`
2. `DURATION_HOURS`
3. `AGE_AT_ENCOUNTER`
4. `PRIOR_ENCOUNTERS_30_DAYS`
5. `PRIOR_ENCOUNTERS_90_DAYS`
6. `DAYS_SINCE_PREVIOUS_ENCOUNTER`
7. `START_DAY_OF_WEEK`
8. `START_MONTH`
9. `START_HOUR`

Encounter ID and patient ID are not required for inference because they are not model predictors.

---

## 3. Inference Script

The inference script is:

`src/models/13_inference.py`

The script performs the following operations:

1. Loads the saved Random Forest pipeline.
2. Reads an input CSV file or creates a demonstration record.
3. Checks that the required predictors are present.
4. Validates encounter classes.
5. Validates numerical predictors.
6. Generates predicted probabilities.
7. Applies the configured decision threshold.
8. Creates prediction labels.
9. Saves the results to a CSV file.

The script accepts command-line arguments for the input and output files.

---

## 4. Input Requirements

The input CSV must contain all nine required predictors.

Example:

```csv
ENCOUNTERCLASS,DURATION_HOURS,AGE_AT_ENCOUNTER,PRIOR_ENCOUNTERS_30_DAYS,PRIOR_ENCOUNTERS_90_DAYS,DAYS_SINCE_PREVIOUS_ENCOUNTER,START_DAY_OF_WEEK,START_MONTH,START_HOUR
ambulatory,1.5,45,2,4,12.0,2,6,10
emergency,3.0,67,1,3,25.0,4,8,15
outpatient,0.8,32,0,1,90.0,1,3,9
```

These records are illustrative examples and do not represent real patients.

### 4.1 Encounter class values

The script accepts the following encounter classes:

- `ambulatory`
- `emergency`
- `inpatient`
- `outpatient`
- `urgentcare`
- `wellness`

Encounter class values are converted to lowercase and stripped of leading or trailing whitespace before validation.

Missing or invalid encounter classes are rejected.

### 4.2 Numerical values

Numerical predictors must contain valid numeric values or legitimate missing values.

The script rejects malformed numerical values, such as text entered into a numerical field.

Missing numerical values are allowed because the fitted preprocessing pipeline includes median imputation.

The inference script does not perform feature engineering from raw encounter or patient records. The input data must already contain the required engineered predictors in the correct format.

---

## 5. Running Inference

Run commands from the project root directory.

### 5.1 Demonstration inference

To run the built-in demonstration:

```powershell
python src/models/13_inference.py
```

The script loads the saved model and generates a prediction for the demonstration record.

### 5.2 Batch inference

To generate predictions for an input CSV:

```powershell
python src/models/13_inference.py --input data/processed/example_inference.csv
```

The script processes all records in the supplied file.

### 5.3 Specify an output filename

An alternative output filename can be provided:

```powershell
python src/models/13_inference.py --input data/processed/example_inference.csv --output batch_predictions.csv
```

The output is saved under:

`reports/inference/`

The default output filename is:

`inference_predictions.csv`

---

## 6. Prediction Output

The inference results contain the input predictors and the following additional columns.

| Output column | Description |
|---|---|
| `PREDICTED_PROBABILITY` | Estimated probability of a subsequent encounter within seven days. |
| `PREDICTED_CLASS` | Binary prediction generated using the decision threshold. |
| `PREDICTION_LABEL` | Text description of the binary prediction. |
| `DECISION_THRESHOLD` | Threshold used to convert the probability into a binary prediction. |

### 6.1 Decision threshold

The current inference threshold is:

`0.50`

The classification rule is:

- Probability greater than or equal to 0.50: class 1.
- Probability below 0.50: class 0.

Class 0 is labelled:

`No subsequent encounter observed within 7 days`

Class 1 is labelled:

`Subsequent encounter predicted within 7 days`

These labels describe the model's prediction. They do not guarantee that a subsequent encounter will or will not occur.

### 6.2 Threshold consideration

Training-set out-of-fold threshold analysis identified 0.32 as a candidate threshold with the highest F1-score.

However, the inference script currently uses 0.50.

The candidate threshold was not automatically adopted because threshold selection should consider the intended operational use, the cost of false negatives and false positives, and the capacity to review alerts.

---

## 7. Inference Testing

The inference workflow was tested using a three-record CSV input.

The script successfully:

- Loaded the saved Random Forest pipeline.
- Read three input records.
- Generated three predicted probabilities.
- Generated three binary predictions.
- Saved the prediction results to CSV.

The demonstration and batch inference tests confirmed that the model could be loaded and used to generate predictions through the command-line interface.

These tests verify the operation of the inference workflow. They do not measure predictive accuracy.

Predictive accuracy was evaluated separately using labelled training and held-out test data.

---

## 8. Input Validation Tests

A test suite was created at:

`tests/test_inference.py`

The tests use Python's built-in `unittest` framework.

The following cases were tested:

| Test | Expected behavior |
|---|---|
| Valid input | Accepted |
| Empty dataset | Rejected |
| Missing required predictor | Rejected |
| Invalid encounter class | Rejected |
| Malformed numerical value | Rejected |
| Missing numerical value | Accepted |
| Prediction output | Valid probability and binary class |

All seven tests passed.

To run the test suite:

```powershell
python -m unittest discover -s tests -p "test_inference.py" -v
```

The successful tests demonstrate that the inference validation functions handle the tested input conditions as expected.

They do not establish that every possible invalid input or runtime failure has been covered.

---

## 9. Monitoring Recommendations

If the model were considered for use in an operational environment, monitoring would be required.

The following monitoring activities are recommended.

### 9.1 Input data quality

Monitor:

- Missing required predictors.
- Invalid encounter class values.
- Unexpected numerical values.
- Changes in missing-value rates.
- Changes in feature distributions.
- Differences between the training data format and incoming data.

Input validation should remain active whenever the model is used.

### 9.2 Prediction distribution

Monitor:

- Distribution of predicted probabilities.
- Proportion of encounters classified as positive.
- Changes in the number of predictions by encounter class.
- Changes in prediction frequency over time.

A substantial change in prediction distribution may indicate changes in the input population, data collection, or model behavior.

### 9.3 Model performance

When actual outcomes become available, evaluate:

- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Average Precision.
- False positive rate.
- False negative rate.

Performance should also be examined by encounter class because the initial evaluation showed substantial variation across classes.

### 9.4 Data and concept drift

Monitor whether the distribution of incoming data differs from the data used to train the model.

Potential changes include:

- Encounter-class frequency.
- Patient encounter history.
- Encounter duration.
- Age distribution.
- Changes in healthcare service patterns.
- Changes in data collection or coding practices.

Changes in data distribution do not automatically prove that model performance has deteriorated, but they may indicate a need for investigation.

### 9.5 Threshold and alert capacity

If predictions are used to generate alerts, monitor:

- Number of positive predictions.
- Number of alerts requiring review.
- False positive burden.
- Missed positive cases.
- Whether the chosen threshold remains appropriate for the intended workflow.

The threshold should not be changed solely to maximize a metric without considering the consequences of the resulting predictions.

---

## 10. Responsible Use

The model was developed for educational and analytical purposes.

It should not be used to:

- Diagnose medical conditions.
- Recommend treatment.
- Replace clinical judgment.
- Independently determine patient care.
- Make decisions about access to healthcare.
- Claim that a patient will definitely return or will definitely not return.

The target represents a subsequent encounter recorded in the available dataset. It is not a clinically validated readmission label.

The model has not been externally validated or deployed in a healthcare environment.

Any future operational use would require appropriate data governance, privacy review, clinical and operational evaluation, validation on relevant populations, and oversight by qualified professionals.

---

## 11. Limitations

The inference workflow has several limitations:

1. The input must already contain the engineered predictors.
2. The script does not automatically retrieve patient history or create features from raw healthcare records.
3. The model was trained on historical data from one organization.
4. The dataset covers 2011–2022.
5. The model has not been validated on an external dataset.
6. The current threshold is a modeling default, not an operationally approved threshold.
7. The inference prototype does not include a user interface, database integration, or automated clinical workflow.
8. Successful inference and input validation do not establish clinical validity.

The current implementation demonstrates a reproducible model inference workflow that can be extended for further analytical development.