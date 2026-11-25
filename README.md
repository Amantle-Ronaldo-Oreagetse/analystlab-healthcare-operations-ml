# Healthcare Operations: Seven-Day Subsequent Encounter Prediction

## Project Overview

This project develops a machine learning workflow to predict whether a patient will have a subsequent healthcare encounter recorded within seven days after the current encounter ends.

The project was completed as part of AnalystLab Africa's Healthcare Analytics and AI Implementation assignment, Machine Learning Track.

The objective was to demonstrate an end-to-end machine learning workflow, including:

- Data inspection and validation.
- Data preparation and feature engineering.
- Target definition and leakage prevention.
- Patient-grouped train-test splitting.
- Baseline and model comparison.
- Model evaluation and error analysis.
- Feature importance analysis.
- Decision threshold analysis.
- Saved-model inference.
- Input validation and testing.
- Documentation and reproducibility.

The final model is a Random Forest classifier.

**This project is an educational and analytical prototype. It is not a clinically validated prediction system.**

---

## Project Question

Can encounter-level information and a patient's historical encounter patterns be used to estimate the probability of a subsequent recorded healthcare encounter within seven days after the current encounter ends?

The model predicts a subsequent encounter in the available dataset. It does not predict a clinically validated readmission outcome.

---

## Dataset

The project uses healthcare encounter data containing:

- Encounters
- Patients
- Procedures
- Organizations
- Payers
- A data dictionary

The source data covers encounters recorded between 2011 and 2022.

### Dataset summary

| Dataset | Records |
|---|---:|
| Encounters | 27,891 |
| Patients | 974 |
| Procedures | 47,701 |
| Payers | 10 |
| Organizations | 1 |

After target construction and observation-window eligibility checks, the final modeling dataset contained:

**27,887 eligible encounters**

The raw datasets are not included in this repository because of their size and the presence of sensitive information.

Users must obtain the source data from an authorized source before reproducing the workflow.

---

## Target Definition

The target variable is:

`REPEAT_WITHIN_7_DAYS`

| Value | Meaning |
|---|---|
| 1 | A subsequent encounter START is observed within seven days after the current encounter STOP. |
| 0 | No subsequent encounter START is observed within seven days, provided the full observation window is available. |

The prediction point is the end of the current encounter.

Encounters without a complete seven-day observation window were excluded to avoid incorrectly assigning negative labels to records with insufficient follow-up.

### Target distribution

| Target | Encounters |
|---|---:|
| 0 | 18,121 |
| 1 | 9,766 |
| Total | 27,887 |

Positive rate: **35.02%**

The target represents a subsequent recorded encounter, not a clinically validated readmission event.

---

## Features

The model uses nine predictors.

| Feature | Description |
|---|---|
| `ENCOUNTERCLASS` | Class of the current encounter |
| `DURATION_HOURS` | Current encounter duration in hours |
| `AGE_AT_ENCOUNTER` | Patient age at the current encounter |
| `PRIOR_ENCOUNTERS_30_DAYS` | Number of previous encounters in the preceding 30 days |
| `PRIOR_ENCOUNTERS_90_DAYS` | Number of previous encounters in the preceding 90 days |
| `DAYS_SINCE_PREVIOUS_ENCOUNTER` | Time since the previous encounter |
| `START_DAY_OF_WEEK` | Day of week of the current encounter |
| `START_MONTH` | Month of the current encounter |
| `START_HOUR` | Hour of the current encounter |

Patient and encounter identifiers were retained for traceability and grouped splitting but were excluded from model predictors.

Future encounter information, target-related fields, and identifying information were excluded to reduce leakage and protect privacy.

---

## Modeling Approach

### Train-test split

The data was split using five-fold stratified group cross-validation.

Configuration:

- Group variable: Patient ID
- Number of folds: 5
- Shuffle: True
- Random state: 42

The first fold was used as the held-out test set.

| Measure | Training | Testing |
|---|---:|---:|
| Encounters | 22,305 | 5,582 |
| Patients | 778 | 196 |
| Positive cases | 7,812 | 1,954 |
| Positive rate | 35.02% | 35.01% |

No patients appeared in both training and testing datasets.

This split evaluates generalization to unseen patients. It does not constitute strict temporal validation.

### Preprocessing

Numerical features were processed using:

- Median imputation
- Standardization

The categorical feature `ENCOUNTERCLASS` was processed using:

- Most-frequent-value imputation
- One-hot encoding

Preprocessing was fitted within the model pipelines using training data only.

### Models compared

The following classifiers were evaluated:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest
4. RBF Support Vector Machine

Random Forest was selected as the final model based on its overall cross-validation and test-set performance.

---

## Model Results

### Five-fold cross-validation

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Dummy Classifier | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.8947 | 0.6669 | 0.7637 | 0.9010 |
| Random Forest | 0.9131 | 0.6660 | 0.7646 | 0.9182 |
| RBF SVM | 0.8322 | 0.6840 | 0.7468 | 0.8865 |

Random Forest achieved the highest mean F1-score and ROC-AUC during cross-validation.

### Held-out test results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.6499 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3501 |
| Logistic Regression | 0.9061 | 0.9503 | 0.7723 | 0.8521 | 0.9302 | 0.9227 |
| Random Forest | 0.9233 | 0.9722 | 0.8040 | 0.8801 | 0.9454 | 0.9402 |
| RBF SVM | 0.8956 | 0.9513 | 0.7395 | 0.8321 | 0.9169 | 0.9044 |

The Random Forest achieved the strongest overall test-set results among the evaluated models.

### Random Forest confusion matrix

At the default decision threshold of 0.50:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 3,583 | 45 |
| Actual positive | 383 | 1,571 |

The model identified 1,571 of the 1,954 positive test encounters.

---

## Error Analysis

Model performance varied across encounter classes.

| Encounter class | Test encounters | Positive cases | Recall |
|---|---:|---:|---:|
| Ambulatory | 3,033 | 1,583 | 91.03% |
| Emergency | 230 | 26 | 0.00% |
| Inpatient | 163 | 45 | 71.11% |
| Outpatient | 1,170 | 130 | 59.23% |
| Urgent care | 586 | 155 | 13.55% |
| Wellness | 400 | 15 | 0.00% |

The model performed well for ambulatory encounters but had substantially lower recall for urgent care and outpatient encounters.

It did not identify positive emergency or wellness encounters at the default threshold.

These results demonstrate that strong overall metrics do not guarantee consistent performance across encounter classes.

---

## Feature Importance

Permutation importance using ROC-AUC scoring identified the following leading predictors:

| Feature | Mean importance |
|---|---:|
| `ENCOUNTERCLASS` | 0.02972 |
| `DAYS_SINCE_PREVIOUS_ENCOUNTER` | 0.02076 |
| `PRIOR_ENCOUNTERS_90_DAYS` | 0.01705 |
| `PRIOR_ENCOUNTERS_30_DAYS` | 0.01460 |
| `AGE_AT_ENCOUNTER` | 0.01000 |

Encounter class and historical encounter patterns were the most influential predictors in this analysis.

Permutation importance measures predictive reliance, not causation.

---

## Decision Threshold Analysis

The current inference threshold is:

`0.50`

Threshold analysis using training-set out-of-fold predictions identified **0.32** as a candidate threshold with the highest F1-score.

| Metric | Threshold 0.50 | Threshold 0.32 |
|---|---:|---:|
| Precision | 0.9141 | 0.8048 |
| Recall | 0.6663 | 0.8007 |
| F1-score | 0.7708 | 0.8027 |
| Specificity | 0.9663 | 0.8953 |

The candidate threshold improves recall but increases false positives.

The threshold of 0.32 has not been adopted in the inference script. The appropriate operational threshold would depend on the intended use, the consequences of false positives and false negatives, and the capacity to review alerts.

---

## Repository Structure

```text
analystlab-healthcare-operations-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       └── test/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── models/
│   └── repeat_encounter_7day/
│
├── reports/
│   ├── data_inspection/
│   ├── modeling/
│   ├── evaluation/
│   └── inference/
│
├── figures/
│   ├── eda/
│   └── evaluation/
│
├── documentation/
│   ├── data_preparation_and_feature_engineering.md
│   ├── model_development_and_evaluation.md
│   └── inference_and_monitoring.md
│
├── tests/
│   └── test_inference.py
│
├── README.md
└── .gitignore
```

The exact contents of generated report and figure folders may vary depending on which scripts have been run.

---

## Reproducibility

### Requirements

The project uses Python and common machine learning and data analysis libraries, including:

- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib

Install the project's dependencies using the provided requirements file, if available:

```powershell
python -m pip install -r requirements.txt
```

If the dependency file is not present in a copy of the repository, install the required libraries in a Python environment compatible with the project.

### Data setup

Place the authorized source CSV files in:

```text
data/raw/
```

The raw data is not included in this repository.

### Workflow

The project scripts are organized into stages:

1. Inspect source datasets.
2. Validate dataset relationships.
3. Investigate and construct the target.
4. Audit feature eligibility and leakage.
5. Create the modeling dataset.
6. Prepare the patient-grouped split.
7. Train and compare models.
8. Evaluate fitted models.
9. Perform error analysis.
10. Calculate feature importance.
11. Perform threshold analysis.
12. Run inference.

The scripts are located in the `src/` directory.

Use the numbered scripts in their intended order. The final inference script is:

`src/models/13_inference.py`

---

## Running Inference

The saved Random Forest pipeline is located at:

```text
models/repeat_encounter_7day/random_forest.joblib
```

### Demonstration

```powershell
python src/models/13_inference.py
```

### Batch inference

```powershell
python src/models/13_inference.py --input data/processed/example_inference.csv
```

### Custom output filename

```powershell
python src/models/13_inference.py --input data/processed/example_inference.csv --output batch_predictions.csv
```

Prediction results are saved under:

```text
reports/inference/
```

The input CSV must contain all nine required predictors.

---

## Testing

The inference validation tests are located at:

`tests/test_inference.py`

Run the tests with:

```powershell
python -m unittest discover -s tests -p "test_inference.py" -v
```

The test suite contains seven tests covering:

- Valid input.
- Empty input.
- Missing predictors.
- Invalid encounter classes.
- Malformed numerical values.
- Missing numerical values.
- Prediction output validity.

All seven tests passed during development.

These tests verify the tested input-validation and inference functions. They do not establish clinical validity or predictive accuracy.

---

## Documentation

Additional technical documentation is available in the `documentation/` directory.

| Document | Contents |
|---|---|
| `data_preparation_and_feature_engineering.md` | Source data, target construction, feature engineering, leakage prevention and split preparation |
| `model_development_and_evaluation.md` | Model comparison, evaluation metrics, error analysis, feature importance and threshold analysis |
| `inference_and_monitoring.md` | Inference instructions, input validation, tests, monitoring and responsible use |

---

## Limitations

This project has several important limitations:

- The dataset represents one organization.
- The data is historical and covers 2011–2022.
- The target represents a subsequent recorded encounter, not a clinically validated readmission outcome.
- Patient-grouped validation evaluates unseen-patient generalization, not strict future-time performance.
- The held-out test set was inspected during model comparison and error analysis.
- Model performance varies substantially by encounter class.
- The model has not been externally validated.
- The model has not been deployed in a healthcare environment.
- No clinical benefit or clinical validity has been established.

The model should not be used to diagnose medical conditions, recommend treatment, or independently determine patient care.

Any future operational use would require appropriate privacy and data governance review, external validation, monitoring, and oversight by qualified professionals.

---

## Author and Project Context

Developed as part of the AnalystLab Africa Healthcare Analytics and AI Implementation assignment, Machine Learning Track.

This project demonstrates the application of machine learning and healthcare data analytics to an encounter-level prediction problem, with emphasis on reproducibility, leakage prevention, evaluation, and responsible use.