# Data Preparation and Feature Engineering

## 1. Overview

This document describes the data preparation, target construction, feature engineering, and leakage prevention procedures used in the Healthcare Analytics and AI Implementation project.

The objective of the machine learning component is to predict whether a patient will have a subsequent healthcare encounter within seven days after the current encounter ends.

The project uses encounter-level data. Each row in the final modeling dataset represents one eligible healthcare encounter.

The modeling workflow was designed to:

- Validate the structure and quality of the source datasets.
- Establish relationships between encounters, patients, and procedures.
- Define a prediction target with a clear time window.
- Exclude encounters without sufficient follow-up observation.
- Construct predictors using information available by the prediction point.
- Prevent patient identifiers and future information from entering the model.
- Produce a reproducible dataset for model training and evaluation.

---

## 2. Source Datasets

The project uses the following CSV files:

| Dataset | Description |
|---|---|
| `encounters.csv` | Healthcare encounter records, including encounter start and stop timestamps, encounter class, and cost-related fields. |
| `patients.csv` | Patient-level demographic and administrative information. |
| `procedures.csv` | Procedures associated with healthcare encounters. |
| `organizations.csv` | Healthcare organization information. |
| `payers.csv` | Payer information. |
| `data_dictionary.csv` | Dataset field descriptions and metadata. |

The source data covers encounters recorded between January 2011 and February 2022.

The latest encounter START timestamp observed in the dataset was:

`2022-02-05 20:27:36+00:00`

The dataset contains personal and identifying information. Such information was excluded from the model predictors and must not be included in public project files.

---

## 3. Data Quality and Relationship Validation

Initial data inspection was performed before feature engineering.

The main datasets contained:

| Dataset | Records |
|---|---:|
| Encounters | 27,891 |
| Patients | 974 |
| Payers | 10 |
| Organizations | 1 |
| Procedures | 47,701 |

### 3.1 Identifier checks

The encounter and patient datasets contained unique primary identifiers.

Relationship validation confirmed that:

- Encounter patient identifiers matched valid patient records.
- Procedure encounter identifiers matched valid encounter records.
- No invalid encounter-to-patient references were identified.
- No invalid procedure-to-encounter references were identified.

Procedures were linked to 14,670 unique encounters. A total of 13,221 encounters had no recorded procedures.

The absence of a procedure record was not interpreted as proof that no healthcare service occurred. It only indicates that no associated procedure was recorded in the procedure dataset.

### 3.2 Encounter timestamp checks

The encounter START and STOP timestamps were inspected for missing values and invalid durations.

Results:

- Missing START timestamps: 0
- Missing STOP timestamps: 0
- STOP timestamps earlier than START timestamps: 0
- Encounters with duration greater than 24 hours: 75

A maximum encounter duration of approximately 44,930 hours was observed. This extreme value was retained during the initial data investigation and was not used as a basis for claiming that all encounter durations represent typical clinical activity.

### 3.3 Cost-related fields

Cost variables were inspected for missing, negative, and zero values.

`BASE_ENCOUNTER_COST` had no missing or negative values.

`TOTAL_CLAIM_COST` was highly skewed and contained 89 zero values.

`PAYER_COVERAGE` contained 13,586 zero values, approximately 48.7% of encounters.

Because the prediction target concerns subsequent encounters rather than financial outcomes, cost-related fields were not selected as predictors.

### 3.4 Missing information

Encounter reason codes and descriptions were frequently missing.

Approximately:

- 70% of encounter reason fields were missing.
- 77% of procedure reason fields were missing.

These fields were not selected for the primary model.

---

## 4. Prediction Target Definition

### 4.1 Target

The target variable is:

`REPEAT_WITHIN_7_DAYS`

It is a binary classification target.

| Target value | Definition |
|---|---|
| 1 | A subsequent encounter START is observed within seven days after the current encounter STOP. |
| 0 | No subsequent encounter START is observed within seven days after the current encounter STOP, provided the full observation window is available. |

The target is defined using the current encounter's STOP timestamp rather than its START timestamp.

This reflects the project's prediction point: the end of the current encounter.

### 4.2 Subsequent encounter identification

Encounters were ordered chronologically for each patient.

For each encounter, the next encounter START timestamp occurring after the current encounter STOP timestamp was identified.

The time difference was calculated as:

`NEXT_START_AFTER_STOP - CURRENT_ENCOUNTER_STOP`

If the difference was greater than zero and less than or equal to seven days, the encounter was assigned a positive target.

The target does not represent a clinically validated readmission outcome. It represents a subsequent encounter recorded in the available dataset.

### 4.3 Prediction point

The prediction point is:

**At the end of the current encounter, after the encounter STOP timestamp.**

The model is intended to estimate the probability of a subsequent encounter occurring within the following seven days.

Features that depend on information recorded after this prediction point were excluded.

---

## 5. Observation Window and Censoring

A negative target requires sufficient observation time.

For example, if an encounter ends only two days before the dataset's latest observable encounter START timestamp, the absence of a subsequent encounter during those two days cannot establish that no subsequent encounter occurred within seven days.

To avoid incorrectly labeling such records as negative, an encounter was eligible only when:

`CURRENT_ENCOUNTER_STOP + 7 days <= OBSERVATION_END`

The observation boundary was defined as the maximum encounter START timestamp in the dataset.

This conservative boundary was selected because the target event is a subsequent encounter START. Using the maximum encounter STOP timestamp could extend the assumed observation period beyond the latest observable encounter START.

This is a dataset-based assumption. It does not guarantee that every encounter occurring in the real world was captured.

### 5.1 Target construction results

| Measure | Result |
|---|---:|
| Total encounters | 27,891 |
| Eligible encounters | 27,887 |
| Excluded encounters | 4 |
| Positive cases | 9,766 |
| Negative cases | 18,121 |
| Positive rate | 35.02% |

The four excluded encounters did not have a complete seven-day observation window.

The final modeling dataset contains 27,887 eligible encounters.

---

## 6. Target Distribution by Encounter Class

The target distribution varies considerably across encounter classes.

| Encounter class | Eligible encounters | Positive cases | Positive rate |
|---|---:|---:|---:|
| Ambulatory | 12,536 | 6,441 | 51.38% |
| Emergency | 2,322 | 871 | 37.51% |
| Inpatient | 1,135 | 273 | 24.05% |
| Outpatient | 6,300 | 557 | 8.84% |
| Urgent care | 3,665 | 1,562 | 42.62% |
| Wellness | 1,929 | 62 | 3.21% |

These differences indicate that the target is not distributed uniformly across encounter classes.

Encounter class was therefore retained as a predictor, while class-specific model performance was examined during evaluation.

---

## 7. Feature Engineering

Nine predictors were selected for the primary model.

| Feature | Description |
|---|---|
| `ENCOUNTERCLASS` | Type or class of the current encounter. |
| `DURATION_HOURS` | Duration of the current encounter in hours. |
| `AGE_AT_ENCOUNTER` | Patient age at the time of the current encounter. |
| `PRIOR_ENCOUNTERS_30_DAYS` | Number of previous encounters within the 30 days before the current encounter. |
| `PRIOR_ENCOUNTERS_90_DAYS` | Number of previous encounters within the 90 days before the current encounter. |
| `DAYS_SINCE_PREVIOUS_ENCOUNTER` | Number of days between the previous encounter and the current encounter. |
| `START_DAY_OF_WEEK` | Day of the week on which the current encounter started. |
| `START_MONTH` | Month in which the current encounter started. |
| `START_HOUR` | Hour of the day at which the current encounter started. |

### 7.1 Historical encounter features

The historical encounter features were constructed using encounters occurring before the current encounter.

The 30-day and 90-day encounter counts summarize recent encounter frequency.

`DAYS_SINCE_PREVIOUS_ENCOUNTER` measures the interval between the previous encounter and the current encounter.

These features were designed to capture historical utilization patterns without using future encounters.

### 7.2 Encounter duration

Encounter duration was calculated from the current encounter's START and STOP timestamps.

Because the prediction point is the end of the current encounter, its duration is available at the defined prediction point.

### 7.3 Calendar features

The following calendar features were derived from the current encounter START timestamp:

- Day of week
- Month
- Hour of day

The timestamps were processed using their recorded UTC representation.

These features describe the timing of the encounter and do not use future event information.

---

## 8. Feature Eligibility and Leakage Prevention

Feature selection was based on whether the information could reasonably be available at the prediction point.

The following information was excluded from the model predictors.

### 8.1 Identifiers

The following identifiers were retained for data management and splitting purposes but excluded from the predictor matrix:

- Encounter ID (`Id`)
- Patient ID (`PATIENT`)

Patient identifiers were required to perform patient-grouped validation and to verify that the same patient did not appear in both training and testing datasets.

They were not used as model inputs.

### 8.2 Future information

The following fields were excluded because they directly describe the target event or depend on future information:

- `NEXT_START_AFTER_STOP`
- `DAYS_TO_NEXT_AFTER_STOP`
- `REPEAT_WITHIN_7_DAYS`

The target variable was never included among the predictors.

### 8.3 Financial and administrative fields

Cost-related fields, payer information, and encounter reason codes were not selected for the primary model.

Their availability at the prediction point was not sufficiently established for this project, and some fields may be recorded or finalized after the encounter.

### 8.4 Sensitive and identifying information

Patient names, addresses, and other identifying fields were excluded.

Demographic and administrative fields were reviewed during the feature eligibility audit. The primary model retained age but excluded other patient demographic and location fields that were not justified for the initial modeling objective.

### 8.5 Organization information

The dataset contained one organization.

Organization identity therefore provided no useful variation for the primary model and was excluded.

---

## 9. Missing Value Handling

The final selected predictors contained missing values only in:

`DAYS_SINCE_PREVIOUS_ENCOUNTER`

Missingness:

| Feature | Missing values |
|---|---:|
| `DAYS_SINCE_PREVIOUS_ENCOUNTER` | 974 |
| All other selected predictors | 0 |

The 974 missing values correspond to approximately 3.49% of the modeling dataset.

These values can occur when a patient has no previous encounter available in the dataset.

Missing numerical values were handled using median imputation within the model preprocessing pipeline.

The imputer was fitted using training data only. The same fitted preprocessing pipeline was then applied to validation and testing data.

This prevents information from the testing dataset from influencing the imputation values.

---

## 10. Modeling Dataset

The final modeling dataset was saved to:

`data/processed/repeat_encounter_7day_modeling_dataset.csv`

The dataset contains:

- Encounter ID
- Patient ID
- Nine selected predictors
- Binary target

Final shape:

`27,887 rows × 12 columns`

The modeling dataset was checked for:

- Duplicate encounter identifiers
- Missing target values
- Invalid target values
- Target values outside the binary 0/1 range
- Missingness in selected predictors
- Inclusion of target or patient identifier in the predictor list
- Potential use of future information in historical features

The final dataset passed the documented leakage checks.

---

## 11. Patient-Grouped Train-Test Split

To reduce the risk of patient-level information leakage, the dataset was divided using:

`StratifiedGroupKFold`

Configuration:

- Number of folds: 5
- Shuffle: True
- Random state: 42
- Group variable: Patient ID

The first fold was used as the held-out test set.

### 11.1 Split results

| Measure | Training set | Testing set |
|---|---:|---:|
| Encounters | 22,305 | 5,582 |
| Patients | 778 | 196 |
| Positive cases | 7,812 | 1,954 |
| Positive rate | 35.02% | 35.01% |

No patients appeared in both training and testing datasets.

No encounters appeared in both datasets.

This split evaluates model performance on patients not seen during training.

It does not constitute a strict temporal validation. The project does not claim that the model has been validated on future calendar periods.

---

## 12. Reproducibility

The data preparation and feature engineering workflow is implemented through project scripts under:

- `src/data/`
- `src/features/`
- `src/models/`

The scripts generate the processed modeling dataset, feature audit reports, and patient-grouped training and testing datasets.

The project uses a fixed random state of 42 for the documented patient-grouped split and model evaluation procedures.

The raw datasets are not included in the public repository because of their size and the presence of sensitive information.

Users attempting to reproduce the workflow must obtain the source data from an authorized source and place it in the documented project data directory.

---

## 13. Limitations

The data preparation process has several limitations:

1. The dataset represents records from a single organization.
2. The data covers historical encounters from 2011 to 2022.
3. The observation boundary is based on the latest recorded encounter START timestamp.
4. The dataset may not capture every subsequent encounter that occurred in the real world.
5. The target represents a subsequent recorded encounter, not a clinically validated readmission outcome.
6. Patient-grouped validation evaluates performance on unseen patients but does not establish future-period generalization.
7. The selected features are based on the available dataset and the project's defined prediction point.

The model is intended for educational and analytical purposes. It should not be used to make diagnoses, recommend treatment, or independently determine clinical decisions.