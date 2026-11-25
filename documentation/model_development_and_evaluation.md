# Model Development and Evaluation

## 1. Overview

This document describes the machine learning model development and evaluation process for the Healthcare Analytics and AI Implementation project.

The objective was to develop a classification model that estimates the probability of a subsequent healthcare encounter being recorded within seven days after the current encounter ends.

The modeling workflow included:

- Patient-grouped training and testing.
- Baseline model development.
- Comparison of multiple classification algorithms.
- Cross-validation using training data.
- Final evaluation on a held-out test set.
- Error analysis.
- Feature importance analysis.
- Decision threshold analysis.

The selected model is intended for analytical and educational purposes. It is not a clinically validated prediction system.

---

## 2. Modeling Dataset

The modeling dataset was created through the data preparation and feature engineering workflow.

Target:

`REPEAT_WITHIN_7_DAYS`

Predictors:

1. `ENCOUNTERCLASS`
2. `DURATION_HOURS`
3. `AGE_AT_ENCOUNTER`
4. `PRIOR_ENCOUNTERS_30_DAYS`
5. `PRIOR_ENCOUNTERS_90_DAYS`
6. `DAYS_SINCE_PREVIOUS_ENCOUNTER`
7. `START_DAY_OF_WEEK`
8. `START_MONTH`
9. `START_HOUR`

The dataset contained 27,887 eligible encounters.

The target distribution was:

| Target | Encounters |
|---|---:|
| 0 | 18,121 |
| 1 | 9,766 |

The positive class represented approximately 35.02% of eligible encounters.

Encounter ID and patient ID were retained for traceability and grouped splitting but were excluded from model predictors.

---

## 3. Train-Test Split and Validation Strategy

A patient-grouped split was used to reduce the risk of the model learning patient-specific patterns that would appear in both training and testing data.

The split was created using:

`StratifiedGroupKFold`

Configuration:

- Number of folds: 5
- Shuffle: True
- Random state: 42
- Group variable: Patient ID

The first fold was selected as the held-out test set.

### 3.1 Split summary

| Measure | Training | Testing |
|---|---:|---:|
| Encounters | 22,305 | 5,582 |
| Patients | 778 | 196 |
| Positive cases | 7,812 | 1,954 |
| Negative cases | 14,493 | 3,628 |
| Positive rate | 35.02% | 35.01% |

Patient separation was verified.

No patients appeared in both training and testing datasets.

### 3.2 Cross-validation

Model comparison was performed using five-fold stratified group cross-validation on the training data.

The patient ID was used as the grouping variable so that encounters belonging to the same patient remained within the same fold.

The held-out test set was not used to fit the models during cross-validation.

This strategy evaluates generalization to unseen patients. It does not represent strict future-time validation.

---

## 4. Preprocessing Pipeline

Preprocessing was performed within each model pipeline to ensure that transformations were fitted using training data only.

### 4.1 Numerical predictors

Numerical features were processed using:

1. Median imputation.
2. Standardization using `StandardScaler`.

Median imputation was used to handle missing values in `DAYS_SINCE_PREVIOUS_ENCOUNTER`.

Standardization was included because Logistic Regression and the RBF Support Vector Machine are sensitive to feature scale.

### 4.2 Categorical predictor

`ENCOUNTERCLASS` was processed using:

1. Most-frequent-value imputation.
2. One-hot encoding.

The encoder used:

- `handle_unknown="ignore"`
- `min_frequency=10`

This allowed the pipeline to handle previously unseen categories without failing during prediction.

### 4.3 Pipeline design

The preprocessing steps were combined with each classifier in a single pipeline.

This ensured that:

- Imputation was fitted only on training data.
- Scaling was fitted only on training data.
- Categorical encoding was fitted only on training data.
- The same transformations were applied consistently during evaluation and inference.

---

## 5. Models Compared

Four classification approaches were evaluated.

### 5.1 Dummy Classifier

The Dummy Classifier used the most-frequent-class strategy.

It served as a simple baseline to demonstrate the performance of a model that does not learn meaningful relationships from the predictors.

### 5.2 Logistic Regression

Logistic Regression was used as a linear classification baseline.

Configuration:

`max_iter=2000`

It provided a relatively simple model against which more complex algorithms could be compared.

### 5.3 Random Forest

Random Forest was selected as a nonlinear ensemble model.

Configuration:

- Number of trees: 300
- Minimum samples per leaf: 2
- Random state: 42
- Number of jobs: 1

Random Forest can model nonlinear relationships and interactions between predictors without requiring the same feature scaling assumptions as linear models.

### 5.4 RBF Support Vector Machine

An RBF Support Vector Machine was included to evaluate a nonlinear margin-based classification approach.

Configuration:

- Kernel: RBF
- C: 1.0
- Gamma: scale
- Class weight: balanced
- Random state: 42

The class-weight setting was used to account for the difference in class frequencies during model fitting.

---

## 6. Cross-Validation Results

The following table summarizes the five-fold cross-validation results on the training dataset.

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Dummy Classifier | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.8947 | 0.6669 | 0.7637 | 0.9010 |
| Random Forest | 0.9131 | 0.6660 | 0.7646 | 0.9182 |
| RBF SVM | 0.8322 | 0.6840 | 0.7468 | 0.8865 |

Random Forest achieved the highest mean F1-score and ROC-AUC during cross-validation.

Logistic Regression produced a similar F1-score, while the RBF SVM achieved slightly higher recall but lower precision and overall F1-score.

The Dummy Classifier predicted the majority class and did not identify positive cases. Its accuracy would be approximately 65%, but its zero recall demonstrates why accuracy alone is insufficient for this task.

---

## 7. Final Test-Set Evaluation

The fitted models were evaluated on the held-out test dataset containing 5,582 encounters from 196 patients.

The test set contained:

- 1,954 positive cases.
- 3,628 negative cases.
- Positive rate: 35.01%.

### 7.1 Test results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.6499 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3501 |
| Logistic Regression | 0.9061 | 0.9503 | 0.7723 | 0.8521 | 0.9302 | 0.9227 |
| Random Forest | 0.9233 | 0.9722 | 0.8040 | 0.8801 | 0.9454 | 0.9402 |
| RBF SVM | 0.8956 | 0.9513 | 0.7395 | 0.8321 | 0.9169 | 0.9044 |

Random Forest achieved the highest reported test-set performance across the listed metrics.

### 7.2 Random Forest performance

The Random Forest model achieved:

- Accuracy: 92.33%
- Precision: 97.22%
- Recall: 80.40%
- F1-score: 88.01%
- ROC-AUC: 94.54%
- Average Precision: 94.02%

At the default decision threshold of 0.50, the model identified 1,571 of the 1,954 positive test encounters.

It missed 383 positive cases and incorrectly classified 45 negative cases as positive.

### 7.3 Confusion matrix

The Random Forest confusion matrix was:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 3,583 | 45 |
| Actual positive | 383 | 1,571 |

The model showed high precision, meaning that most encounters predicted as positive were positive in the test dataset.

However, it did not identify every positive case.

---

## 8. Cross-Validation and Test-Set Comparison

The Random Forest cross-validation and test-set results were:

| Metric | Cross-validation | Test set |
|---|---:|---:|
| Precision | 0.9131 | 0.9722 |
| Recall | 0.6660 | 0.8040 |
| F1-score | 0.7646 | 0.8801 |
| ROC-AUC | 0.9182 | 0.9454 |

The test-set results were higher than the cross-validation averages.

This difference should not automatically be interpreted as evidence that the model has no overfitting.

Possible explanations include differences in patient composition, encounter-class distribution, and variation between patient-grouped folds.

The difference should be reported transparently.

The test set has been inspected during model comparison and error analysis. Therefore, it should not be treated as a completely untouched external validation dataset.

No threshold tuning was performed using the test set.

---

## 9. Error Analysis

Error analysis was performed on the Random Forest test-set predictions using the default threshold of 0.50.

### 9.1 Overall errors

| Measure | Result |
|---|---:|
| True positives | 1,571 |
| True negatives | 3,583 |
| False positives | 45 |
| False negatives | 383 |
| False positive rate | 1.24% |
| False negative rate | 19.60% |

The model produced relatively few false positives but missed approximately one in five positive encounters.

### 9.2 Performance by encounter class

| Encounter class | Test encounters | Positive cases | Recall | False negatives |
|---|---:|---:|---:|---:|
| Ambulatory | 3,033 | 1,583 | 91.03% | 142 |
| Emergency | 230 | 26 | 0.00% | 26 |
| Inpatient | 163 | 45 | 71.11% | 13 |
| Outpatient | 1,170 | 130 | 59.23% | 53 |
| Urgent care | 586 | 155 | 13.55% | 134 |
| Wellness | 400 | 15 | 0.00% | 15 |

The model's performance varied considerably by encounter class.

It performed particularly well for ambulatory encounters.

Recall was substantially lower for urgent care and outpatient encounters. The model did not identify any positive emergency or wellness encounters at the default threshold.

These findings indicate that strong overall performance does not imply equally strong performance across all encounter classes.

### 9.3 False negative distribution

The 383 false negatives were distributed as follows:

| Encounter class | False negatives |
|---|---:|
| Ambulatory | 142 |
| Urgent care | 134 |
| Outpatient | 53 |
| Emergency | 26 |
| Wellness | 15 |
| Inpatient | 13 |

Urgent care accounted for a substantial number of missed positive cases.

This is an important limitation when interpreting the model's overall recall.

### 9.4 Prediction confidence

The mean predicted probabilities for the main outcome groups were:

| Outcome group | Mean predicted probability |
|---|---:|
| True negatives | 0.1198 |
| True positives | 0.9593 |
| False positives | 0.6355 |
| False negatives | 0.2217 |

Some false negatives had probabilities close to the 0.50 decision threshold, while others received much lower probabilities.

This suggests that changing the threshold could recover some missed cases, but it would also increase the number of false positives.

---

## 10. Feature Importance

Permutation importance was calculated using the held-out test dataset and ROC-AUC scoring.

The baseline Random Forest test ROC-AUC was:

`0.9454`

The importance score represents the decrease in ROC-AUC when a feature is randomly permuted.

### 10.1 Permutation importance results

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| `ENCOUNTERCLASS` | 0.02972 | 0.00316 |
| `DAYS_SINCE_PREVIOUS_ENCOUNTER` | 0.02076 | 0.00095 |
| `PRIOR_ENCOUNTERS_90_DAYS` | 0.01705 | 0.00207 |
| `PRIOR_ENCOUNTERS_30_DAYS` | 0.01460 | 0.00121 |
| `AGE_AT_ENCOUNTER` | 0.01000 | 0.00186 |
| `DURATION_HOURS` | 0.00432 | 0.00042 |
| `START_DAY_OF_WEEK` | 0.00094 | 0.00050 |
| `START_MONTH` | 0.00093 | 0.00083 |
| `START_HOUR` | 0.00061 | 0.00075 |

### 10.2 Interpretation

Encounter class was the most important predictor according to permutation importance.

Historical encounter features were also influential, particularly the time since the previous encounter and the number of encounters in the previous 90 and 30 days.

Age had a moderate contribution.

Calendar features had comparatively low importance in this evaluation.

Permutation importance measures predictive reliance, not causation.

Correlated predictors may share or redistribute importance, so the results should not be interpreted as proof that one feature causes a subsequent encounter.

---

## 11. Decision Threshold Analysis

The default classification threshold was:

`0.50`

A prediction was classified as positive when:

`Predicted probability >= 0.50`

Threshold analysis was conducted using out-of-fold predictions generated from the training dataset.

The analysis used five-fold stratified group cross-validation and did not use the held-out test data to select the threshold.

### 11.1 Out-of-fold performance

The out-of-fold predictions achieved:

- ROC-AUC: 0.9186
- Average Precision: 0.8972

At threshold 0.50, the out-of-fold results were:

| Metric | Value |
|---|---:|
| Precision | 0.9141 |
| Recall | 0.6663 |
| F1-score | 0.7708 |
| Specificity | 0.9663 |

These results were broadly consistent with the Random Forest cross-validation results.

### 11.2 Candidate threshold

The highest F1-score in the threshold analysis occurred at:

`0.32`

The corresponding out-of-fold metrics were:

| Metric | Threshold 0.32 |
|---|---:|
| Precision | 0.8048 |
| Recall | 0.8007 |
| F1-score | 0.8027 |
| Specificity | 0.8953 |
| False positives | 1,517 |
| False negatives | 1,557 |

Compared with threshold 0.50, threshold 0.32 improved recall by approximately 13.44 percentage points.

However, it also reduced precision and increased the number of false positives.

### 11.3 Threshold decision

The threshold of 0.32 is documented as a candidate rather than the current inference setting.

The inference script continues to use threshold 0.50.

A final operational threshold would require consideration of:

- The consequences of false negatives.
- The consequences of false positives.
- The number of alerts that staff could review.
- The intended use of the prediction.
- Whether the model's probabilities are sufficiently calibrated for the intended decision.

The highest F1-score alone does not determine the most appropriate operational threshold.

---

## 12. Selected Model

Random Forest was selected as the final model because it achieved the strongest reported overall performance among the evaluated models.

The fitted pipeline was saved to:

`models/repeat_encounter_7day/random_forest.joblib`

The saved pipeline includes preprocessing and the fitted classifier.

It is used by the inference script to generate probabilities and binary predictions for new input records.

---

## 13. Evaluation Artifacts

The modeling and evaluation workflow generated the following outputs.

### Model artifacts

- `models/repeat_encounter_7day/dummy_classifier.joblib`
- `models/repeat_encounter_7day/logistic_regression.joblib`
- `models/repeat_encounter_7day/random_forest.joblib`
- `models/repeat_encounter_7day/rbf_svm.joblib`

### Evaluation reports

- Cross-validation model comparison results.
- Final test-set evaluation metrics.
- Confusion matrices.
- Error analysis reports.
- Class-specific performance reports.
- Permutation importance results.
- Out-of-fold threshold metrics.

### Evaluation figures

- Model evaluation plots.
- Random Forest permutation importance figure.
- Random Forest threshold analysis figure.

The reports and figures are stored under the project's `reports/` and `figures/` directories.

---

## 14. Limitations

The model evaluation has several limitations:

1. The data represents one organization.
2. The dataset is historical and covers 2011–2022.
3. Patient-grouped validation evaluates unseen-patient generalization, not strict future-time performance.
4. The test set has been inspected during model comparison and error analysis.
5. Performance varies substantially across encounter classes.
6. The target represents a subsequent recorded encounter, not a clinically validated readmission outcome.
7. The model has not been externally validated.
8. No clinical deployment or clinical benefit has been established.
9. Threshold selection remains dependent on the intended operational use.

The results demonstrate the feasibility of building a reproducible encounter-level classification workflow using the available data.

They do not establish clinical validity or suitability for autonomous healthcare decision-making.