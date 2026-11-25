from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENCOUNTERS_PATH = PROJECT_ROOT / "data" / "raw" / "encounters.csv"
PATIENTS_PATH = PROJECT_ROOT / "data" / "raw" / "patients.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PREDICTION CONTEXT
# ============================================================

PREDICTION_POINT = (
    "End of the current encounter, immediately after STOP."
)

PREDICTION_OBJECTIVE = (
    "Predict whether the patient will have another encounter "
    "within seven days after the current encounter ends."
)


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print("Loading datasets...")

encounters = pd.read_csv(ENCOUNTERS_PATH)
patients = pd.read_csv(PATIENTS_PATH)

print(f"Encounters: {encounters.shape}")
print(f"Patients: {patients.shape}")


# ============================================================
# 4. DEFINE FEATURE ELIGIBILITY DECISIONS
# ============================================================

# Each entry documents:
# - eligibility decision
# - availability at prediction time
# - leakage risk
# - reason for the decision
# - action required before modelling

feature_decisions = {

    # --------------------------------------------------------
    # ENCOUNTER DATASET
    # --------------------------------------------------------

    "Id": {
        "decision": "Exclude",
        "availability": "Available",
        "leakage_risk": "High",
        "reason": "Unique encounter identifier. It is not a meaningful predictor.",
        "action": "Retain only for record identification."
    },

    "START": {
        "decision": "Eligible after transformation",
        "availability": "Known at encounter start",
        "leakage_risk": "Low",
        "reason": "Can be transformed into calendar features such as weekday, month and hour.",
        "action": "Use derived calendar features rather than the raw timestamp."
    },

    "STOP": {
        "decision": "Eligible for feature derivation only",
        "availability": "Known at prediction time",
        "leakage_risk": "High if used directly",
        "reason": "Defines the prediction point and can be used with START to calculate encounter duration. The raw timestamp should not be used directly.",
        "action": "Derive duration and other justified features; exclude raw STOP."
    },

    "PATIENT": {
        "decision": "Exclude",
        "availability": "Available",
        "leakage_risk": "High",
        "reason": "Patient identifier. Using it could encourage memorisation of individual patients.",
        "action": "Retain for historical feature construction and patient-aware data splitting only."
    },

    "ORGANIZATION": {
        "decision": "Conditional",
        "availability": "Potentially available",
        "leakage_risk": "Uncertain",
        "reason": "May identify the healthcare organization responsible for the encounter. Availability and generalisation implications require review.",
        "action": "Verify prediction-time availability and assess whether the model should generalise across organizations."
    },

    "PROVIDER": {
        "decision": "Conditional",
        "availability": "Potentially available",
        "leakage_risk": "Uncertain",
        "reason": "Identifies the provider associated with the encounter. May introduce high-cardinality and provider-specific effects.",
        "action": "Verify availability and assess whether provider-level information is appropriate."
    },

    "PAYER": {
        "decision": "Conditional",
        "availability": "Potentially available",
        "leakage_risk": "Uncertain",
        "reason": "Insurance payer information may be available by encounter end, but this must be verified.",
        "action": "Confirm availability and assess whether its use is appropriate."
    },

    "ENCOUNTERCLASS": {
        "decision": "Eligible",
        "availability": "Known during encounter",
        "leakage_risk": "Low",
        "reason": "Describes the type of current encounter.",
        "action": "Can be considered for the primary model."
    },

    "CODE": {
        "decision": "Conditional",
        "availability": "Requires verification",
        "leakage_risk": "Uncertain",
        "reason": "The code's meaning and the time at which it is assigned must be established.",
        "action": "Inspect the data dictionary and confirm it is available by encounter end."
    },

    "DESCRIPTION": {
        "decision": "Conditional",
        "availability": "Requires verification",
        "leakage_risk": "Uncertain",
        "reason": "The description may contain information about the encounter that is not necessarily available at the prediction point.",
        "action": "Inspect its meaning and timing before considering inclusion."
    },

    "BASE_ENCOUNTER_COST": {
        "decision": "Conditional",
        "availability": "Requires verification",
        "leakage_risk": "Uncertain",
        "reason": "An encounter-level base cost may be available at encounter end, but its calculation and timing have not been established.",
        "action": "Verify when this value becomes available before considering inclusion."
    },

    "TOTAL_CLAIM_COST": {
        "decision": "Exclude",
        "availability": "Potentially available after encounter",
        "leakage_risk": "High",
        "reason": "Claim totals may be finalised after the encounter and therefore may not be available at prediction time.",
        "action": "Exclude from the primary model."
    },

    "PAYER_COVERAGE": {
        "decision": "Exclude",
        "availability": "Potentially available after encounter",
        "leakage_risk": "High",
        "reason": "Coverage amounts may depend on subsequent claim processing.",
        "action": "Exclude from the primary model."
    },

    "REASONCODE": {
        "decision": "Conditional",
        "availability": "Requires verification",
        "leakage_risk": "Uncertain",
        "reason": "The timing and meaning of reason coding have not been established.",
        "action": "Inspect the data dictionary and verify availability at encounter end."
    },

    "REASONDESCRIPTION": {
        "decision": "Conditional",
        "availability": "Requires verification",
        "leakage_risk": "Uncertain",
        "reason": "The description may reflect information entered or finalised after the encounter.",
        "action": "Inspect its meaning and timing before considering inclusion."
    },

    # --------------------------------------------------------
    # PATIENT DATASET
    # --------------------------------------------------------

    "BIRTHDATE": {
        "decision": "Eligible for feature derivation only",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "Can be used with encounter START to calculate age at the time of the encounter.",
        "action": "Derive AGE_AT_ENCOUNTER and exclude raw BIRTHDATE."
    },

    "GENDER": {
        "decision": "Conditional",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "May be available at prediction time, but its use requires a responsible-use and fairness assessment.",
        "action": "Evaluate whether inclusion is justified and assess performance across relevant groups."
    },

    "RACE": {
        "decision": "Exclude from primary model",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "Sensitive demographic attribute. Its inclusion requires a strong justification and careful fairness assessment.",
        "action": "Do not use as a primary predictor. Consider appropriate fairness auditing where permitted."
    },

    "ETHNICITY": {
        "decision": "Exclude from primary model",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "Sensitive demographic attribute. Its inclusion requires a strong justification and careful fairness assessment.",
        "action": "Do not use as a primary predictor. Consider appropriate fairness auditing where permitted."
    },

    "DEATHDATE": {
        "decision": "Exclude",
        "availability": "Potentially recorded after encounter",
        "leakage_risk": "Very high",
        "reason": "May contain future information relative to the prediction point.",
        "action": "Exclude from the primary model."
    },

    "Id": {
        "decision": "Exclude",
        "availability": "Available",
        "leakage_risk": "High",
        "reason": "Unique patient identifier. It is not a meaningful predictor.",
        "action": "Retain only for joining datasets and patient-aware splitting."
    },

    "SSN": {
        "decision": "Exclude",
        "availability": "Sensitive identifier",
        "leakage_risk": "High",
        "reason": "Personally identifiable information with no appropriate predictive role.",
        "action": "Exclude from modelling and public reports."
    },

    "DRIVERS": {
        "decision": "Exclude",
        "availability": "Sensitive identifier",
        "leakage_risk": "High",
        "reason": "Personally identifiable information.",
        "action": "Exclude from modelling and public reports."
    },

    "PASSPORT": {
        "decision": "Exclude",
        "availability": "Sensitive identifier",
        "leakage_risk": "High",
        "reason": "Personally identifiable information.",
        "action": "Exclude from modelling and public reports."
    },

    "PREFIX": {
        "decision": "Exclude",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "Title or prefix is not necessary for the operational prediction objective.",
        "action": "Exclude from the primary model."
    },

    "FIRST": {
        "decision": "Exclude",
        "availability": "Personal information",
        "leakage_risk": "High",
        "reason": "Patient first name is personally identifiable information.",
        "action": "Exclude from modelling and public reports."
    },

    "LAST": {
        "decision": "Exclude",
        "availability": "Personal information",
        "leakage_risk": "High",
        "reason": "Patient surname is personally identifiable information.",
        "action": "Exclude from modelling and public reports."
    },

    "MAIDEN": {
        "decision": "Exclude",
        "availability": "Personal information",
        "leakage_risk": "High",
        "reason": "Personally identifiable information.",
        "action": "Exclude from modelling and public reports."
    },

    "SUFFIX": {
        "decision": "Exclude",
        "availability": "Personal information",
        "leakage_risk": "Low",
        "reason": "Not necessary for the operational prediction objective.",
        "action": "Exclude from the primary model."
    },

    "MARITAL": {
        "decision": "Conditional",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "May be available at prediction time, but its operational relevance and fairness implications require review.",
        "action": "Consider only if there is a defensible justification."
    },

    "BIRTHPLACE": {
        "decision": "Exclude from primary model",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "Potentially sensitive demographic information with no established operational justification.",
        "action": "Exclude from the primary model."
    },

    "ADDRESS": {
        "decision": "Exclude",
        "availability": "Personal information",
        "leakage_risk": "High",
        "reason": "Personally identifiable location information.",
        "action": "Exclude from modelling and public reports."
    },

    "CITY": {
        "decision": "Conditional",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "May provide geographic context but could introduce privacy and fairness concerns.",
        "action": "Consider only if geographically relevant and appropriately justified."
    },

    "STATE": {
        "decision": "Conditional",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "May provide geographic context but could introduce privacy and fairness concerns.",
        "action": "Consider only if geographically relevant and appropriately justified."
    },

    "COUNTY": {
        "decision": "Conditional",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "May provide geographic context but could introduce privacy and fairness concerns.",
        "action": "Consider only if geographically relevant and appropriately justified."
    },

    "ZIP": {
        "decision": "Exclude from primary model",
        "availability": "Patient demographic information",
        "leakage_risk": "Low",
        "reason": "Detailed geographic information can create privacy and fairness concerns.",
        "action": "Exclude from the primary model."
    },

    "LAT": {
        "decision": "Exclude from primary model",
        "availability": "Patient location information",
        "leakage_risk": "Low",
        "reason": "Precise geographic information introduces privacy concerns.",
        "action": "Exclude from the primary model."
    },

    "LON": {
        "decision": "Exclude from primary model",
        "availability": "Patient location information",
        "leakage_risk": "Low",
        "reason": "Precise geographic information introduces privacy concerns.",
        "action": "Exclude from the primary model."
    },

    "HEALTHCARE_EXPENSES": {
        "decision": "Exclude",
        "availability": "Potentially updated after encounters",
        "leakage_risk": "High",
        "reason": "May include accumulated financial information that is not available at the prediction point.",
        "action": "Exclude from the primary model."
    },

    "HEALTHCARE_COVERAGE": {
        "decision": "Exclude",
        "availability": "Potentially updated after encounters",
        "leakage_risk": "High",
        "reason": "May contain accumulated coverage information updated after the encounter.",
        "action": "Exclude from the primary model."
    }
}


# ============================================================
# 5. BUILD AUDIT FOR ACTUAL DATASET COLUMNS
# ============================================================

print("\nAuditing actual dataset columns...")

audit_records = []

datasets = {
    "encounters": encounters,
    "patients": patients
}

for dataset_name, dataframe in datasets.items():

    for column in dataframe.columns:

        if column in feature_decisions:

            decision = feature_decisions[column]

        else:

            decision = {
                "decision": "Unreviewed",
                "availability": "Unknown",
                "leakage_risk": "Unknown",
                "reason": (
                    "No predefined eligibility assessment exists "
                    "for this column."
                ),
                "action": (
                    "Inspect the data dictionary and establish "
                    "prediction-time availability before modelling."
                )
            }

        audit_records.append({

            "dataset": dataset_name,

            "feature": column,

            "data_type": str(dataframe[column].dtype),

            "missing_count": int(
                dataframe[column].isna().sum()
            ),

            "missing_percentage": round(
                dataframe[column].isna().mean() * 100, 2
            ),

            "unique_values": int(
                dataframe[column].nunique(dropna=True)
            ),

            "prediction_time_availability":
                decision["availability"],

            "leakage_risk":
                decision["leakage_risk"],

            "eligibility_decision":
                decision["decision"],

            "reason":
                decision["reason"],

            "required_action":
                decision["action"]
        })


audit = pd.DataFrame(audit_records)


# ============================================================
# 6. ADD DERIVED FEATURES
# ============================================================

derived_features = [

    {
        "dataset": "derived",
        "feature": "DURATION_HOURS",
        "data_type": "float",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Known at encounter end",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Calculated using current encounter START and STOP.",
        "required_action": "Inspect outliers and missing values."
    },

    {
        "dataset": "derived",
        "feature": "AGE_AT_ENCOUNTER",
        "data_type": "float",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Available from birthdate and encounter START",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Age calculated at the time of the current encounter.",
        "required_action": "Inspect missing and implausible ages."
    },

    {
        "dataset": "derived",
        "feature": "PRIOR_ENCOUNTERS_30_DAYS",
        "data_type": "integer",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Historical information",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Counts encounters before the current encounter within 30 days.",
        "required_action": "Verify historical calculation and avoid including the current encounter."
    },

    {
        "dataset": "derived",
        "feature": "PRIOR_ENCOUNTERS_90_DAYS",
        "data_type": "integer",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Historical information",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Counts encounters before the current encounter within 90 days.",
        "required_action": "Verify historical calculation and avoid including the current encounter."
    },

    {
        "dataset": "derived",
        "feature": "DAYS_SINCE_PREVIOUS_ENCOUNTER",
        "data_type": "float",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Historical information",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Measures time between the previous encounter START and current encounter START.",
        "required_action": "Handle missing values for patients without previous encounters."
    },

    {
        "dataset": "derived",
        "feature": "START_WEEKDAY",
        "data_type": "integer",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Known at encounter start",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Day of the week on which the current encounter began.",
        "required_action": "Can be derived from START."
    },

    {
        "dataset": "derived",
        "feature": "START_MONTH",
        "data_type": "integer",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Known at encounter start",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Calendar month in which the current encounter began.",
        "required_action": "Can be derived from START."
    },

    {
        "dataset": "derived",
        "feature": "START_HOUR",
        "data_type": "integer",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Known at encounter start",
        "leakage_risk": "Low",
        "eligibility_decision": "Eligible",
        "reason": "Hour of the day in which the current encounter began.",
        "required_action": "Can be derived from START."
    },

    {
        "dataset": "derived",
        "feature": "NEXT_START_AFTER_STOP",
        "data_type": "datetime",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Future information",
        "leakage_risk": "Very high",
        "eligibility_decision": "Exclude",
        "reason": "Used to establish whether a subsequent encounter occurred.",
        "required_action": "Use only for target construction."
    },

    {
        "dataset": "derived",
        "feature": "DAYS_TO_NEXT_ENCOUNTER",
        "data_type": "float",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Future information",
        "leakage_risk": "Very high",
        "eligibility_decision": "Exclude",
        "reason": "Directly derived from the future encounter timestamp.",
        "required_action": "Use only for target construction."
    },

    {
        "dataset": "derived",
        "feature": "REPEAT_ENCOUNTER_7D",
        "data_type": "integer",
        "missing_count": None,
        "missing_percentage": None,
        "unique_values": None,
        "prediction_time_availability": "Future outcome",
        "leakage_risk": "Very high",
        "eligibility_decision": "Target only",
        "reason": "This is the outcome the model is intended to predict.",
        "required_action": "Use as the target variable, never as a predictor."
    }
]

derived_audit = pd.DataFrame(derived_features)

final_audit = pd.concat(
    [audit, derived_audit],
    ignore_index=True
)


# ============================================================
# 7. SAVE AUDIT REPORTS
# ============================================================

audit_path = OUTPUT_DIR / "formal_feature_eligibility_audit.csv"

final_audit.to_csv(
    audit_path,
    index=False
)


# Save a summary of eligibility decisions.
decision_summary = (
    final_audit.groupby(
        ["dataset", "eligibility_decision"]
    )
    .size()
    .reset_index(name="feature_count")
)

decision_summary.to_csv(
    OUTPUT_DIR / "feature_eligibility_summary.csv",
    index=False
)


# Save unresolved features requiring further investigation.
unresolved = final_audit[
    final_audit["eligibility_decision"].isin(
        ["Conditional", "Unreviewed"]
    )
].copy()

unresolved.to_csv(
    OUTPUT_DIR / "features_requiring_investigation.csv",
    index=False
)


# Save a readable text summary.
summary_path = OUTPUT_DIR / "formal_feature_audit_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("FORMAL FEATURE ELIGIBILITY AUDIT\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Prediction point: {PREDICTION_POINT}\n\n")

    file.write(f"Prediction objective: {PREDICTION_OBJECTIVE}\n\n")

    file.write("FEATURE ELIGIBILITY SUMMARY\n")
    file.write("-" * 70 + "\n")

    file.write(
        decision_summary.to_string(index=False)
    )

    file.write("\n\nFEATURES REQUIRING INVESTIGATION\n")
    file.write("-" * 70 + "\n")

    if unresolved.empty:
        file.write("No unresolved features.\n")
    else:
        file.write(
            unresolved[
                ["dataset", "feature", "reason", "required_action"]
            ].to_string(index=False)
        )

    file.write("\n\nALL FEATURE DECISIONS\n")
    file.write("-" * 70 + "\n")

    file.write(
        final_audit[
            [
                "dataset",
                "feature",
                "eligibility_decision",
                "leakage_risk"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FORMAL FEATURE ELIGIBILITY AUDIT COMPLETED")
print("=" * 70)

print("\nELIGIBILITY SUMMARY")

print(
    decision_summary.to_string(index=False)
)

print("\nFEATURES REQUIRING INVESTIGATION")

if unresolved.empty:
    print("No unresolved features.")
else:
    print(
        unresolved[
            [
                "dataset",
                "feature",
                "eligibility_decision",
                "required_action"
            ]
        ].to_string(index=False)
    )

print("\nAUDIT FILES SAVED:")

print(f"1. {audit_path}")
print(f"2. {OUTPUT_DIR / 'feature_eligibility_summary.csv'}")
print(f"3. {OUTPUT_DIR / 'features_requiring_investigation.csv'}")
print(f"4. {summary_path}")