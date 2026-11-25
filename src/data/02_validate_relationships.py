from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REPORT_DIR = PROJECT_ROOT / "reports" / "data_inspection"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading healthcare datasets...")

encounters = pd.read_csv(
    RAW_DATA_DIR / "encounters.csv"
)

patients = pd.read_csv(
    RAW_DATA_DIR / "patients.csv"
)

procedures = pd.read_csv(
    RAW_DATA_DIR / "procedures.csv"
)

payers = pd.read_csv(
    RAW_DATA_DIR / "payers.csv"
)

organizations = pd.read_csv(
    RAW_DATA_DIR / "organizations.csv"
)


# ============================================================
# 1. PRIMARY KEY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("PRIMARY KEY CHECKS")
print("=" * 60)


key_checks = {
    "encounters": (
        encounters,
        "Id"
    ),

    "patients": (
        patients,
        "Id"
    ),

    "payers": (
        payers,
        "Id"
    ),

    "organizations": (
        organizations,
        "Id"
    )
}


for table_name, (df, key_column) in key_checks.items():

    duplicate_keys = df[key_column].duplicated().sum()

    missing_keys = df[key_column].isna().sum()

    print(f"\n{table_name}")

    print(
        f"Duplicate keys: {duplicate_keys:,}"
    )

    print(
        f"Missing keys: {missing_keys:,}"
    )


# ============================================================
# 2. ENCOUNTER-PATIENT RELATIONSHIP
# ============================================================

print("\n" + "=" * 60)
print("ENCOUNTER-PATIENT RELATIONSHIP")
print("=" * 60)


patient_ids = set(
    patients["Id"].dropna()
)

encounter_patient_ids = set(
    encounters["PATIENT"].dropna()
)

unmatched_patient_ids = (
    encounter_patient_ids - patient_ids
)

missing_patient_references = (
    encounters["PATIENT"].isna().sum()
)

print(
    "Unique patients referenced by encounters:",
    len(encounter_patient_ids)
)

print(
    "Patient IDs not found in patients.csv:",
    len(unmatched_patient_ids)
)

print(
    "Encounters with missing patient references:",
    missing_patient_references
)


# ============================================================
# 3. PROCEDURE-ENCOUNTER RELATIONSHIP
# ============================================================

print("\n" + "=" * 60)
print("PROCEDURE-ENCOUNTER RELATIONSHIP")
print("=" * 60)


encounter_ids = set(
    encounters["Id"].dropna()
)

procedure_encounter_ids = set(
    procedures["ENCOUNTER"].dropna()
)

unmatched_encounter_ids = (
    procedure_encounter_ids - encounter_ids
)

missing_procedure_encounters = (
    procedures["ENCOUNTER"].isna().sum()
)

print(
    "Unique encounters referenced by procedures:",
    len(procedure_encounter_ids)
)

print(
    "Procedure encounter IDs not found:",
    len(unmatched_encounter_ids)
)

print(
    "Procedures with missing encounter references:",
    missing_procedure_encounters
)


# ============================================================
# 4. ENCOUNTERS WITHOUT PROCEDURES
# ============================================================

print("\n" + "=" * 60)
print("ENCOUNTERS WITHOUT PROCEDURES")
print("=" * 60)


procedure_counts = (
    procedures.groupby("ENCOUNTER")
    .size()
    .rename("procedure_count")
)

encounters_with_procedures = encounters.merge(
    procedure_counts,
    left_on="Id",
    right_index=True,
    how="left"
)

encounters_with_procedures["procedure_count"] = (
    encounters_with_procedures["procedure_count"]
    .fillna(0)
    .astype(int)
)

no_procedure_count = (
    encounters_with_procedures["procedure_count"] == 0
).sum()

print(
    "Encounters without recorded procedures:",
    no_procedure_count
)

print(
    "Encounters with recorded procedures:",
    len(encounters) - no_procedure_count
)

encounters_with_procedures[
    ["Id", "PATIENT", "procedure_count"]
].to_csv(
    REPORT_DIR / "encounter_procedure_counts.csv",
    index=False
)


# ============================================================
# 5. ENCOUNTERS PER PATIENT
# ============================================================

print("\n" + "=" * 60)
print("ENCOUNTERS PER PATIENT")
print("=" * 60)


patient_encounter_counts = (
    encounters.groupby("PATIENT")
    .size()
    .rename("encounter_count")
    .reset_index()
)

print(
    "Patients with at least one encounter:",
    len(patient_encounter_counts)
)

print(
    "Patients with multiple encounters:",
    (
        patient_encounter_counts["encounter_count"] > 1
    ).sum()
)

print(
    "Maximum encounters for one patient:",
    patient_encounter_counts["encounter_count"].max()
)

patient_encounter_counts.to_csv(
    REPORT_DIR / "patient_encounter_counts.csv",
    index=False
)


# ============================================================
# 6. ENCOUNTER DURATION
# ============================================================

print("\n" + "=" * 60)
print("ENCOUNTER DURATION")
print("=" * 60)


encounters["START"] = pd.to_datetime(
    encounters["START"],
    errors="coerce",
    utc=True
)

encounters["STOP"] = pd.to_datetime(
    encounters["STOP"],
    errors="coerce",
    utc=True
)

encounters["duration_hours"] = (
    encounters["STOP"] - encounters["START"]
).dt.total_seconds() / 3600


negative_duration = (
    encounters["duration_hours"] < 0
).sum()

zero_duration = (
    encounters["duration_hours"] == 0
).sum()

missing_duration = (
    encounters["duration_hours"].isna()
).sum()

print(
    "Negative durations:",
    negative_duration
)

print(
    "Zero durations:",
    zero_duration
)

print(
    "Missing durations:",
    missing_duration
)

print(
    "Median duration in hours:",
    encounters["duration_hours"].median()
)

print(
    "Encounters lasting more than 24 hours:",
    (
        encounters["duration_hours"] > 24
    ).sum()
)


# ============================================================
# 7. COST VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("COST VALIDATION")
print("=" * 60)


cost_columns = [
    "BASE_ENCOUNTER_COST",
    "TOTAL_CLAIM_COST",
    "PAYER_COVERAGE"
]


for column in cost_columns:

    print(f"\n{column}")

    print(
        "Missing values:",
        encounters[column].isna().sum()
    )

    print(
        "Negative values:",
        (
            encounters[column] < 0
        ).sum()
    )

    print(
        "Zero values:",
        (
            encounters[column] == 0
        ).sum()
    )

    print(
        "Minimum:",
        encounters[column].min()
    )

    print(
        "Maximum:",
        encounters[column].max()
    )


# ============================================================
# 8. PAYER COVERAGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("PAYER COVERAGE")
print("=" * 60)


zero_coverage = (
    encounters["PAYER_COVERAGE"] == 0
).sum()

missing_coverage = (
    encounters["PAYER_COVERAGE"].isna()
).sum()

print(
    "Encounters with zero payer coverage:",
    zero_coverage
)

print(
    "Encounters with missing payer coverage:",
    missing_coverage
)


# ============================================================
# 9. PAYER AND ORGANIZATION REFERENCES
# ============================================================

print("\n" + "=" * 60)
print("PAYER AND ORGANIZATION REFERENCES")
print("=" * 60)


payer_ids = set(
    payers["Id"].dropna()
)

encounter_payer_ids = set(
    encounters["PAYER"].dropna()
)

unmatched_payers = (
    encounter_payer_ids - payer_ids
)

print(
    "Payer IDs not found in payers.csv:",
    len(unmatched_payers)
)


organization_ids = set(
    organizations["Id"].dropna()
)

encounter_organization_ids = set(
    encounters["ORGANIZATION"].dropna()
)

unmatched_organizations = (
    encounter_organization_ids - organization_ids
)

print(
    "Organization IDs not found:",
    len(unmatched_organizations)
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)

print(
    "\nReports saved to:",
    REPORT_DIR
)