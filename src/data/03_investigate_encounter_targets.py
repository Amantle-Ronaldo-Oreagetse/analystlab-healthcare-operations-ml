from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "encounters.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "target_investigation"
FIGURE_DIR = PROJECT_ROOT / "figures" / "eda"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading encounters dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. PREPARE ANALYSIS VARIABLES
# ============================================================

df["START"] = pd.to_datetime(df["START"], errors="coerce")
df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")

numeric_columns = [
    "BASE_ENCOUNTER_COST",
    "TOTAL_CLAIM_COST",
    "PAYER_COVERAGE"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# Calculate encounter duration in hours.
# STOP is used only to construct the descriptive target.
# It will not be treated as a model feature.

df["duration_hours"] = (
    df["STOP"] - df["START"]
).dt.total_seconds() / 3600


# Identify invalid or unusual duration records.

df["valid_duration"] = (
    df["duration_hours"].notna()
    & (df["duration_hours"] > 0)
)


# ============================================================
# 4. ENCOUNTER CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("ENCOUNTER CLASS DISTRIBUTION")
print("=" * 60)

class_distribution = (
    df["ENCOUNTERCLASS"]
    .value_counts(dropna=False)
    .rename_axis("encounter_class")
    .reset_index(name="encounter_count")
)

class_distribution["percentage"] = (
    class_distribution["encounter_count"]
    / len(df) * 100
)

print(class_distribution.to_string(index=False))

class_distribution.to_csv(
    REPORT_DIR / "encounter_class_distribution.csv",
    index=False
)


# Visualisation

plt.figure(figsize=(11, 6))

sns.barplot(
    data=class_distribution,
    x="encounter_class",
    y="encounter_count"
)

plt.title("Distribution of Healthcare Encounter Classes")
plt.xlabel("Encounter Class")
plt.ylabel("Number of Encounters")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "encounter_class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. ENCOUNTER DURATION INVESTIGATION
# ============================================================

print("\n" + "=" * 60)
print("ENCOUNTER DURATION INVESTIGATION")
print("=" * 60)

valid_durations = df.loc[
    df["valid_duration"],
    "duration_hours"
]

print(f"Total encounters: {len(df):,}")
print(f"Valid positive durations: {len(valid_durations):,}")
print(f"Invalid or missing durations: {(~df['valid_duration']).sum():,}")

if not valid_durations.empty:

    print("\nOverall duration statistics:")

    print(
        valid_durations.describe(
            percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )

    encounters_over_24h = (
        valid_durations > 24
    ).sum()

    print(
        f"\nEncounters exceeding 24 hours: "
        f"{encounters_over_24h:,}"
    )

    print(
        f"Percentage exceeding 24 hours: "
        f"{encounters_over_24h / len(valid_durations) * 100:.2f}%"
    )


# Duration statistics by encounter class.

duration_summary = (
    df.loc[df["valid_duration"]]
    .groupby("ENCOUNTERCLASS", dropna=False)
    ["duration_hours"]
    .agg(
        encounter_count="count",
        mean_duration="mean",
        median_duration="median",
        standard_deviation="std",
        minimum_duration="min",
        maximum_duration="max"
    )
    .reset_index()
)

print("\nDuration statistics by encounter class:")
print(duration_summary.to_string(index=False))

duration_summary.to_csv(
    REPORT_DIR / "duration_summary_by_class.csv",
    index=False
)


# Visualisation: duration by encounter class.

duration_plot_data = df.loc[
    df["valid_duration"]
].copy()

plt.figure(figsize=(12, 7))

sns.boxplot(
    data=duration_plot_data,
    x="ENCOUNTERCLASS",
    y="duration_hours"
)

plt.title("Encounter Duration by Encounter Class")
plt.xlabel("Encounter Class")
plt.ylabel("Duration (Hours)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "duration_by_class_boxplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. ENCOUNTER COST INVESTIGATION
# ============================================================

print("\n" + "=" * 60)
print("ENCOUNTER COST INVESTIGATION")
print("=" * 60)

print("\nOverall cost statistics:")

print(
    df[
        [
            "BASE_ENCOUNTER_COST",
            "TOTAL_CLAIM_COST",
            "PAYER_COVERAGE"
        ]
    ].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)


# Cost statistics by encounter class.

cost_summary = (
    df.groupby("ENCOUNTERCLASS", dropna=False)
    .agg(
        encounter_count=("Id", "count"),
        mean_claim_cost=("TOTAL_CLAIM_COST", "mean"),
        median_claim_cost=("TOTAL_CLAIM_COST", "median"),
        standard_deviation=("TOTAL_CLAIM_COST", "std"),
        minimum_claim_cost=("TOTAL_CLAIM_COST", "min"),
        maximum_claim_cost=("TOTAL_CLAIM_COST", "max"),
        mean_base_cost=("BASE_ENCOUNTER_COST", "mean")
    )
    .reset_index()
)

print("\nCost statistics by encounter class:")
print(cost_summary.to_string(index=False))

cost_summary.to_csv(
    REPORT_DIR / "cost_summary_by_class.csv",
    index=False
)


# Visualisation: log-transformed claim costs.
# log1p allows zero-cost records to remain visible.

cost_plot_data = df.loc[
    df["TOTAL_CLAIM_COST"].notna()
    & (df["TOTAL_CLAIM_COST"] >= 0)
].copy()

cost_plot_data["log_claim_cost"] = (
    __import__("numpy").log1p(
        cost_plot_data["TOTAL_CLAIM_COST"]
    )
)

plt.figure(figsize=(12, 7))

sns.boxplot(
    data=cost_plot_data,
    x="ENCOUNTERCLASS",
    y="log_claim_cost"
)

plt.title("Log-Transformed Total Claim Cost by Encounter Class")
plt.xlabel("Encounter Class")
plt.ylabel("log(1 + Total Claim Cost)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "total_claim_cost_by_class_boxplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. PAYER COVERAGE INVESTIGATION
# ============================================================

print("\n" + "=" * 60)
print("PAYER COVERAGE INVESTIGATION")
print("=" * 60)

coverage = df["PAYER_COVERAGE"]

zero_coverage = (coverage == 0).sum()
positive_coverage = (coverage > 0).sum()
missing_coverage = coverage.isna().sum()

coverage_summary = pd.DataFrame({
    "category": [
        "Zero payer coverage",
        "Positive payer coverage",
        "Missing payer coverage"
    ],
    "encounter_count": [
        zero_coverage,
        positive_coverage,
        missing_coverage
    ]
})

coverage_summary["percentage"] = (
    coverage_summary["encounter_count"]
    / len(df) * 100
)

print(coverage_summary.to_string(index=False))

coverage_summary.to_csv(
    REPORT_DIR / "payer_coverage_summary.csv",
    index=False
)


# Visualisation

plt.figure(figsize=(8, 6))

sns.barplot(
    data=coverage_summary,
    x="category",
    y="encounter_count"
)

plt.title("Distribution of Recorded Payer Coverage")
plt.xlabel("Payer Coverage Category")
plt.ylabel("Number of Encounters")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "payer_coverage_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. RELATIONSHIP BETWEEN DURATION AND CLAIM COST
# ============================================================

print("\n" + "=" * 60)
print("DURATION AND CLAIM COST RELATIONSHIP")
print("=" * 60)

relationship_data = df.loc[
    df["valid_duration"]
    & df["TOTAL_CLAIM_COST"].notna()
    & (df["TOTAL_CLAIM_COST"] >= 0)
].copy()

if len(relationship_data) > 1:

    correlation = relationship_data[
        ["duration_hours", "TOTAL_CLAIM_COST"]
    ].corr(method="spearman").iloc[0, 1]

    print(
        f"Spearman correlation between duration "
        f"and total claim cost: {correlation:.4f}"
    )

    relationship_summary = pd.DataFrame({
        "measure": [
            "Spearman correlation",
            "Records included"
        ],
        "value": [
            correlation,
            len(relationship_data)
        ]
    })

    relationship_summary.to_csv(
        REPORT_DIR / "duration_cost_relationship.csv",
        index=False
    )

    # Sample points for a clearer visualisation if necessary.

    plot_data = relationship_data.sample(
        n=min(5000, len(relationship_data)),
        random_state=42
    )

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=plot_data,
        x="duration_hours",
        y="TOTAL_CLAIM_COST",
        alpha=0.4
    )

    plt.title("Encounter Duration vs Total Claim Cost")
    plt.xlabel("Encounter Duration (Hours)")
    plt.ylabel("Total Claim Cost")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "duration_vs_claim_cost.png",
        dpi=300
    )

    plt.close()

else:
    print("Insufficient valid records for relationship analysis.")


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETED")
print("=" * 60)

print("\nReports saved to:")
print(REPORT_DIR)

print("\nFigures saved to:")
print(FIGURE_DIR)

print("\nGenerated reports:")
for file in sorted(REPORT_DIR.glob("*.csv")):
    print(f" - {file.name}")

print("\nGenerated figures:")
for file in sorted(FIGURE_DIR.glob("*.png")):
    print(f" - {file.name}")

print("\nNo original dataset was modified.")