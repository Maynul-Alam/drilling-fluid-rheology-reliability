"""
src/data/audit_dataset.py

Stage 6 -- Dataset verification for the WBM rheology reliability project.

Purpose
-------
Verify that the uploaded Excel file matches the assumptions established
during the earlier dataset audit, BEFORE any modeling is attempted.

If this script contradicts the Stage 6 brief, stop and report before
proceeding to baseline modeling.

Paths are resolved relative to the project root (two levels above this file),
not the current working directory, so you can run this from anywhere.

Run from project root (PowerShell):
    python src/data/audit_dataset.py
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Resolve project root: this file is at <root>/src/data/audit_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
SHEET = "Full_Dataset"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Dataset path:", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at:\n  {DATA_PATH}\n\n"
            f"Place the Excel file at that exact location and re-run.\n"
            f"Expected folder: {DATA_PATH.parent}"
        )

    section("1. Workbook sheets")
    xls = pd.ExcelFile(DATA_PATH)
    print("Sheets:", xls.sheet_names)

    section("2. Load Full_Dataset")
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    print("Shape:", df.shape)

    section("3. Columns")
    for i, c in enumerate(df.columns, start=1):
        print(f"{i:>3}. {c}")

    section("4. Missing values")
    missing = df.isna().sum()
    print(missing[missing > 0] if (missing > 0).any() else "No missing values.")

    section("5. Duplicates")
    print("Duplicate full rows:", int(df.duplicated().sum()))
    if "Sample_ID" in df.columns:
        print("Duplicate Sample_IDs:", int(df["Sample_ID"].duplicated().sum()))

    section("6. Test temperature counts")
    if "Test_Temperature_C" in df.columns:
        print(df["Test_Temperature_C"].value_counts().sort_index())

    section("7. Formulation family counts")
    if "Formulation_Family" in df.columns:
        print(df["Formulation_Family"].value_counts())

    section("8. Family x Temperature crosstab")
    if {"Formulation_Family", "Test_Temperature_C"}.issubset(df.columns):
        print(pd.crosstab(df["Formulation_Family"], df["Test_Temperature_C"]))

    section("9. AV relation check: AV_cP - (PV_cP + YP_lb100ft2 / 2)")
    needed = {"AV_cP", "PV_cP", "YP_lb100ft2"}
    if needed.issubset(df.columns):
        residual = df["AV_cP"] - (df["PV_cP"] + df["YP_lb100ft2"] / 2.0)
        print(residual.describe())
        print("\nMax |residual|:", float(np.abs(residual).max()))
    else:
        print("Required columns missing:", needed - set(df.columns))

    section("10. k_ConsistencyIndex_mPasn distribution")
    if "k_ConsistencyIndex_mPasn" in df.columns:
        print(df["k_ConsistencyIndex_mPasn"].describe())
        print("\nTop 10 values:")
        print(df["k_ConsistencyIndex_mPasn"].value_counts().head(10))
        print("\nNumber of rows exactly equal to 80:",
              int((df["k_ConsistencyIndex_mPasn"] == 80).sum()))
    else:
        print("Column not found.")

    section("11. Aging vs Test temperature equality")
    if {"Aging_Temperature_C", "Test_Temperature_C"}.issubset(df.columns):
        equal = (df["Aging_Temperature_C"] == df["Test_Temperature_C"]).sum()
        print(f"Rows where Aging_Temperature_C == Test_Temperature_C: {equal} / {len(df)}")

    section("12. Barite vs MW correlation")
    if {"Barite_ppb", "MW_ppg"}.issubset(df.columns):
        print(df[["Barite_ppb", "MW_ppg"]].corr())

    section("13. Water_Volume_mL uniqueness")
    if "Water_Volume_mL" in df.columns:
        print(df["Water_Volume_mL"].describe())
        print("Unique values:", df["Water_Volume_mL"].unique())

    section("14. 150C rows by family")
    if {"Test_Temperature_C", "Formulation_Family"}.issubset(df.columns):
        hot = df[df["Test_Temperature_C"] == 150]
        print("Rows at 150C:", len(hot))
        print(hot["Formulation_Family"].value_counts())

    section("15. Year distribution")
    if "Test_Date" in df.columns:
        dates = pd.to_datetime(df["Test_Date"], errors="coerce")
        print(dates.dt.year.value_counts().sort_index())

    section("AUDIT COMPLETE")
    print("If every section matches the Stage 6 brief, proceed to baseline modeling.")
    print("If anything contradicts the brief, STOP and report before modeling.")


if __name__ == "__main__":
    main()