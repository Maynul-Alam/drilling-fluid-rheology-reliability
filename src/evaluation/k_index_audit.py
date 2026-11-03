"""
src/evaluation/k_index_audit.py

Stage 24 -- Cross-tabulate the 12 non-80 k_ConsistencyIndex_mPasn rows
against Operator / Shift / Viscometer / Test_Date to test the "reporting
artefact" hypothesis.

Runtime: < 5 seconds.

Output: results/metrics/k_index_audit.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "k_index_audit.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)
    k = df["k_ConsistencyIndex_mPasn"]

    non80 = df[k != 80].copy()
    print(f"Total rows: {len(df)}")
    print(f"Rows with k == 80: {(k == 80).sum()}")
    print(f"Rows with k != 80: {len(non80)}")
    print()

    print("Non-80 values:")
    print(non80[["Sample_ID", "Test_Date", "Operator", "Shift",
                 "Viscometer_Model", "Formulation_Family",
                 "Test_Temperature_C", "k_ConsistencyIndex_mPasn"]]
          .to_string(index=False))

    rows = []
    for col in ["Operator", "Shift", "Viscometer_Model", "Formulation_Family"]:
        full_ct = df[col].value_counts(normalize=True).rename("full_frac")
        non80_ct = non80[col].value_counts(normalize=True).rename("non80_frac")
        ct = pd.concat([full_ct, non80_ct], axis=1).fillna(0)
        ct["enrichment"] = ct["non80_frac"] / ct["full_frac"].replace(0, float("nan"))
        for cat, row in ct.iterrows():
            rows.append({
                "column": col,
                "category": str(cat),
                "full_count": int((df[col] == cat).sum()),
                "full_frac": float(row["full_frac"]),
                "non80_count": int((non80[col] == cat).sum()),
                "non80_frac": float(row["non80_frac"]),
                "enrichment": float(row["enrichment"])
                  if not pd.isna(row["enrichment"]) else float("nan"),
            })

    out = pd.DataFrame(rows)
    print("\nEnrichment of non-80 rows by metadata (enrichment > 1 = over-represented):")
    print(out.to_string(index=False))

    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")

    # Year cross-tab
    non80_dates = pd.to_datetime(non80["Test_Date"], errors="coerce")
    print("\nNon-80 Test_Date range:",
          non80_dates.min(), "to", non80_dates.max())
    print("Non-80 rows by year:")
    print(non80_dates.dt.year.value_counts().sort_index())


if __name__ == "__main__":
    main()