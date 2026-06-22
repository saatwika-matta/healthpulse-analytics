import pandas as pd
import numpy as np
import json
import os

PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "reports"

def run_eda():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_csv(f"{PROCESSED_DIR}/master.csv", parse_dates=["admission_date","discharge_date"])
    print(f"Master table loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    insights = {}

    # 1. Readmission by department
    readmit_by_dept = (
        df.groupby("department")["readmit_30day"]
        .agg(["mean","count"])
        .rename(columns={"mean":"readmit_rate","count":"n_claims"})
        .sort_values("readmit_rate", ascending=False)
        .reset_index()
    )
    readmit_by_dept["readmit_rate"] = (readmit_by_dept["readmit_rate"] * 100).round(2)
    insights["readmit_by_dept"] = readmit_by_dept.to_dict(orient="records")
    print("Top 3 departments by 30-day readmission rate:")
    print(readmit_by_dept.head(3).to_string(index=False))

    # 2. Revenue by payer
    rev_by_payer = (
        df.groupby("payer_type")
        .agg(
            total_charged    = ("charge_amount",  "sum"),
            total_reimbursed = ("reimbursed_amt",  "sum"),
            total_gap        = ("revenue_gap",     "sum"),
            avg_collection   = ("collection_rate", "mean"),
            n_claims         = ("claim_id",        "count"),
            denial_rate      = ("claim_denied",    "mean"),
        )
        .round(2)
        .reset_index()
    )
    rev_by_payer["avg_collection_pct"] = (rev_by_payer["avg_collection"] * 100).round(1)
    rev_by_payer["denial_rate_pct"]    = (rev_by_payer["denial_rate"]    * 100).round(1)
    insights["revenue_by_payer"] = rev_by_payer.to_dict(orient="records")
    print("\nTop 3 payers by revenue gap:")
    print(rev_by_payer.nlargest(3,"total_gap")[["payer_type","total_gap","denial_rate_pct"]].to_string(index=False))

    # 3. Risk factors
    risk_factors = {}
    for col in ["has_diabetes","has_hypertension","is_smoker","is_high_risk"]:
        grp = df.groupby(col)["readmit_30day"].mean() * 100
        risk_factors[col] = {
            "without": round(grp.get(0, 0), 2),
            "with":    round(grp.get(1, 0), 2),
            "lift_pct": round((grp.get(1,0) - grp.get(0,0)) / max(grp.get(0,0.001),0.001) * 100, 1),
        }
    insights["risk_factor_readmit"] = risk_factors
    print("\nReadmission rate by risk factor:")
    for factor, vals in risk_factors.items():
        print(f"  {factor:<20} without={vals['without']:.1f}%  with={vals['with']:.1f}%  lift={vals['lift_pct']:+.1f}%")

    # 4. Monthly revenue trend
    monthly = (
        df.set_index("admission_date")
        .resample("ME")[["charge_amount","reimbursed_amt","revenue_gap"]]
        .sum()
        .reset_index()
    )
    monthly["month"] = monthly["admission_date"].dt.strftime("%Y-%m")
    monthly = monthly.drop(columns=["admission_date"]).round(2)
    insights["monthly_revenue"] = monthly.to_dict(orient="records")
    print(f"\nMonthly revenue trend: {len(monthly)} months")

    # 5. KPI snapshot
    insights["kpi_snapshot"] = {
        "total_claims":            int(len(df)),
        "unique_patients":         int(df["patient_id"].nunique()),
        "total_revenue_charged":   round(float(df["charge_amount"].sum()), 2),
        "total_revenue_collected": round(float(df["reimbursed_amt"].sum()), 2),
        "total_revenue_gap":       round(float(df["revenue_gap"].sum()), 2),
        "collection_rate_pct":     round(float(df["collection_rate"].mean()) * 100, 2),
        "readmit_rate_pct":        round(float(df["readmit_30day"].mean()) * 100, 2),
        "denial_rate_pct":         round(float(df["claim_denied"].mean()) * 100, 2),
        "avg_length_of_stay":      round(float(df["length_of_stay"].mean()), 2),
        "avg_charge_per_claim":    round(float(df["charge_amount"].mean()), 2),
    }

    print("\nKPI Snapshot:")
    for k,v in insights["kpi_snapshot"].items():
        print(f"  {k:<30} {v}")

    with open(f"{REPORTS_DIR}/eda_summary.json", "w") as f:
        json.dump(insights, f, indent=2, default=str)
    print(f"\n  EDA insights saved → {REPORTS_DIR}/eda_summary.json")

    with open(f"{REPORTS_DIR}/eda_insights.txt", "w") as f:
        kpi = insights["kpi_snapshot"]
        f.write("HealthPulse Analytics — Key Findings\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total claims:          {kpi['total_claims']:,}\n")
        f.write(f"Unique patients:       {kpi['unique_patients']:,}\n")
        f.write(f"Revenue charged:       ${kpi['total_revenue_charged']:,.0f}\n")
        f.write(f"Revenue collected:     ${kpi['total_revenue_collected']:,.0f}\n")
        f.write(f"Revenue gap:           ${kpi['total_revenue_gap']:,.0f}\n")
        f.write(f"Collection rate:       {kpi['collection_rate_pct']}%\n")
        f.write(f"30-day readmit rate:   {kpi['readmit_rate_pct']}%\n")
        f.write(f"Denial rate:           {kpi['denial_rate_pct']}%\n")
        f.write(f"Avg length of stay:    {kpi['avg_length_of_stay']} days\n")
    print(f"  Insights saved → {REPORTS_DIR}/eda_insights.txt")
    print("\nWeek 1 COMPLETE! Next: Week 2 -> readmission prediction model")

if __name__ == "__main__":
    run_eda()
