import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "reports"

def run_revenue_analysis():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_csv(f"{PROCESSED_DIR}/master.csv", parse_dates=["admission_date","discharge_date"])
    print(f"Loaded: {df.shape[0]:,} rows\n")

    results = {}

    # 1. Monthly revenue trend + 3-month rolling average
    print("1. Monthly revenue trend...")
    monthly = (
        df.set_index("admission_date")
        .resample("ME")[["charge_amount","reimbursed_amt","revenue_gap"]]
        .sum()
        .reset_index()
    )
    monthly["month"]            = monthly["admission_date"].dt.strftime("%Y-%m")
    monthly["rolling_avg_3m"]   = monthly["reimbursed_amt"].rolling(3, min_periods=1).mean().round(2)
    monthly["mom_growth_pct"]   = monthly["reimbursed_amt"].pct_change().mul(100).round(2)
    results["monthly_trend"]    = monthly.drop(columns=["admission_date"]).round(2).to_dict(orient="records")
    print(f"   {len(monthly)} months of data")

    # 2. Revenue by department
    print("2. Revenue by department...")
    by_dept = (
        df.groupby("department")
        .agg(
            total_charged    = ("charge_amount",  "sum"),
            total_collected  = ("reimbursed_amt",  "sum"),
            total_gap        = ("revenue_gap",     "sum"),
            avg_collection   = ("collection_rate", "mean"),
            n_claims         = ("claim_id",        "count"),
            denial_rate      = ("claim_denied",    "mean"),
            avg_los          = ("length_of_stay",  "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("total_charged", ascending=False)
    )
    by_dept["avg_collection_pct"] = (by_dept["avg_collection"] * 100).round(1)
    by_dept["denial_rate_pct"]    = (by_dept["denial_rate"]    * 100).round(1)
    by_dept["avg_los"]            = by_dept["avg_los"].round(1)
    results["revenue_by_dept"]    = by_dept.to_dict(orient="records")
    print(f"   Top dept by revenue: {by_dept.iloc[0]['department']} (${by_dept.iloc[0]['total_charged']:,.0f})")

    # 3. Revenue by payer with leakage analysis
    print("3. Payer leakage analysis...")
    by_payer = (
        df.groupby("payer_type")
        .agg(
            total_charged    = ("charge_amount",  "sum"),
            total_collected  = ("reimbursed_amt",  "sum"),
            total_gap        = ("revenue_gap",     "sum"),
            avg_collection   = ("collection_rate", "mean"),
            n_claims         = ("claim_id",        "count"),
            denial_rate      = ("claim_denied",    "mean"),
            avg_payment_delay= ("payment_delay_days","mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("total_gap", ascending=False)
    )
    by_payer["avg_collection_pct"] = (by_payer["avg_collection"] * 100).round(1)
    by_payer["denial_rate_pct"]    = (by_payer["denial_rate"]    * 100).round(1)
    by_payer["leakage_pct"]        = (by_payer["total_gap"] / by_payer["total_charged"] * 100).round(1)
    results["revenue_by_payer"]    = by_payer.to_dict(orient="records")
    print(f"   Highest leakage: {by_payer.iloc[0]['payer_type']} ({by_payer.iloc[0]['leakage_pct']}% gap)")

    # 4. Denial rate analysis by department + payer
    print("4. Denial rate analysis...")
    denial_by_dept = (
        df.groupby("department")["claim_denied"]
        .agg(["mean","sum","count"])
        .rename(columns={"mean":"denial_rate","sum":"n_denied","count":"n_claims"})
        .reset_index()
        .sort_values("denial_rate", ascending=False)
    )
    denial_by_dept["denial_rate_pct"] = (denial_by_dept["denial_rate"] * 100).round(2)
    results["denial_by_dept"] = denial_by_dept.to_dict(orient="records")

    denial_by_payer = (
        df.groupby("payer_type")["claim_denied"]
        .agg(["mean","sum","count"])
        .rename(columns={"mean":"denial_rate","sum":"n_denied","count":"n_claims"})
        .reset_index()
        .sort_values("denial_rate", ascending=False)
    )
    denial_by_payer["denial_rate_pct"] = (denial_by_payer["denial_rate"] * 100).round(2)
    results["denial_by_payer"] = denial_by_payer.to_dict(orient="records")
    print(f"   Highest denial dept: {denial_by_dept.iloc[0]['department']} ({denial_by_dept.iloc[0]['denial_rate_pct']}%)")

    # 5. Quarterly revenue trend (year-over-year)
    print("5. Quarterly YoY analysis...")
    df["quarter"] = df["admission_date"].dt.to_period("Q").astype(str)
    quarterly = (
        df.groupby("quarter")
        .agg(
            total_charged   = ("charge_amount",  "sum"),
            total_collected = ("reimbursed_amt",  "sum"),
            total_gap       = ("revenue_gap",     "sum"),
            n_claims        = ("claim_id",        "count"),
        )
        .round(2)
        .reset_index()
    )
    quarterly["yoy_growth_pct"] = quarterly["total_collected"].pct_change(4).mul(100).round(2)
    results["quarterly_trend"]  = quarterly.to_dict(orient="records")
    print(f"   {len(quarterly)} quarters of data")

    # 6. CPT code revenue analysis
    print("6. Procedure code revenue analysis...")
    by_cpt = (
        df.groupby("cpt_code")
        .agg(
            total_revenue  = ("charge_amount",  "sum"),
            total_collected= ("reimbursed_amt",  "sum"),
            n_procedures   = ("claim_id",        "count"),
            avg_charge     = ("charge_amount",   "mean"),
            denial_rate    = ("claim_denied",    "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    by_cpt["denial_rate_pct"] = (by_cpt["denial_rate"] * 100).round(1)
    results["revenue_by_cpt"] = by_cpt.to_dict(orient="records")
    print(f"   Top procedure: CPT {by_cpt.iloc[0]['cpt_code']} (${by_cpt.iloc[0]['total_revenue']:,.0f})")

    # 7. Key revenue KPIs
    total_charged   = df["charge_amount"].sum()
    total_collected = df["reimbursed_amt"].sum()
    total_gap       = df["revenue_gap"].sum()
    results["revenue_kpis"] = {
        "total_charged":          round(float(total_charged), 2),
        "total_collected":        round(float(total_collected), 2),
        "total_gap":              round(float(total_gap), 2),
        "overall_collection_pct": round(float(total_collected/total_charged*100), 2),
        "overall_denial_rate":    round(float(df["claim_denied"].mean()*100), 2),
        "avg_payment_delay_days": round(float(df["payment_delay_days"].mean()), 1),
        "total_write_offs":       round(float(df["write_off_amount"].sum()), 2),
        "avg_revenue_per_claim":  round(float(df["reimbursed_amt"].mean()), 2),
    }

    # Save
    with open(f"{REPORTS_DIR}/revenue_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {REPORTS_DIR}/revenue_analysis.json")

    # Print summary
    kpi = results["revenue_kpis"]
    print("\n" + "="*55)
    print("REVENUE ANALYSIS SUMMARY")
    print("="*55)
    print(f"  Total charged:          ${kpi['total_charged']:>15,.2f}")
    print(f"  Total collected:        ${kpi['total_collected']:>15,.2f}")
    print(f"  Total revenue gap:      ${kpi['total_gap']:>15,.2f}")
    print(f"  Collection rate:        {kpi['overall_collection_pct']:>14.1f}%")
    print(f"  Denial rate:            {kpi['overall_denial_rate']:>14.1f}%")
    print(f"  Avg payment delay:      {kpi['avg_payment_delay_days']:>13.1f} days")
    print(f"  Total write-offs:       ${kpi['total_write_offs']:>15,.2f}")
    print(f"\n  Top payer gaps:")
    for p in by_payer.head(3).to_dict(orient="records"):
        print(f"    {p['payer_type']:<15} ${p['total_gap']:>12,.0f} ({p['leakage_pct']}% leakage)")
    print(f"\n  Top departments by denial rate:")
    for d in denial_by_dept.head(3).to_dict(orient="records"):
        print(f"    {d['department']:<20} {d['denial_rate_pct']}%")

    print("\nWeek 3 COMPLETE! Next: Week 4 -> dashboard")

if __name__ == "__main__":
    run_revenue_analysis()
