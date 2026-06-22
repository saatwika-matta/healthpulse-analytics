import pandas as pd
import numpy as np
import os
from datetime import datetime

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "reports"

AGE_MIN, AGE_MAX    = 18, 95
BMI_MIN, BMI_MAX    = 15.0, 60.0
LOS_MAX             = 180
CHARGE_UPPER_ZSCORE = 4.0
VALID_PAYERS = {"Medicare","Medicaid","BlueCross","Aetna","UnitedHealth","Cigna","Self-Pay"}

class QualityReport:
    def __init__(self):
        self.issues = []
        self.stats  = {}

    def log(self, table, check, n_affected, action):
        self.issues.append({"table":table,"check":check,"n_affected":n_affected,"action":action})

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = ["="*60, "  HealthPulse — Data Quality Report",
                 f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "="*60, ""]
        for tbl in ["patients","claims","billing"]:
            tbl_issues = [i for i in self.issues if i["table"]==tbl]
            if not tbl_issues: continue
            lines.append(f"TABLE: {tbl.upper()}")
            lines.append("-"*40)
            for issue in tbl_issues:
                lines.append(f"  [{issue['n_affected']:>5} rows]  {issue['check']}")
                lines.append(f"            → {issue['action']}")
            lines.append("")
        if self.stats:
            lines.append("FINAL STATS")
            lines.append("-"*40)
            for k,v in self.stats.items():
                lines.append(f"  {k}: {v}")
        with open(path,"w") as f:
            f.write("\n".join(lines))
        print(f"  Quality report → {path}")

def clean_patients(df, qr):
    orig_len = len(df)
    dupes = df.duplicated(subset=["mrn"], keep="first").sum()
    qr.log("patients","Duplicate MRN",dupes,"Keep first, drop rest")
    df = df.drop_duplicates(subset=["mrn"], keep="first")
    bad_age = (~df["age"].between(AGE_MIN, AGE_MAX)).sum()
    qr.log("patients",f"Age outside [{AGE_MIN},{AGE_MAX}]",int(bad_age),"Set to NaN, impute with median")
    df.loc[~df["age"].between(AGE_MIN, AGE_MAX), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)
    bad_bmi = ((df["bmi"] < BMI_MIN) | (df["bmi"] > BMI_MAX)).sum()
    qr.log("patients",f"BMI outside [{BMI_MIN},{BMI_MAX}]",int(bad_bmi),"Cap to valid range")
    df["bmi"] = df["bmi"].clip(BMI_MIN, BMI_MAX)
    null_bmi = df["bmi"].isna().sum()
    qr.log("patients","Null BMI",int(null_bmi),"Impute with median")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median()).round(1)
    df["gender"] = df["gender"].str.strip().str.upper()
    df["gender"] = df["gender"].map({"M":"M","F":"F"}).fillna("Unknown")
    null_state = df["state"].isna().sum()
    qr.log("patients","Null state",int(null_state),"Fill with Unknown")
    df["state"] = df["state"].fillna("Unknown")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["is_high_risk"] = ((df["has_diabetes"]==1)|(df["has_hypertension"]==1)|(df["is_smoker"]==1)|(df["age"]>=65)).astype(int)
    df["bmi_category"] = pd.cut(df["bmi"],bins=[0,18.5,25.0,30.0,100],labels=["Underweight","Normal","Overweight","Obese"])
    df["age_group"] = pd.cut(df["age"],bins=[17,30,45,60,75,96],labels=["18-30","31-45","46-60","61-75","76+"])
    qr.log("patients","Total removed (dedup)",orig_len-len(df),f"Final: {len(df):,} rows")
    return df

def clean_claims(df, valid_patient_ids, qr):
    orig_len = len(df)
    dupes = df.duplicated(subset=["claim_id"], keep="first").sum()
    qr.log("claims","Duplicate claim_id",int(dupes),"Keep first")
    df = df.drop_duplicates(subset=["claim_id"], keep="first")
    orphans = (~df["patient_id"].isin(valid_patient_ids)).sum()
    qr.log("claims","No matching patient",int(orphans),"Drop rows")
    df = df[df["patient_id"].isin(valid_patient_ids)]
    df["admission_date"]  = pd.to_datetime(df["admission_date"],  errors="coerce")
    df["discharge_date"]  = pd.to_datetime(df["discharge_date"],  errors="coerce")
    bad_dates = (df["discharge_date"] < df["admission_date"]).sum()
    qr.log("claims","Discharge before admission",int(bad_dates),"Swap dates")
    mask = df["discharge_date"] < df["admission_date"]
    df.loc[mask, ["admission_date","discharge_date"]] = df.loc[mask, ["discharge_date","admission_date"]].values
    df["length_of_stay"] = (df["discharge_date"] - df["admission_date"]).dt.days.clip(0, LOS_MAX)
    z_scores = np.abs((df["charge_amount"] - df["charge_amount"].mean()) / df["charge_amount"].std())
    outliers = (z_scores > CHARGE_UPPER_ZSCORE).sum()
    qr.log("claims",f"Charge outliers z>{CHARGE_UPPER_ZSCORE}",int(outliers),"Cap to 99th percentile")
    df["charge_amount"] = df["charge_amount"].clip(upper=df["charge_amount"].quantile(0.99)).round(2)
    bad_reimb = (df["reimbursed_amt"] > df["charge_amount"]).sum()
    qr.log("claims","Reimbursement > charge",int(bad_reimb),"Cap to charge amount")
    df["reimbursed_amt"] = df[["reimbursed_amt","charge_amount"]].min(axis=1)
    df["icd10_primary"] = df["icd10_primary"].fillna("Z99.9")
    df["revenue_gap"]       = (df["charge_amount"] - df["reimbursed_amt"]).round(2)
    df["collection_rate"]   = (df["reimbursed_amt"] / df["charge_amount"]).round(4)
    df["admission_year"]    = df["admission_date"].dt.year
    df["admission_month"]   = df["admission_date"].dt.month
    df["admission_quarter"] = df["admission_date"].dt.to_period("Q").astype(str)
    qr.log("claims","Total removed",orig_len-len(df),f"Final: {len(df):,} rows")
    return df

def clean_billing(df, valid_claim_ids, qr):
    orphans = (~df["claim_id"].isin(valid_claim_ids)).sum()
    qr.log("billing","No matching claim",int(orphans),"Drop rows")
    df = df[df["claim_id"].isin(valid_claim_ids)]
    df["payment_delay_days"] = df["payment_delay_days"].clip(lower=0)
    return df

def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR,   exist_ok=True)
    qr = QualityReport()

    print("Loading raw data...")
    patients = pd.read_csv(f"{RAW_DIR}/patients.csv")
    claims   = pd.read_csv(f"{RAW_DIR}/claims.csv")
    billing  = pd.read_csv(f"{RAW_DIR}/billing.csv")
    print(f"  patients: {len(patients):,} | claims: {len(claims):,} | billing: {len(billing):,}")

    print("\nCleaning patients...")
    patients_clean = clean_patients(patients.copy(), qr)
    patients_clean.to_csv(f"{PROCESSED_DIR}/patients_clean.csv", index=False)
    print(f"  Done: {len(patients_clean):,} rows")

    print("\nCleaning claims...")
    claims_clean = clean_claims(claims.copy(), set(patients_clean["patient_id"]), qr)
    claims_clean.to_csv(f"{PROCESSED_DIR}/claims_clean.csv", index=False)
    print(f"  Done: {len(claims_clean):,} rows")

    print("\nCleaning billing...")
    billing_clean = clean_billing(billing.copy(), set(claims_clean["claim_id"]), qr)
    billing_clean.to_csv(f"{PROCESSED_DIR}/billing_clean.csv", index=False)
    print(f"  Done: {len(billing_clean):,} rows")

    print("\nBuilding master table...")
    master = (
        claims_clean
        .merge(patients_clean[["patient_id","age","gender","state","insurance_type",
                                "bmi","has_diabetes","has_hypertension","is_smoker",
                                "is_high_risk","bmi_category","age_group"]].drop_duplicates("patient_id"),
               on="patient_id", how="left")
        .merge(billing_clean, on="claim_id", how="left")
    )
    master.to_csv(f"{PROCESSED_DIR}/master.csv", index=False)
    print(f"  Done: {master.shape[0]:,} rows x {master.shape[1]} columns")

    qr.stats = {
        "Total patients":     f"{len(patients_clean):,}",
        "Total claims":       f"{len(claims_clean):,}",
        "Date range":         f"{master['admission_date'].min()} to {master['admission_date'].max()}",
        "Avg charge":         f"${master['charge_amount'].mean():,.2f}",
        "Collection rate":    f"{master['collection_rate'].mean()*100:.1f}%",
        "30-day readmit rate":f"{master['readmit_30day'].mean()*100:.1f}%",
        "Denial rate":        f"{master['claim_denied'].mean()*100:.1f}%",
    }
    qr.save(f"{REPORTS_DIR}/data_quality_report.txt")

    print("\n" + "="*50)
    print("  PIPELINE COMPLETE")
    print("="*50)
    for k,v in qr.stats.items():
        print(f"  {k:<25} {v}")
    print("\nNext: run python3 src/eda.py")

if __name__ == "__main__":
    run_pipeline()
