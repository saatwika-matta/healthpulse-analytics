import numpy as np
import pandas as pd
import random
import string
from datetime import datetime, timedelta
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_PATIENTS = 5000
N_CLAIMS   = 18000
START_DATE = datetime(2021, 1, 1)
END_DATE   = datetime(2024, 12, 31)

FIRST_NAMES = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","William","Barbara","David","Elizabeth","Richard","Susan","Joseph","Jessica","Thomas","Sarah","Charles","Karen","Priya","Arjun","Wei","Mei","Carlos","Sofia","Amara","Kwame","Fatima","Omar","Yuki","Kenji","Aisha","Hassan"]
LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Wilson","Anderson","Taylor","Thomas","Jackson","White","Harris","Martin","Thompson","Lee","Patel","Nguyen","Kim","Rodriguez","Martinez","Hernandez"]
STATES = ["MA","NY","CA","TX","FL","IL","PA","OH","GA","NC","MI","NJ","WA","AZ","TN"]
DEPARTMENTS = ["Cardiology","Orthopedics","Oncology","Emergency","Neurology","Pediatrics","Radiology","General Surgery","Internal Medicine","Psychiatry"]
PAYER_TYPES  = ["Medicare","Medicaid","BlueCross","Aetna","UnitedHealth","Cigna","Self-Pay"]
PAYER_WEIGHTS = [0.25, 0.20, 0.18, 0.15, 0.12, 0.07, 0.03]
ICD10_LIST = ["I21.0","I50.9","J18.9","E11.9","M54.5","F32.9","J44.1","N18.3","C34.90","S72.001A","I63.9","G43.909","K92.1","Z51.11","R05.9","I10"]
CPT_CODES = {"99213":150,"99214":220,"93000":90,"71046":300,"80053":130,"43239":1800,"27447":12000,"70553":2200,"99285":850,"96413":650}
CPT_LIST = list(CPT_CODES.keys())

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

def random_mrn():
    return "MRN" + "".join(random.choices(string.digits, k=7))

def inject_noise(df, col, null_rate=0.04):
    mask = np.random.random(len(df)) < null_rate
    df.loc[mask, col] = np.nan
    return df

def inject_duplicates(df, n=60):
    dupe_idx = np.random.choice(df.index, size=n, replace=False)
    return pd.concat([df, df.loc[dupe_idx].copy()], ignore_index=True)

def inject_outliers(series, n=25, multiplier=8):
    idx = np.random.choice(series.index, size=n, replace=False)
    series.loc[idx] = series.loc[idx] * multiplier
    return series

def generate_patients(n):
    ages = np.random.normal(loc=52, scale=18, size=n).clip(18, 95).astype(int)
    records = []
    for i in range(n):
        age = ages[i]
        has_diabetes = int(np.random.random() < (0.15 + 0.004 * max(0, age - 40)))
        has_hypertension = int(np.random.random() < (0.10 + 0.006 * max(0, age - 35)))
        has_smoking = int(np.random.random() < 0.18)
        bmi = float(np.clip(np.random.normal(loc=27 + 2*has_diabetes + 1.5*has_hypertension, scale=5), 16, 55))
        records.append({
            "patient_id": f"P{i+1:05d}",
            "mrn": random_mrn(),
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "age": age,
            "gender": random.choice(["M","F","M","F","M"]),
            "state": random.choice(STATES),
            "insurance_type": np.random.choice(PAYER_TYPES, p=PAYER_WEIGHTS),
            "bmi": round(bmi, 1),
            "has_diabetes": has_diabetes,
            "has_hypertension": has_hypertension,
            "is_smoker": has_smoking,
            "registration_date": random_date(START_DATE, END_DATE).strftime("%Y-%m-%d"),
        })
    df = pd.DataFrame(records)
    df = inject_noise(df, "bmi", null_rate=0.05)
    df = inject_noise(df, "state", null_rate=0.02)
    df = inject_duplicates(df, n=40)
    bad_idx = np.random.choice(df.index, 10, replace=False)
    df.loc[bad_idx, "age"] = np.random.choice([-1, 0, 150, 999], size=10)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

def generate_claims(patients_df, n):
    valid_ids = patients_df["patient_id"].unique().tolist()
    high_risk = patients_df[(patients_df["has_diabetes"]==1)|(patients_df["has_hypertension"]==1)]["patient_id"].tolist()
    pid_pool = valid_ids + high_risk * 3
    records = []
    for i in range(n):
        pid = random.choice(pid_pool)
        adm_date = random_date(START_DATE, END_DATE)
        los = max(0, int(np.random.exponential(scale=3.5)))
        dis_date = adm_date + timedelta(days=los)
        cpt = random.choice(CPT_LIST)
        charge = round(CPT_CODES[cpt] * np.random.uniform(0.85, 1.30), 2)
        payer = np.random.choice(PAYER_TYPES, p=PAYER_WEIGHTS)
        payer_rate = {"Medicare":0.80,"Medicaid":0.65,"BlueCross":0.87,"Aetna":0.85,"UnitedHealth":0.88,"Cigna":0.84,"Self-Pay":0.35}
        reimbursed = round(charge * payer_rate[payer] * np.random.uniform(0.90, 1.05), 2)
        records.append({
            "claim_id": f"CLM{i+1:06d}",
            "patient_id": pid,
            "department": random.choice(DEPARTMENTS),
            "icd10_primary": random.choice(ICD10_LIST),
            "cpt_code": cpt,
            "admission_date": adm_date.strftime("%Y-%m-%d"),
            "discharge_date": dis_date.strftime("%Y-%m-%d"),
            "length_of_stay": los,
            "payer_type": payer,
            "charge_amount": charge,
            "reimbursed_amt": reimbursed,
            "claim_denied": int(np.random.random() < 0.08),
            "readmit_30day": int(np.random.random() < (0.08 + 0.04*int(los > 5) + 0.05*int(payer=="Medicaid"))),
            "attending_dept": random.choice(DEPARTMENTS),
        })
    df = pd.DataFrame(records)
    df = inject_noise(df, "icd10_primary", null_rate=0.03)
    df = inject_noise(df, "length_of_stay", null_rate=0.04)
    df["charge_amount"] = inject_outliers(df["charge_amount"], n=30)
    df = inject_duplicates(df, n=80)
    bad_idx = np.random.choice(df.index, 15, replace=False)
    df.loc[bad_idx, "discharge_date"] = df.loc[bad_idx, "admission_date"].apply(
        lambda d: (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=random.randint(1,5))).strftime("%Y-%m-%d") if isinstance(d, str) else d)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

def generate_billing(claims_df):
    records = []
    for _, row in claims_df.iterrows():
        if row["claim_denied"] == 1:
            status = random.choice(["Denied","Appeal Pending","Written Off"])
        else:
            status = random.choices(["Paid","Partial Pay","Pending","Written Off"], weights=[0.65,0.18,0.12,0.05])[0]
        records.append({
            "claim_id": row["claim_id"],
            "billing_status": status,
            "payment_delay_days": int(np.random.exponential(scale=22)) if status in ["Paid","Partial Pay"] else None,
            "write_off_amount": round(float(row["charge_amount"]) * np.random.uniform(0.05, 0.40), 2) if status == "Written Off" else 0.0,
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    print("Generating patients...")
    patients = generate_patients(N_PATIENTS)
    patients.to_csv("data/raw/patients.csv", index=False)
    print(f"  Done: {len(patients):,} rows")
    print("Generating claims...")
    claims = generate_claims(patients, N_CLAIMS)
    claims.to_csv("data/raw/claims.csv", index=False)
    print(f"  Done: {len(claims):,} rows")
    print("Generating billing...")
    billing = generate_billing(claims)
    billing.to_csv("data/raw/billing.csv", index=False)
    print(f"  Done: {len(billing):,} rows")
    print("\nAll raw data generated! Run src/clean_pipeline.py next.")
