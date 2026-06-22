import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "reports"
MODELS_DIR    = "models"

def load_features(path):
    df = pd.read_csv(path, parse_dates=["admission_date","discharge_date"])
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    features = [
        "age", "bmi", "length_of_stay",
        "charge_amount", "reimbursed_amt", "revenue_gap", "collection_rate",
        "has_diabetes", "has_hypertension", "is_smoker", "is_high_risk",
        "claim_denied", "admission_month",
    ]
    cat_features = ["gender", "payer_type", "department", "bmi_category", "age_group"]

    df_model = df[features + cat_features + ["readmit_30day"]].copy()
    df_model = df_model.dropna(subset=["readmit_30day"])

    # Encode categoricals
    df_encoded = pd.get_dummies(df_model, columns=cat_features, drop_first=True)

    # Fill any remaining nulls with median
    for col in df_encoded.columns:
        if df_encoded[col].isna().any():
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

    X = df_encoded.drop(columns=["readmit_30day"])
    y = df_encoded["readmit_30day"].astype(int)

    print(f"Features: {X.shape[1]} | Target distribution: {y.value_counts().to_dict()}")
    return X, y

def evaluate_model(name, model, X_test, y_test, results):
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:,1]
    auc         = roc_auc_score(y_test, y_prob)
    report      = classification_report(y_test, y_pred, output_dict=True)
    cm          = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name] = {
        "auc_roc":   round(auc, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"], 4),
        "f1":        round(report["1"]["f1-score"], 4),
        "accuracy":  round(report["accuracy"], 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist()[::10], "tpr": tpr.tolist()[::10]},
    }

    print(f"\n  {name}")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Precision: {report['1']['precision']:.4f}")
    print(f"    Recall:    {report['1']['recall']:.4f}")
    print(f"    F1-Score:  {report['1']['f1-score']:.4f}")
    print(f"    Accuracy:  {report['accuracy']:.4f}")
    print(f"    Confusion Matrix:\n      {cm}")

def get_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        est = model.named_steps.get("clf") or list(model.named_steps.values())[-1]
        if hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
        elif hasattr(est, "coef_"):
            imp = np.abs(est.coef_[0])
        else:
            return []
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return []

    fi = pd.DataFrame({"feature": feature_names, "importance": imp})
    fi = fi.sort_values("importance", ascending=False).head(15)
    return fi.to_dict(orient="records")

def run_models():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading features...")
    X, y = load_features(f"{PROCESSED_DIR}/master.csv")

    print(f"\nSplitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }

    results = {}
    print("\nTraining and evaluating models...")
    print("="*55)

    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(X_train, y_train)
        evaluate_model(name, model, X_test, y_test, results)

    # Cross-validation on best model
    print("\n" + "="*55)
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (AUC={results[best_name]['auc_roc']})")

    print(f"\nRunning 5-fold cross-validation on {best_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  CV AUC scores: {[round(s,4) for s in cv_scores]}")
    print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    results[best_name]["cv_auc_mean"] = round(cv_scores.mean(), 4)
    results[best_name]["cv_auc_std"]  = round(cv_scores.std(), 4)

    # Feature importance
    print(f"\nTop 10 features ({best_name}):")
    fi = get_feature_importance(best_model, X.columns.tolist(), best_name)
    if fi:
        for i, f in enumerate(fi[:10], 1):
            print(f"  {i:>2}. {f['feature']:<40} {f['importance']:.4f}")
        results[best_name]["feature_importance"] = fi

    # Summary
    print("\n" + "="*55)
    print("MODEL COMPARISON SUMMARY")
    print("="*55)
    print(f"  {'Model':<25} {'AUC-ROC':>8} {'F1':>8} {'Recall':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in results.items():
        marker = " ← best" if name == best_name else ""
        print(f"  {name:<25} {r['auc_roc']:>8.4f} {r['f1']:>8.4f} {r['recall']:>8.4f}{marker}")

    # Save results
    with open(f"{REPORTS_DIR}/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {REPORTS_DIR}/model_results.json")

    # Save interview-ready summary
    with open(f"{REPORTS_DIR}/model_summary.txt", "w") as f:
        f.write("HealthPulse Analytics — Model Results\n")
        f.write("="*50 + "\n\n")
        best = results[best_name]
        f.write(f"Best Model:        {best_name}\n")
        f.write(f"AUC-ROC:           {best['auc_roc']}\n")
        f.write(f"CV AUC (5-fold):   {best.get('cv_auc_mean','N/A')} +/- {best.get('cv_auc_std','N/A')}\n")
        f.write(f"Precision:         {best['precision']}\n")
        f.write(f"Recall:            {best['recall']}\n")
        f.write(f"F1-Score:          {best['f1']}\n\n")
        f.write("Top Predictive Features:\n")
        for i, feat in enumerate(best.get("feature_importance", [])[:10], 1):
            f.write(f"  {i:>2}. {feat['feature']}\n")
    print(f"  Summary saved → {REPORTS_DIR}/model_summary.txt")
    print("\nWeek 2 COMPLETE! Next: Week 3 -> revenue analysis")

if __name__ == "__main__":
    run_models()
