import pandas as pd
import numpy as np
import optuna
import joblib
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, average_precision_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

def optimize_lr(X, y):
    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2", "none"])
        solver = "lbfgs" if penalty == "none" else trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        
        model = LogisticRegression(
            C=C if penalty != "none" else 1.0,
            penalty=penalty if penalty != "none" else None,
            solver=solver,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_auc_scorer = make_scorer(average_precision_score, response_method="predict_proba")
        return cross_val_score(model, X, y, cv=cv, scoring=pr_auc_scorer).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params, study.best_value

def optimize_rf(X, y):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=42
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_auc_scorer = make_scorer(average_precision_score, response_method="predict_proba")
        return cross_val_score(model, X, y, cv=cv, scoring=pr_auc_scorer).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params, study.best_value

def optimize_xgb(X, y):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 5.0)
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_auc_scorer = make_scorer(average_precision_score, response_method="predict_proba")
        return cross_val_score(model, X, y, cv=cv, scoring=pr_auc_scorer).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params, study.best_value

def main():
    print("=" * 50)
    print("=> Costumer Churn Risk Tool - Training Pipeline")
    print("=" * 50)
    
    print("\n[1/4] Loading pre-processed data...")
    feats_path = "data/processed/scripts/historical_feats.csv"
    clean_path = "data/processed/scripts/historical_clean.csv"
    
    if not os.path.exists(feats_path) or not os.path.exists(clean_path):
        print("Error: Pre-processed data not found.")
        print("Please run 'python src/data_prep.py' and 'python src/features.py' first.")
        return
        
    df_feats = pd.read_csv(feats_path)
    df_clean = pd.read_csv(clean_path, parse_dates=["signup_date", "last_login_date"])
    
    REFERENCE_DATE = pd.Timestamp("2026-08-18")
    dsll_median = float((REFERENCE_DATE - df_clean["last_login_date"]).dt.days.median())
    
    print("[2/4] Preparing features for modeling...")
    _NON_FEATURE_COLS = ["customer_id", "cancelled_flag", "churn_label"]
    feature_cols = [c for c in df_feats.columns if c not in _NON_FEATURE_COLS]
    
    X = df_feats[feature_cols]
    y = df_feats["cancelled_flag"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("[3/4] Tuning Models with Optuna (5-Fold CV)...")
    print("      -> Tuning Logistic Regression...")
    lr_params, lr_score = optimize_lr(X_scaled, y)
    print(f"         LR Best PR-AUC: {lr_score:.4f}")
    
    print("      -> Tuning Random Forest...")
    rf_params, rf_score = optimize_rf(X_scaled, y)
    print(f"         RF Best PR-AUC: {rf_score:.4f}")
    
    print("      -> Tuning XGBoost...")
    xgb_params, xgb_score = optimize_xgb(X_scaled, y)
    print(f"         XGB Best PR-AUC: {xgb_score:.4f}")
    
    best_name = "Logistic Regression"
    best_score = lr_score
    best_params = lr_params
    
    if rf_score > best_score:
        best_name = "Random Forest"
        best_score = rf_score
        best_params = rf_params
    if xgb_score > best_score:
        best_name = "XGBoost"
        best_score = xgb_score
        best_params = xgb_params
        
    print(f"\n      [!] Best Model is {best_name} with PR-AUC {best_score:.4f}")
    
    print("[4/4] Training final model and exporting artifact...")
    if best_name == "Logistic Regression":
        if best_params.get("penalty") == "none":
            final_model = LogisticRegression(penalty=None, solver=best_params["solver"], class_weight="balanced", max_iter=1000, random_state=42)
        else:
            final_model = LogisticRegression(C=best_params["C"], penalty=best_params["penalty"], solver=best_params["solver"], class_weight="balanced", max_iter=1000, random_state=42)
    elif best_name == "Random Forest":
        final_model = RandomForestClassifier(**best_params, class_weight="balanced", random_state=42)
    else:
        final_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric="logloss")
        
    final_model.fit(X_scaled, y)
    
    artifact = {
        "model": final_model,
        "scaler": scaler,
        "dsll_median": dsll_median,
        "feature_cols": feature_cols
    }
    
    os.makedirs("models", exist_ok=True)
    artifact_path = "models/churn_model.joblib"
    joblib.dump(artifact, artifact_path)
    
    print("\n[OK] Training complete!")
    print(f"[*] Model Artifact saved to: {artifact_path}")
    print("   Contains: 'model', 'scaler', 'dsll_median', 'feature_cols'")
    print("=" * 50)

if __name__ == "__main__":
    main()
