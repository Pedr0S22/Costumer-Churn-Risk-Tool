import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np

from data_prep import clean
from features import build_features

# Plain-language mapping for feature values that push the score up.
# For Logistic Regression, a positive contribution comes from:
# - Positive coefficient * Above-average value (scaled > 0)
# - Negative coefficient * Below-average value (scaled < 0)
REASON_MAPPING = {
    'monthly_spend': {
        'high': "High monthly spend",
        'low': "Low monthly spend"
    },
    'num_logins_30d': {
        'high': "High recent login activity",
        'low': "Low recent login activity"
    },
    'support_tickets_30d': {
        'high': "High support-ticket volume",
        'low': "Low support-ticket volume"
    },
    'account_tenure_days': {
        'high': "Long account tenure",
        'low': "Short account tenure"
    },
    'days_since_last_login': {
        'high': "Long time since last login",
        'low': "Recent login activity"
    },
    'plan_type_encoded': {
        'high': "High-tier plan",
        'low': "Low-tier plan"
    },
    'preferred_language_en': {'high': "English language preference", 'low': "Non-English language preference"},
    'preferred_language_es': {'high': "Spanish language preference", 'low': "Non-Spanish language preference"},
    'preferred_language_fr': {'high': "French language preference", 'low': "Non-French language preference"},
    'preferred_language_nl': {'high': "Dutch language preference", 'low': "Non-Dutch language preference"},
    'preferred_language_pt': {'high': "Portuguese language preference", 'low': "Non-Portuguese language preference"}
}

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["signup_date", "last_login_date"])
    return df

def get_top_reasons(model, row_scaled, feature_names):
    """
    Extract up to 3 plain-language reasons why a customer has a high churn risk.
    """
    # If the model is Logistic Regression, use coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        contributions = row_scaled * coefs
    else:
        # If the model is a Tree ensemble (RF / XGBoost), use SHAP values
        import shap
        explainer = shap.TreeExplainer(model)
        # SHAP expects 2D array, row_scaled is 1D
        shap_values = explainer.shap_values(row_scaled.reshape(1, -1))
        
        # RF returns a list of arrays (one for each class). XGBoost returns a single array.
        if isinstance(shap_values, list):
            contributions = shap_values[1][0]
        else:
            contributions = shap_values[0]
            
    # Sort indices by highest positive contribution (pushing towards Churn)
    sorted_idx = np.argsort(contributions)[::-1]
    
    reasons = []
    for idx in sorted_idx:
        if len(reasons) >= 3:
            break
            
        # Only include features that actively push the score up (positive contribution)
        if contributions[idx] <= 0:
            continue
            
        feat_name = feature_names[idx]
        val_scaled = row_scaled[idx]
        
        # If val_scaled > 0, the feature value was above the training average ("high")
        # If val_scaled < 0, the feature value was below the training average ("low")
        direction = "high" if val_scaled > 0 else "low"
        
        reason = REASON_MAPPING.get(feat_name, {}).get(direction, f"{direction.capitalize()} {feat_name}")
        reasons.append(reason)
        
    return reasons

def main():
    parser = argparse.ArgumentParser(description="Score current customers for churn risk.")
    parser.add_argument("--input", default="data/raw/current_customers_ds.csv")
    parser.add_argument("--model", default="models/churn_model_v1.joblib")
    parser.add_argument("--output", default="data/output/scores")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format: json or csv")
    args = parser.parse_args()
    
    print(f"[*] Loading artifact from {args.model}...")
    artifact = joblib.load(args.model)
    model = artifact["model"]
    scaler = artifact["scaler"]
    dsll_median = artifact["dsll_median"]
    feature_cols = artifact["feature_cols"]
    
    print(f"[*] Loading data from {args.input}...")
    df_raw = load_data(args.input)
    customer_ids = df_raw["customer_id"].copy()
    
    print("[*] Cleaning data...")
    df_clean = clean(df_raw, is_historical=False, verbose=False)
    
    print("[*] Building features...")
    df_feats, _ = build_features(df_clean, days_since_login_median=dsll_median, fit=False)
    
    # Ensure exact column match with training
    X = df_feats[feature_cols]
    
    print("[*] Scaling features...")
    X_scaled = scaler.transform(X)
    
    print("[*] Predicting probabilities and generating reasons...")
    # Get probabilities for class 1 (Churn)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # Extract coefficients if available, else None
    coefs = getattr(model, "coef_", [None])[0]
    
    results = []
    for i in range(len(df_raw)):
        cust_id = customer_ids.iloc[i]
        prob = probs[i]
        
        # Convert prob to 1-100 score
        # max(1, ...) ensures 0.0 prob still yields score of 1 instead of 0
        score = max(1, int(round(prob * 100)))
        
        # Assign categories
        if score <= 33:
            category = "Low"
        elif score <= 66:
            category = "Medium"
        else:
            category = "High"
            
        # Extract top 3 reasons based on Log-Odds contribution or SHAP
        reasons = get_top_reasons(model, X_scaled[i], feature_cols)
        
        results.append({
            "customer_id": str(cust_id),
            "risk_score": score,
            "risk_category": category,
            "top_reasons": reasons
        })
        
    out_path = args.output
    if args.format == "json" and not out_path.endswith(".json"):
        out_path += ".json"
    elif args.format == "csv" and not out_path.endswith(".csv"):
        out_path += ".csv"
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if args.format == "json":
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        # Convert to DataFrame for CSV export
        df_out = pd.DataFrame(results)
        # Join the list of reasons into a single string separated by semicolons
        df_out["top_reasons"] = df_out["top_reasons"].apply(lambda x: "; ".join(x))
        df_out.to_csv(out_path, index=False)
        
    print(f"[OK] Successfully scored {len(results)} customers.")
    print(f"[*] Results saved to {out_path}")

if __name__ == "__main__":
    main()
