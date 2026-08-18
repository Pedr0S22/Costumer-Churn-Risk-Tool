import pandas as pd
import joblib
import os
import sys

# Add the src folder to the path so we can import the ML modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data_prep import clean
from features import build_features
from score import get_top_reasons
from api.schemas.customer import CustomerRecord

class ScoringService:
    def __init__(self):
        self.artifact = None

    def load_model(self):
        model_path = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "../../models/churn_model_v1.joblib"))
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Please run train.py first.")
            return
        self.artifact = joblib.load(model_path)

    def score_customer(self, record: CustomerRecord) -> dict:
        if self.artifact is None:
            raise ValueError("Model artifact not found. Please train the model first.")
            
        model = self.artifact["model"]
        scaler = self.artifact["scaler"]
        dsll_median = self.artifact["dsll_median"]
        feature_cols = self.artifact["feature_cols"]
        
        # 1. Convert incoming record to DataFrame
        df_raw = pd.DataFrame([record.model_dump()])
        
        # Ensure dates are parsed correctly
        df_raw["signup_date"] = pd.to_datetime(df_raw["signup_date"], errors="coerce")
        df_raw["last_login_date"] = pd.to_datetime(df_raw["last_login_date"], errors="coerce")
        
        # 2. Clean and build features
        df_clean = clean(df_raw, is_historical=False, verbose=False)
        df_feats, _ = build_features(df_clean, days_since_login_median=dsll_median, fit=False)
        
        # 3. Ensure exact column match with training and Scale
        X = df_feats[feature_cols]
        X_scaled = scaler.transform(X)
        
        # 4. Predict
        prob = model.predict_proba(X_scaled)[0, 1]
        
        # 5. Calculate Score and Categories
        score = max(1, int(round(prob * 100)))
        if score <= 33:
            category = "Low"
        elif score <= 66:
            category = "Medium"
        else:
            category = "High"
            
        # 6. Extract Dynamic Top Reasons using SHAP or log-odds
        reasons = get_top_reasons(model, X_scaled[0], feature_cols)
        
        return {
            "customer_id": record.customer_id,
            "risk_score": score,
            "risk_category": category,
            "top_reasons": reasons
        }

# Singleton instance
scoring_service = ScoringService()
