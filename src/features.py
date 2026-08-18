import pandas as pd
import numpy as np
from typing import Optional, List

# Constants
REFERENCE_DATE = pd.Timestamp("2026-08-18")

PLAN_ORDER = {"Basic": 0, "Standard": 1, "Professional": 2, "Enterprise": 3}

# All language codes seen in both datasets — used to align one-hot columns
ALL_LANGUAGES = ["de", "en", "es", "fr", "nl", "pt"]

# Columns that are identifiers or targets — never used as model features
_NON_FEATURE_COLS = ["customer_id", "cancelled_flag", "churn_label"]

# The final ordered feature list (set after first training call).
FEATURE_COLUMNS: List[str] = []


def build_features(
    df: pd.DataFrame,
    days_since_login_median: Optional[float] = None,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Engineer features from a cleaned customer DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_prep.clean(). Must still contain signup_date,
        last_login_date, plan_type, preferred_language.
    days_since_login_median : float or None
        Median days_since_last_login to use for NaT imputation.
    fit : bool
        True  → compute and store the imputation median from this data.
        False → use the supplied days_since_login_median (inference mode).
    """
    global FEATURE_COLUMNS

    df = df.copy()

    # ── Temporal features ────────────────────────────────────────────────────
    df["account_tenure_days"] = (REFERENCE_DATE - df["signup_date"]).dt.days
    df["days_since_last_login"] = (REFERENCE_DATE - df["last_login_date"]).dt.days

    # Impute NaT-derived NaN with median (cust_0010, cust_0202 after cleaning)
    if fit:
        days_since_login_median = float(df["days_since_last_login"].median())

    if days_since_login_median is None:
        raise ValueError(
            "days_since_login_median must be provided when fit=False. "
            "Pass the value saved from the training call."
        )

    df["days_since_last_login"] = df["days_since_last_login"].fillna(days_since_login_median)

    # Sanity checks
    assert (df["account_tenure_days"] >= 0).all()
    assert (df["days_since_last_login"] >= 0).all()

    #  Categorical encoding

    # plan_type: ordinal
    df["plan_type_encoded"] = df["plan_type"].map(PLAN_ORDER)

    # preferred_language: one-hot
    lang_dummies = pd.get_dummies(df["preferred_language"], prefix="preferred_language", dtype=int)
    
    # Ensure all expected columns exist (fill 0 for languages not in this batch)
    for lang in ALL_LANGUAGES:
        col = f"preferred_language_{lang}"
        if col not in lang_dummies.columns:
            lang_dummies[col] = 0
            
    # drop_first equivalent: drop the reference category ('de')
    lang_dummies = lang_dummies.drop(columns=["preferred_language_de"], errors="ignore")
    df = pd.concat([df, lang_dummies], axis=1)

    # Drop raw columns (replaced by engineered features)

    # NOTE: We do not engineer extra ratios or date parts here anymore,
    # as ablation testing proved they cause overfitting on this small dataset.

    cols_to_drop = ["signup_date", "last_login_date", "plan_type", "preferred_language"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Build and validate final feature column list
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    if fit:
        FEATURE_COLUMNS = feature_cols
    else:
        # At inference: enforce column alignment with the training feature set
        if FEATURE_COLUMNS:
            for col in FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0  # add missing columns
            df = df[[c for c in _NON_FEATURE_COLS if c in df.columns] + FEATURE_COLUMNS]

    # Final NaN check
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        raise ValueError(f"NaN values remain in feature columns:\n{nan_counts[nan_counts > 0]}")

    return df, days_since_login_median


def get_feature_columns(df_feats: pd.DataFrame) -> List[str]:
    return [c for c in df_feats.columns if c not in _NON_FEATURE_COLS]

if __name__ == "__main__":
    from data_prep import clean_historical
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", default="data/raw/historical_customers_ds.csv")
    args = parser.parse_args()
    hist_clean = clean_historical(args.historical)
    hist_feats, median_dsll = build_features(hist_clean, fit=True)
    
    out_dir = "data/processed/scripts"
    os.makedirs(out_dir, exist_ok=True)
    hist_feats.to_csv(f"{out_dir}/historical_feats.csv", index=False)
    print(f"Saved feature dataset shape: {hist_feats.shape}")
