import argparse
import os

import pandas as pd
import numpy as np

# Fixed reference date (keeps results reproducible)
REFERENCE_DATE = pd.Timestamp("2026-08-18")

# Columns that must be non-negative — anything below 0 is invalid
_NON_NEGATIVE_COLS = ["monthly_spend", "num_logins_30d", "support_tickets_30d"]

# Numeric columns that will be imputed with plan-type median when missing
_IMPUTE_COLS = ["monthly_spend", "num_logins_30d", "support_tickets_30d"]


def load_raw(filepath: str) -> pd.DataFrame:
    
    df = pd.read_csv(
        filepath,
        parse_dates=["signup_date", "last_login_date"],
    )
    return df


def clean(df: pd.DataFrame, is_historical: bool = True, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    original_shape = df.shape

    # Step 1: Drop internal_import_batch

    # System artifact; no causal or correlational value for churn prediction.
    if "internal_import_batch" in df.columns:
        df = df.drop(columns=["internal_import_batch"])
        if verbose:
            print("[1] Dropped 'internal_import_batch' (system artifact).")

    # Step 2: Drop exact duplicate rows

    # 3 pairs in historical (cust_0017, cust_0075, cust_0184) — import errors.
    # Duplicated on all columns except customer_id (which is the identifier).
    cols_for_dup = [c for c in df.columns if c != "customer_id"]
    before = len(df)
    df = df.drop_duplicates(subset=cols_for_dup, keep="first")
    n_dropped = before - len(df)
    if verbose:
        print(f"[2] Removed {n_dropped} exact duplicate row(s). Shape: {df.shape}")

    # Step 3: Drop rows with future signup_date

    # cust_0140: signup_date = 2027-02-14 (impossible; also causes login < signup).
    # No reliable way to recover the true date → drop the row.
    future_signup = df["signup_date"] > REFERENCE_DATE
    n_future = future_signup.sum()
    if n_future > 0:
        if verbose:
            ids = df.loc[future_signup, "customer_id"].tolist()
            print(f"[3] Dropped {n_future} row(s) with future signup_date: {ids}")
        df = df[~future_signup]

    # Step 4: Set future last_login_date to NaT

    # cust_0202: last_login_date = 2026-12-01 (4 months in the future).
    # Rest of the record is valid → neutralise just the bad date.
    # days_since_last_login will be NaN and imputed in feature engineering.
    future_login = df["last_login_date"] > REFERENCE_DATE
    n_future_login = future_login.sum()
    if n_future_login > 0:
        if verbose:
            ids = df.loc[future_login, "customer_id"].tolist()
            print(f"[4] Set {n_future_login} future last_login_date(s) to NaT: {ids}")
        df.loc[future_login, "last_login_date"] = pd.NaT

    # Step 5: Fix last_login_date < signup_date

    # cust_0166 (historical): login 20 days before signup — likely timezone /
    # data-entry error. Conservative fix: set last_login = signup_date.
    both_present = df["last_login_date"].notna() & df["signup_date"].notna()
    login_before = both_present & (df["last_login_date"] < df["signup_date"])
    n_fixed = login_before.sum()
    if n_fixed > 0:
        if verbose:
            ids = df.loc[login_before, "customer_id"].tolist()
            print(f"[5] Fixed {n_fixed} row(s) where last_login < signup -> set login = signup: {ids}")
        df.loc[login_before, "last_login_date"] = df.loc[login_before, "signup_date"]

    # Step 6: Set negative monthly_spend to NaN

    # cust_0118 (−49.00) and cust_0223 (−9.99): impossible subscription values.
    # Likely refunds or data-entry errors. Set to NaN → imputed below.
    for col in _NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        neg_mask = df[col] < 0
        n_neg = neg_mask.sum()
        if n_neg > 0:
            if verbose:
                ids = df.loc[neg_mask, "customer_id"].tolist()
                print(f"[6] Set {n_neg} negative {col} value(s) to NaN (will impute): {ids}")
            df.loc[neg_mask, col] = np.nan

    # Step 7: Impute missing numerics with plan-type median

    # Affected columns / rows:
    #   monthly_spend      → cust_0059 (original NaN) + cust_0118, cust_0223 (set above)
    #   num_logins_30d     → cust_0032
    #   support_tickets_30d→ cust_0093
    # Plan-type median is more accurate than global median because spend and
    # activity differ significantly across tiers (Basic vs Enterprise).
    for col in _IMPUTE_COLS:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            plan_medians = df.groupby("plan_type")[col].transform("median")
            df[col] = df[col].fillna(plan_medians)
            if verbose:
                print(f"[7] Imputed {n_missing} missing {col} value(s) with plan-type median.")

    # Audit assertions

    # last_login_date may still have NaT (cust_0010 original NaT, cust_0202 set above).
    # These become NaN in days_since_last_login and are handled in feature engineering.
    assert (df["monthly_spend"] >= 0).all(), "Negative monthly_spend remaining!"
    assert df["monthly_spend"].isna().sum() == 0, "NaN monthly_spend remaining!"
    assert df["num_logins_30d"].isna().sum() == 0, "NaN num_logins_30d remaining!"
    assert df["support_tickets_30d"].isna().sum() == 0, "NaN support_tickets_30d remaining!"
    assert (df["signup_date"] <= REFERENCE_DATE).all(), "Future signup_date remaining!"

    if verbose:
        n_nat_login = df["last_login_date"].isna().sum()
        print(
            f"\n=== Cleaning complete ===\n"
            f"   Original shape : {original_shape}\n"
            f"   Cleaned shape  : {df.shape}\n"
            f"   Rows removed   : {original_shape[0] - df.shape[0]}\n"
            f"   Cols removed   : {original_shape[1] - df.shape[1]}\n"
            f"   NaT login dates: {n_nat_login} (imputed in feature engineering)\n"
            f"   All assertions : PASSED\n"
        )

    return df


def clean_historical(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """Load and clean the historical training CSV."""
    df = load_raw(filepath)
    return clean(df, is_historical=True, verbose=verbose)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean raw customer CSVs.")
    parser.add_argument("--historical", default="data/raw/historical_customers_ds.csv")
    parser.add_argument("--out-dir", default="data/processed/scripts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== Cleaning historical dataset ===")
    hist_clean = clean_historical(args.historical)
    out_hist = os.path.join(args.out_dir, "historical_clean.csv")
    hist_clean.to_csv(out_hist, index=False)
    print(f"Saved: {out_hist}")