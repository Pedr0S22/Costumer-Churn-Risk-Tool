# Customer Churn Risk Tool

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A reproducible churn-risk modelling pipeline that helps a retention team identify customers who may cancel their subscription. Built as part of the **RedLight Software — Data Science Internship Technical Challenge**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Setup \& Installation](#setup--installation)
- [Pipeline Execution](#pipeline-execution)
- [Data Quality \& Preparation](#data-quality--preparation)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation \& Validation](#model-evaluation--validation)
- [Risk Scoring Methodology](#risk-scoring-methodology)
- [Explainability (Top Reasons)](#explainability-top-reasons)
- [Error Analysis \& Limitations](#error-analysis--limitations)
- [Key Assumptions \& Trade-offs](#key-assumptions--trade-offs)
- [API Reference](#api-reference)
- [Output Format](#output-format)
- [Author \& License](#author--license)

---

## Project Overview

The goal is to build a **supervised classification model** using `cancelled_flag` from historical customer data, then **score every customer** in the current customer dataset with:

| Field            | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `risk_score`     | Integer 1–100 representing churn probability                                |
| `risk_category`  | **Low** (1–33), **Medium** (34–66), or **High** (67–100)                    |
| `top_reasons`    | Up to 3 ranked, plain-language reasons explaining the score                 |

The pipeline is fully reproducible via Docker or local Python commands, and also exposes a **FastAPI REST endpoint** for real-time single-customer scoring.

### Key Results

| Metric   | Baseline (Stratified Random) | Best Model (Logistic Regression) |
|----------|------------------------------|----------------------------------|
| PR-AUC   | ~0.26                        | **~0.78**                        |

> The best model was selected automatically via **Optuna** hyperparameter tuning across Logistic Regression, Random Forest, and XGBoost, evaluated with 5-fold stratified cross-validation on PR-AUC.

---

## Project Architecture

```
Costumer-Churn-Risk-Tool/
├── api/                          # FastAPI REST API (MVC pattern)
│   ├── main.py                   # App entry point & startup hooks
│   ├── routers/
│   │   └── scoring.py            # POST /costume-record endpoint
│   ├── schemas/
│   │   └── customer.py           # Pydantic request/response models
│   └── services/
│       └── scoring_service.py    # Model loading & single-customer inference
│
├── src/                          # Core ML pipeline modules
│   ├── data_prep.py              # Loading, cleaning, imputation, validation
│   ├── features.py               # Feature engineering & encoding
│   ├── train.py                  # Model selection, Optuna tuning, export
│   └── score.py                  # Batch scoring with SHAP/log-odds reasons
│
├── notebooks/                    # Exploratory analysis & development
│   ├── eda_data_quality.ipynb    # EDA & data-quality assessment
│   ├── clean_features_modeling.ipynb  # End-to-end modelling notebook
│   └── test_clean_features.ipynb      # Feature pipeline validation
│
├── data/
│   ├── raw/                      # Provided CSV files (unchanged)
│   │   ├── historical_customers_ds.csv   (244 rows)
│   │   └── current_customers_ds.csv      (81 rows)
│   ├── processed/                # Cleaned & feature-engineered datasets
│   │   ├── notebooks/            # Outputs from notebook exploration
│   │   └── scripts/              # Outputs from CLI pipeline
│   └── output/                   # Final scored results
│       ├── scores.json           # JSON array (primary deliverable)
│       └── scores.csv            # CSV (optional deliverable)
│
├── models/                       # Serialised model artifacts (.joblib)
│   └── churn-model_v1.joblib     # Best model + scaler + metadata
│
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Multi-service orchestration
├── pyproject.toml                # Python project metadata & dependencies
├── LICENSE                       # MIT License
└── README.md                     # ← You are here
```

---

## Setup & Installation

### Prerequisites

- **Python ≥ 3.11** (tested on 3.11)
- **pip** (comes with Python)
- **Docker & Docker Compose** *(optional, for containerised execution)*

### Option A — Local (pip install)

```bash
# 1. Clone the repository
git clone https://github.com/Pedr0S22/Costumer-Churn-Risk-Tool.git
cd Costumer-Churn-Risk-Tool

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install the package and all dependencies
pip install .
```

### Option B — Docker

```bash
# Build the image (all dependencies are installed inside the container)
docker compose build
```

> **Note:** The raw data files (`data/raw/`) are included in the repository. No additional data download is required.

---

## Pipeline Execution

The pipeline consists of three sequential stages: **clean → train → score**.

### Local Execution

```bash
# Step 1: Clean both datasets
python src/data_prep.py

# Step 2: Engineer features
python src/features.py

# Step 3: Train model (includes Optuna tuning ≈ 1–2 min)
python src/train.py

# Step 4: Score current customers (JSON output by default)
python src/score.py

# Or score with CSV output
python src/score.py --format csv
```

### Docker Execution

```bash
# Train the model (runs clean → features → train in sequence)
docker compose run --rm train

# Score current customers (JSON)
docker compose run --rm score

# Score current customers (CSV)
docker compose run --rm score-csv

# Start the REST API
docker compose up api
```

### CLI Arguments (score.py)

| Argument     | Default                                | Description                          |
|--------------|----------------------------------------|--------------------------------------|
| `--input`    | `data/raw/current_customers_ds.csv`    | Path to CSV of customers to score    |
| `--model`    | `models/churn-model_v1.joblib`         | Path to trained model artifact       |
| `--output`   | `data/output/scores`                   | Output path (extension auto-added)   |
| `--format`   | `json`                                 | Output format: `json` or `csv`       |

---

## Data Quality & Preparation

> **Module:** [`src/data_prep.py`](src/data_prep.py)  
> **Notebook:** [`notebooks/eda_data_quality.ipynb`](notebooks/eda_data_quality.ipynb)

The provided datasets intentionally contain realistic imperfections. Below is a complete catalogue of every issue found and how each was handled:

### Issues Found & Resolution

| #  | Issue                          | Affected Rows                        | Action      | Rationale                                                                                   |
|----|--------------------------------|--------------------------------------|-------------|---------------------------------------------------------------------------------------------|
| 1  | `internal_import_batch` column | All rows                             | **Dropped** | System artifact with no causal or correlational value for churn prediction.                  |
| 2  | Exact duplicate rows           | 3 pairs (cust_0017, cust_0075, cust_0184) | **Dropped** | Import errors — duplicated on all columns except `customer_id`. Kept first occurrence.       |
| 3  | Future `signup_date`           | cust_0140 (2027-02-14)               | **Dropped** | Impossible date; no reliable way to recover the true value. Row removed entirely.            |
| 4  | Future `last_login_date`       | cust_0202 (2026-12-01)               | **Set NaT** | Rest of the record is valid — neutralise only the bad date; imputed later in feature eng.    |
| 5  | `last_login` < `signup_date`   | cust_0166                            | **Corrected** | Login 20 days before signup — likely timezone/data-entry error. Set login = signup.         |
| 6  | Negative `monthly_spend`       | cust_0118 (−49.00), cust_0223 (−9.99) | **Set NaN** | Impossible subscription values (refunds or data-entry errors). Imputed below.               |
| 7  | Missing `monthly_spend`        | cust_0059 + above                    | **Imputed** | **Plan-type median** — more accurate than global median since spend differs across tiers.    |
| 8  | Missing `num_logins_30d`       | cust_0032                            | **Imputed** | Plan-type median.                                                                           |
| 9  | Missing `support_tickets_30d`  | cust_0093                            | **Imputed** | Plan-type median.                                                                           |
| 10 | Missing `last_login_date`      | cust_0010, cust_0202 (after fix)     | **Imputed** | `days_since_last_login` set to training median in feature engineering.                       |

### Validation Assertions

After cleaning, the pipeline asserts:
- No negative `monthly_spend` values remain.
- No `NaN` in `monthly_spend`, `num_logins_30d`, or `support_tickets_30d`.
- No `signup_date` in the future relative to the reference date (`2026-08-18`).

### Historical Dataset Summary

| Metric                  | Value              |
|-------------------------|--------------------|
| Raw rows                | 244                |
| Cleaned rows            | 240                |
| Rows removed            | 4 (3 duplicates + 1 future signup) |
| Columns removed         | 1 (`internal_import_batch`)        |
| NaT login dates         | 2 (handled in feature engineering) |

---

## Feature Engineering

> **Module:** [`src/features.py`](src/features.py)

All transformations are deterministic and documented. The same pipeline is applied identically to both training and scoring data to prevent train/serve skew.

### Features Used (9 total)

| Feature                    | Type        | Source                                          | Description                                                |
|----------------------------|-------------|------------------------------------------------|------------------------------------------------------------|
| `monthly_spend`            | Raw numeric | Raw data                                        | Customer's monthly subscription charge.                    |
| `num_logins_30d`           | Raw numeric | Raw data                                        | Number of logins in the last 30 days.                      |
| `support_tickets_30d`      | Raw numeric | Raw data                                        | Support tickets filed in the last 30 days.                 |
| `account_tenure_days`      | Engineered  | `REFERENCE_DATE − signup_date`                  | How long the customer has been subscribed (days).          |
| `days_since_last_login`    | Engineered  | `REFERENCE_DATE − last_login_date`              | Days of inactivity since last login.                       |
| `plan_type_encoded`        | Encoded     | Ordinal: Basic=0, Standard=1, Professional=2, Enterprise=3 | Subscription tier as ordinal integer.          |
| `preferred_language_en`    | One-hot     | `preferred_language`                            | Language indicator (reference: `de` dropped).              |
| `preferred_language_es`    | One-hot     | `preferred_language`                            | Language indicator.                                        |
| `preferred_language_fr`    | One-hot     | `preferred_language`                            | Language indicator.                                        |
| `preferred_language_nl`    | One-hot     | `preferred_language`                            | Language indicator.                                        |
| `preferred_language_pt`    | One-hot     | `preferred_language`                            | Language indicator.                                        |

### Feature Selection Rationale

Initial experiments with 18 features (including engineered ratios like `spend_per_login`, `tickets_per_tenure_month`, etc.) showed severe **overfitting** on this small dataset (~240 training rows). Ablation testing revealed:

- Tree models (Random Forest, XGBoost) suffered most — PR-AUC crashed to ~0.44.
- Stripping back to 9 core features yielded stable, generalizable performance.

This is documented in the modelling notebook (`clean_features_modeling.ipynb`).

### Reference Date

A fixed reference date of **`2026-08-18`** is used for all temporal feature calculations to ensure full reproducibility.

---

## Model Evaluation & Validation

> **Module:** [`src/train.py`](src/train.py)  
> **Notebook:** [`notebooks/clean_features_modeling.ipynb`](notebooks/clean_features_modeling.ipynb)

### Model Selection Process

Three model families were evaluated using **Optuna** hyperparameter tuning (30 trials each) with **5-fold stratified cross-validation**:

| Model                | PR-AUC (CV)   | Notes                                                |
|----------------------|---------------|------------------------------------------------------|
| Baseline (stratified random) | ~0.26  | Random predictions proportional to class distribution |
| Logistic Regression  | **~0.78**     | **Best.** Stable on small data. `class_weight=balanced` |
| Random Forest        | ~0.65–0.72    | Prone to overfitting on 240 rows                     |
| XGBoost              | ~0.60–0.70    | Also overfit on small data; needs more training data  |

### Why PR-AUC?

With a **~25.9% churn rate** (moderate class imbalance), PR-AUC is more informative than ROC-AUC because:

- ROC-AUC can be **overly optimistic** when the negative class dominates.
- PR-AUC focuses on the model's ability to correctly identify the **minority class** (churned customers), which is exactly what the retention team cares about.

### Why Logistic Regression Won

On a dataset with only ~240 training rows and 9 features:

1. **Low variance** — linear models generalise better with limited data.
2. **`class_weight='balanced'`** — automatically upweights the minority class.
3. **Interpretable coefficients** — enables direct log-odds decomposition for explainability (see [Explainability](#explainability-top-reasons)).
4. **No overfitting** — unlike tree ensembles which memorised noise on this dataset.

### Baseline Comparison

The stratified random baseline achieves PR-AUC ≈ 0.26 (roughly the churn prevalence). The selected model achieves PR-AUC ≈ 0.78 — a **3× improvement** over chance, confirming the model captures genuine signal.

### Scaling

`StandardScaler` is fitted on training data and applied identically at scoring time. The scaler is persisted inside the model artifact to prevent train/serve skew.

---

## Risk Scoring Methodology

### Probability → Risk Score Conversion

The model outputs a probability `p ∈ [0, 1]` representing the likelihood of churn. This is converted to an integer score:

```
risk_score = max(1, round(p × 100))
```

| Probability | Risk Score | Interpretation                     |
|-------------|------------|-------------------------------------|
| 0.00        | 1          | Virtually no churn risk             |
| 0.33        | 33         | Upper bound of Low                  |
| 0.50        | 50         | Coin-flip — Medium risk             |
| 0.67        | 67         | Lower bound of High                 |
| 1.00        | 100        | Near-certain churn                  |

### Risk Categories

| Category   | Score Range | Interpretation                                        |
|------------|-------------|-------------------------------------------------------|
| **Low**    | 1 – 33     | Healthy customer; no immediate action needed.          |
| **Medium** | 34 – 66    | At-risk; consider proactive engagement.                |
| **High**   | 67 – 100   | Likely to churn; immediate retention intervention.     |

### Design Choice

The `round(p × 100)` mapping is a **direct linear transformation** that preserves the model's rank ordering. This means:

- If Customer A has a higher model probability than Customer B, A will always have a higher risk score.
- The retention team can prioritise customers by score and get the same ordering as the model's internal ranking.

---

## Explainability (Top Reasons)

> **Requirement:** Provide up to 3 ranked, plain-language reasons for each customer's score.

### Method

The explainability approach adapts to the model type:

- **Logistic Regression** (current best model): Uses **log-odds coefficient decomposition**. For each feature, the contribution is `coefficient × scaled_value`. Features with the highest positive contributions (pushing towards churn) are selected as the top reasons.

- **Tree-based models** (Random Forest / XGBoost): Uses **SHAP (TreeExplainer)** values, which measure each feature's marginal contribution to the prediction.

For each customer, the pipeline:

1. Computes per-feature contributions (coefficient × scaled value, or SHAP values).
2. Ranks features by **absolute contribution magnitude** (only positive contributions — those pushing towards churn).
3. Selects the top 3.
4. Translates each into a **plain-language sentence** using a human-readable reason mapping.

### Reason Mapping Examples

| Feature                  | Direction | Plain-Language Reason               |
|--------------------------|-----------|--------------------------------------|
| `support_tickets_30d`    | High      | "High support-ticket volume"        |
| `num_logins_30d`         | Low       | "Low recent login activity"         |
| `days_since_last_login`  | High      | "Long time since last login"        |
| `monthly_spend`          | Low       | "Low monthly spend"                 |
| `account_tenure_days`    | Low       | "Short account tenure"              |
| `plan_type_encoded`      | Low       | "Low-tier plan"                     |
| `preferred_language_en`  | Low       | "Non-English language preference"   |

### Limitations

1. **Feature-level, not causal**: The reasons explain which features contribute most to the **model's prediction**, not necessarily what **caused** the customer to consider leaving. Correlation ≠ causation.

2. **Linear decomposition is additive**: For logistic regression, the log-odds decomposition assumes features contribute independently. Interaction effects are not captured (though for a linear model, there are none by construction).

3. **Fewer than 3 reasons possible**: If fewer than 3 features have positive contributions towards churn, the output will contain fewer reasons. This typically happens for low-risk customers.

4. **Scaled-value direction heuristic**: The "high" / "low" label is determined by whether the feature's scaled value is above or below the training mean (scaled value > 0 vs < 0). This is a simplification; the actual distribution may be skewed.

5. **Language features may appear as reasons**: One-hot language indicators can surface as top contributors (e.g., "Non-Dutch language preference"). While statistically valid, these may reflect **data collection patterns** rather than actionable churn drivers.

---

## Error Analysis & Limitations

### Dataset Constraints

- **Small sample size** (~240 historical, ~81 current): Limits the complexity of learnable patterns and makes evaluation metrics noisier.
- **No temporal validation**: Without a time-based split, we cannot assess how well the model generalises to future time periods (concept drift).
- **Single snapshot**: The data represents a single cross-section. Behavioural trends over time (e.g., declining logins) would be more predictive but are not available.

### Observed Error Patterns

1. **Medium-risk cluster**: Many current customers score in the 49–51 range. This is expected for a linear model on a small dataset — the model is appropriately uncertain when features do not strongly signal either outcome.

2. **Language as signal**: Language preferences appear frequently in top reasons. This may be a proxy for regional service quality or market maturity, but could also be noise. The retention team should interpret these with caution.

3. **Overfitting with complex models**: XGBoost and Random Forest overfit on 18 features, achieving high training accuracy but poor CV performance. The solution was to reduce to 9 features and select a simpler model (Logistic Regression).

### What Would Improve the Model

| Improvement                        | Expected Impact                                           |
|------------------------------------|-----------------------------------------------------------|
| More training data (1000+ rows)    | Enable tree ensembles to outperform LR                    |
| Temporal/behavioural features      | Login trends, spend changes over time                     |
| Time-based train/test split        | Better estimate of real-world performance                 |
| Calibration analysis (Platt/isotonic) | Ensure probabilities match true churn rates            |
| Feature interactions               | Capture non-linear patterns (e.g., low spend + high tickets) |

---

## Key Assumptions & Trade-offs

### Assumptions

1. **Reference date is `2026-08-18`**: All temporal features are computed relative to this fixed date for reproducibility.
2. **Missing logins ≈ no activity**: Missing `last_login_date` is imputed with the training median of `days_since_last_login`, which is a conservative assumption.
3. **Plan-type median is the best imputation**: Missing numeric values are imputed using the median of the customer's plan type, which accounts for tier-level differences (e.g., Enterprise customers spend more than Basic).
4. **Negative spend = data error**: Negative `monthly_spend` values are treated as invalid (likely refunds or data-entry errors), not as legitimate signals.
5. **Labels are trustworthy**: The `cancelled_flag` in historical data is assumed to be a reliable ground truth.

### Trade-offs

| Decision                         | Pro                                        | Con                                                     |
|----------------------------------|--------------------------------------------|---------------------------------------------------------|
| Logistic Regression over XGBoost | Stable on small data; interpretable        | Cannot capture non-linear interactions                  |
| PR-AUC as primary metric         | Focuses on minority class (churners)       | Not as intuitive to stakeholders as accuracy            |
| 9 features (reduced from 18)     | Prevents overfitting                       | May miss subtle signals in dropped features             |
| Plan-type median imputation      | Tier-aware; more accurate                  | Assumes tier is the most relevant grouping variable     |
| Dropping rows with future signup | Clean data; no guessing                    | Loses 1 training example                                |
| `round(p × 100)` score mapping   | Simple, rank-preserving, easy to explain   | Not calibrated — score of 50 ≠ exactly 50% churn prob  |
| Optuna tuning (30 trials)        | Automated, fair comparison across models   | More trials might find better hyperparameters           |

---

## API Reference

The project includes a **FastAPI** REST API for real-time single-customer scoring.

### Start the API

```bash
# Local
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Docker
docker compose up api
```

### Endpoint

```
POST /costume-record
```

### Request Body

```json
{
  "customer_id": "cust_1234",
  "signup_date": "2024-06-15",
  "last_login_date": "2026-08-10",
  "plan_type": "Standard",
  "monthly_spend": 49.99,
  "num_logins_30d": 8,
  "support_tickets_30d": 3,
  "preferred_language": "en"
}
```

### Response

```json
{
  "customer_id": "cust_1234",
  "risk_score": 62,
  "risk_category": "Medium",
  "top_reasons": [
    "High support-ticket volume",
    "Low recent login activity",
    "Low monthly spend"
  ]
}
```

### Interactive Docs

Once the API is running, visit:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
---

## Output Format

### JSON (Primary Deliverable)

**File:** `data/output/scores.json`

```json
[
  {
    "customer_id": "cust_1001",
    "risk_score": 49,
    "risk_category": "Medium",
    "top_reasons": [
      "Non-English language preference",
      "Non-Spanish language preference"
    ]
  },
  {
    "customer_id": "cust_1002",
    "risk_score": 51,
    "risk_category": "Medium",
    "top_reasons": [
      "Low recent login activity",
      "Long time since last login",
      "Non-Dutch language preference"
    ]
  }
]
```

### CSV (Optional Deliverable)

**File:** `data/output/scores.csv`

```csv
customer_id,risk_score,risk_category,top_reasons
cust_1001,49,Medium,Non-English language preference; Non-Spanish language preference
cust_1002,51,Medium,Low recent login activity; Long time since last login; Non-Dutch language preference
```

All **81 current customers** have been scored. Results are stored in `data/output/`.

---

## Author & License

**Author:** Pedro Silva
**License:** [MIT](LICENSE)

Built for the **RedLight Software — Data Science Internship Technical Challenge**.

---