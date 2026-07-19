# Customer Churn Prediction — Telco Dataset

Predicting which customers are likely to churn, and identifying the business levers that reduce churn risk, using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers).

## Business Problem

Customer acquisition is significantly more expensive than retention. This project builds a model to flag at-risk customers *before* they leave, so a business can proactively target them with retention offers — and identifies which structural factors (contract type, payment method, service add-ons) most strongly predict churn, to inform retention strategy beyond just individual predictions.

## Key Findings

Baseline churn rate across all customers: **26.5%**.

Five independent risk factors were identified, each pushing churn well above baseline:

| Driver | Highest-risk segment | Churn rate | vs. baseline |
|---|---|---|---|
| Contract type | Month-to-month | 42.7% | 1.6x |
| Internet service | Fiber optic | 41.9% | 1.6x |
| Tech support | No support | 41.6% | 1.6x |
| Payment method | Electronic check | 45.3% | 1.7x |
| Tenure | Low tenure (new customers) | avg. 18 mo. vs 37.6 mo. for retained customers | — |

**Risk factors compound.** Customers who are both month-to-month *and* pay by electronic check churn at **53.7%** — more than double the baseline, and higher than either factor alone. This suggests retention efforts should prioritize customers stacking multiple risk factors, not just one.

**Fiber optic is a paradox worth flagging to stakeholders:** it's the highest-revenue segment (avg. $91.50/month vs. $58.10 for DSL) *and* the highest-churn segment — likely reflecting price sensitivity or competitive pressure rather than service dissatisfaction alone. Worth further investigation before assuming it's a service-quality issue.

## Data Cleaning Notes

- `TotalCharges` was stored as text due to 11 blank-string values, all belonging to customers with 0 months tenure (new signups not yet billed). Imputed as 0, since this reflects the underlying reality rather than missing data.
- Six add-on service columns (`TechSupport`, `OnlineSecurity`, `StreamingTV`, etc.) each carried a redundant `"No internet service"` category that just re-stated a customer's `InternetService` status. Collapsed to plain Yes/No to avoid diluting feature importance across six repeated signals.
- Noted (but did not correct, as it reflects reality) a right-censoring artifact: 356 customers cluster at exactly 72 months tenure — the maximum in the dataset — since this appears to be a single snapshot in time. These customers haven't necessarily "stopped churning," they simply hadn't churned as of data collection.

## Modeling Approach

Three models were trained and compared fairly — each evaluated at its own optimal precision/recall-balanced threshold, not just a default 0.5 cutoff:

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **Logistic Regression** | **0.60** | **0.60** | **0.60** | **0.842** |
| Random Forest | 0.595 | 0.594 | 0.594 | 0.827 |
| XGBoost | 0.578 | 0.578 | 0.578 | 0.814 |

**Logistic Regression performed best across every metric.** This is a deliberately unglamorous but honest result: it suggests churn in this dataset is driven by largely independent, additive factors (contract length, payment method, tenure) rather than complex non-linear interactions that tree-based models are built to exploit. The added interpretability of a linear model is a genuine bonus, not just a consolation prize.

A key modeling decision: the target class is imbalanced (73.5%/26.5%), and **missing an actual churner is more costly to the business than a false alarm** (lost customer lifetime value vs. a low-cost retention offer). Class weighting and threshold tuning were used to explicitly optimize for this, rather than defaulting to accuracy or an untuned 0.5 cutoff.

`TotalCharges` was later dropped from the final feature set after diagnosing high multicollinearity with `tenure` (r=0.83) and `MonthlyCharges` (r=0.65) — its coefficient initially showed a counterintuitive sign, and removing it cost negligible AUC (0.842 → 0.839) while making every other coefficient more trustworthy.

## What Drives Churn (Model Coefficients)

Top churn-**reducing** factors:
- Two-year contract (strongest single effect in the model)
- Longer tenure
- No internet service (this segment has few dependencies to be dissatisfied with)
- One-year contract (roughly half the effect of a two-year contract)

Top churn-**increasing** factors:
- Fiber optic internet (strongest single risk factor)
- Electronic check payment
- Streaming TV / Movies add-ons
- Paperless billing

## Repository Structure

```
├── README.md
├── churn_analysis.ipynb      # Full EDA, feature engineering, and modeling notebook
└── requirements.txt
```

## Tools Used

Python, pandas, scikit-learn, XGBoost, matplotlib

## Next Steps

- Deploy as an interactive tool (Streamlit) for uploading customer lists and generating churn scores
- Explore survival analysis to predict *when* a customer is likely to churn, not just whether — accounting properly for the right-censoring identified during EDA
