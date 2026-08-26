# BankLens — Retail Banking Performance Intelligence & Risk Analytics Platform

> Early warning stress detection across 4,000+ FDIC-insured institutions using logistic regression on call report signals, SARIMAX macro-factor forecasting, and a Power BI executive dashboard with DAX measures surfacing institution risk tiers and leading indicators.

---

## Abstract

BankLens addresses a core challenge in banking analytics: detecting institutional stress before it becomes visible in headlines. Three questions are answered empirically using FDIC public call report data:

1. **Can a composite stress score reliably classify institutions entering distress 2+ quarters before peak stress?** Logistic regression on NPA ratio change, deposit outflow rate, loan-to-deposit ratio change, and net interest margin compression — validated by TimeSeriesSplit CV ROC-AUC and early detection rate against known stress patterns.

2. **Do macro factors (Fed funds rate, yield curve slope) improve deposit outflow forecasting?** SARIMAX with FRED macro inputs vs SARIMA baseline — A/B tested by MAPE on 4-quarter holdout.

3. **How do these signals surface for executive decision-making?** Power BI dashboard with three report pages, DAX measures for stress scoring and early warning flags, and conditional formatting for immediate visual risk identification.

---

## System Formulation

**Composite Stress Score:**
$$S_t = w_1 \cdot \Delta NPA_t + w_2 \cdot \Delta Deposit_t + w_3 \cdot \Delta LDR_t + w_4 \cdot \Delta NIM_t$$

Weights estimated via logistic regression coefficients, normalized 0-100.

**Logistic Regression Early Warning:**
$$P(stress=1|X) = \sigma(\beta_0 + \beta_1 \Delta NPA + \beta_2 Deposit_{growth} + \beta_3 \Delta LDR + \beta_4 \Delta NIM + \epsilon)$$

**SARIMAX Macro-Factor Forecast:**
$$y_t = \phi y_{t-1} + \theta \epsilon_{t-1} + \gamma_1 \cdot FedFunds_t + \gamma_2 \cdot YieldCurve_t + \epsilon_t$$

---

## Architecture

```
FDIC BankFind API (free, no auth)    FRED API (free)
      │                                    │
      ▼                                    ▼
┌─────────────────────────────────────────────────┐
│              Data Engineering Layer              │
│   Quarterly panel: NPA, deposits, LDR, NIM      │
│   dbt transformation models + schema tests      │
│   PostgreSQL storage + audit-ready lineage      │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │Composite │ │Logistic  │ │SARIMAX   │
  │Stress    │ │Regression│ │Macro     │
  │Score     │ │Early     │ │Forecast  │
  │(0-100)   │ │Warning   │ │vs SARIMA │
  └────┬─────┘ └────┬─────┘ └────┬─────┘
       └────────────┼────────────┘
                    ▼
     ┌──────────────────────────────┐
     │     Power BI Dashboard        │
     │  Page 1: Institution Scorecard│
     │  Page 2: Early Warning Monitor│
     │  Page 3: Macro Overlay        │
     │  DAX measures + conditional   │
     │  formatting + drill-through   │
     └──────────────────────────────┘
```

---

## Key Results (populated after run)

| Metric | Value |
|---|---|
| Stress Model CV AUC | TBD |
| Early Detection Rate | TBD |
| Avg Lead Time (quarters) | TBD |
| NPA Forecast MAPE Improvement | TBD |
| Macro Factor Treatment Win Rate | TBD |
| Banks Monitored | 50-100 |
| Quarters of History | 20-24 |

---

## Power BI Setup

1. Install Power BI Desktop (free): https://powerbi.microsoft.com/desktop
2. Run pipeline to generate `data/outputs/bank_scores.csv`
3. In Power BI: Get Data → Text/CSV → select `bank_scores.csv`
4. Create measures using DAX from `dashboard/powerbi_dax_measures.txt`
5. Build 3 report pages per architecture above
6. Publish to Power BI Service (free account) for public embed URL

---

## Stack

| Component | Technology |
|---|---|
| Data Sources | FDIC BankFind API, FRED API (both free) |
| Data Engineering | Python, Pandas, dbt, PostgreSQL |
| ML | Scikit-learn (Logistic Regression), TimeSeriesSplit CV |
| Forecasting | statsmodels SARIMAX, SARIMA baseline |
| Experiment Tracking | MLflow |
| BI Dashboard | Power BI Desktop + Service, DAX |
| Infrastructure | Docker, Git |

---

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run demo
python main.py --mode demo

# View results
python main.py --mode results

# Run tests
pytest tests/ -v
```

---

## References

FDIC BankFind Suite API: https://banks.data.fdic.gov/docs/

Federal Reserve FRED API: https://fred.stlouisfed.org/

Altman, E.I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589-609.

Duffie, D. & Singleton, K.J. (2003). *Credit Risk: Pricing, Measurement, and Management*. Princeton University Press.
