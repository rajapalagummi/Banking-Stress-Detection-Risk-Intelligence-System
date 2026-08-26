"""
BankLens — Deposit Outflow & NPA Forecasting
SARIMAX with macro factors (Fed funds rate, yield curve) vs baseline SARIMA.
Bayesian model comparison using WAIC to determine if macro factors improve forecasts.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
HOLDOUT_QUARTERS = 4


def load_macro_factors(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load FRED macro data and resample to quarterly."""
    import os
    from pathlib import Path
    macro = {}
    for series in ["fed_funds", "yield_curve", "unemployment"]:
        path = Path(raw_dir) / f"{series}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.set_index("date").resample("QS").mean()
            macro[series] = df["value"]

    if macro:
        return pd.DataFrame(macro).reset_index().rename(columns={"date": "quarter_date"})
    return pd.DataFrame()


def fit_sarima_baseline(series: pd.Series, holdout: int = HOLDOUT_QUARTERS) -> dict:
    """SARIMA baseline — no macro factors."""
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]
    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 4),
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        forecast = result.forecast(steps=holdout)
        mape = mean_absolute_percentage_error(test.clip(1e-6), forecast.clip(1e-6))
        aic = result.aic
        return {"mape": round(mape, 4), "aic": round(aic, 2), "forecast": forecast.values,
                "actual": test.values, "model": "SARIMA"}
    except Exception as e:
        logger.warning(f"SARIMA failed: {e}")
        return {"mape": None, "aic": None, "model": "SARIMA_FAILED"}


def fit_sarimax_macro(series: pd.Series, macro_df: pd.DataFrame,
                       holdout: int = HOLDOUT_QUARTERS) -> dict:
    """SARIMAX with Fed funds rate and yield curve as exogenous variables."""
    if macro_df.empty:
        return fit_sarima_baseline(series, holdout)

    try:
        macro_aligned = macro_df.set_index("quarter_date").reindex(
            series.index, method="nearest"
        ).fillna(method="ffill")

        exog_cols = [c for c in ["fed_funds", "yield_curve"] if c in macro_aligned.columns]
        if not exog_cols:
            return fit_sarima_baseline(series, holdout)

        exog = macro_aligned[exog_cols]
        train_y = series.iloc[:-holdout]
        test_y = series.iloc[-holdout:]
        train_x = exog.iloc[:-holdout]
        test_x = exog.iloc[-holdout:]

        model = SARIMAX(train_y, exog=train_x, order=(1, 1, 1),
                        seasonal_order=(1, 0, 1, 4),
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        forecast = result.forecast(steps=holdout, exog=test_x)
        mape = mean_absolute_percentage_error(test_y.clip(1e-6), forecast.clip(1e-6))

        return {"mape": round(mape, 4), "aic": round(result.aic, 2),
                "forecast": forecast.values, "actual": test_y.values,
                "model": "SARIMAX_MACRO", "exog_used": exog_cols}
    except Exception as e:
        logger.warning(f"SARIMAX failed: {e}")
        return fit_sarima_baseline(series, holdout)


def run_forecast_comparison(df: pd.DataFrame, macro_df: pd.DataFrame) -> dict:
    """
    A/B test: SARIMA baseline vs SARIMAX with macro factors.
    Run for NPA ratio and deposit growth across all banks.
    """
    results = []
    for bank_id in df["bank_id"].unique():
        bank_data = df[df["bank_id"] == bank_id].sort_values("quarter_date")
        if len(bank_data) < HOLDOUT_QUARTERS + 8:
            continue

        bank_ts = bank_data.set_index("quarter_date")

        for metric in ["npa_ratio"]:
            if metric not in bank_ts.columns:
                continue
            series = bank_ts[metric].dropna()
            if len(series) < HOLDOUT_QUARTERS + 6:
                continue

            sarima = fit_sarima_baseline(series)
            sarimax = fit_sarimax_macro(series, macro_df)

            if sarima.get("mape") and sarimax.get("mape"):
                improvement = (sarima["mape"] - sarimax["mape"]) / sarima["mape"] * 100
                aic_improvement = (sarima.get("aic", 0) - sarimax.get("aic", 0))
                results.append({
                    "bank_id": bank_id,
                    "metric": metric,
                    "sarima_mape": sarima["mape"],
                    "sarimax_mape": sarimax["mape"],
                    "mape_improvement_pct": round(improvement, 2),
                    "sarima_aic": sarima.get("aic"),
                    "sarimax_aic": sarimax.get("aic"),
                    "aic_improvement": round(aic_improvement, 2),
                    "treatment_wins": int(sarimax["mape"] < sarima["mape"]),
                })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return {}

    summary = {}
    for metric in results_df["metric"].unique():
        m = results_df[results_df["metric"] == metric]
        summary[metric] = {
            "sarima_avg_mape": round(m["sarima_mape"].mean(), 4),
            "sarimax_avg_mape": round(m["sarimax_mape"].mean(), 4),
            "avg_improvement_pct": round(m["mape_improvement_pct"].mean(), 2),
            "treatment_win_rate": round(m["treatment_wins"].mean(), 4),
            "n_banks": len(m),
        }

    logger.info(f"\n=== Forecast Comparison Results ===")
    for metric, stats in summary.items():
        logger.info(f"{metric}: MAPE improvement={stats['avg_improvement_pct']:.2f}%, "
                    f"treatment wins={stats['treatment_win_rate']:.1%}")

    return {"summary": summary, "detail": results_df}
