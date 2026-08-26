"""
BankLens — Main Pipeline Orchestrator
Retail Banking Performance Intelligence & Risk Analytics Platform

Usage:
    python main.py --mode demo      # Quick demo (50 banks, 20 quarters)
    python main.py --mode full      # Full pipeline (100 banks, 24 quarters)
    python main.py --mode results   # Print saved results
"""
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_demo():
    from data.ingest import build_bank_panel, run_ingestion
    from models.stress.stress_model import run_stress_analysis
    from models.forecast.forecaster import run_forecast_comparison, load_macro_factors

    logger.info("=== BankLens Demo ===")
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    logger.info("\n--- Stage 1: Data Ingestion ---")
    run_ingestion()
    df = build_bank_panel(n_banks=50, quarters=28)
    logger.info(f"Panel: {len(df)} rows, {df['bank_id'].nunique()} banks")
    logger.info(f"Stress banks: {df['is_stress_bank'].sum() // 20}")
    logger.info(f"Date range: {df['quarter_date'].min()} to {df['quarter_date'].max()}")

    logger.info("\n--- Stage 2: Stress Scoring & Early Warning Model ---")
    stress_results = run_stress_analysis(df)

    logger.info("\n--- Stage 3: Forecast Comparison ---")
    macro_df = load_macro_factors()
    forecast_results = run_forecast_comparison(df, macro_df)

    model = stress_results.get("model", {})
    detection = stress_results.get("detection", {})
    forecast_summary = forecast_results.get("summary", {})

    results = {
        "mode": "demo",
        "n_banks": int(df["bank_id"].nunique()),
        "n_quarters": int(df["quarter_date"].nunique()),
        "stress_model": {
            "mean_cv_auc": model.get("mean_cv_auc"),
            "std_cv_auc": model.get("std_cv_auc"),
            "n_folds": model.get("n_folds"),
            "feature_importance": model.get("feature_importance", {}),
        },
        "early_detection": {
            "early_detection_rate": detection.get("early_detection_rate"),
            "avg_lead_quarters": detection.get("avg_lead_quarters"),
            "n_stress_banks": detection.get("n_stress_banks"),
        },
        "stress_distribution": stress_results.get("stress_distribution", {}),
        "forecast_comparison": forecast_summary,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n=== DEMO RESULTS ===")
    logger.info(f"Stress Model AUC: {model.get('mean_cv_auc')} ± {model.get('std_cv_auc')}")
    logger.info(f"Early detection rate: {detection.get('early_detection_rate')}")
    logger.info(f"Avg lead time: {detection.get('avg_lead_quarters')} quarters")
    logger.info(f"Stress distribution: {stress_results.get('stress_distribution', {})}")
    if forecast_summary:
        for metric, stats in forecast_summary.items():
            logger.info(f"Forecast {metric}: improvement={stats.get('avg_improvement_pct')}%, "
                        f"treatment wins={stats.get('treatment_win_rate')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="BankLens Pipeline")
    parser.add_argument("--mode", choices=["demo", "full", "results"], default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        results = run_demo()
        print("\n=== Final Numbers ===")
        print(json.dumps({k: v for k, v in results.items() if k != "mode"},
                         indent=2, default=str))

    elif args.mode == "results":
        path = OUTPUT_DIR / "results.json"
        if path.exists():
            with open(path) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("No results found. Run --mode demo first.")


if __name__ == "__main__":
    main()
