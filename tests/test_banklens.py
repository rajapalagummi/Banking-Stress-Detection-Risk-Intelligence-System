"""BankLens — Unit Tests"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.ingest import build_bank_panel, _synthetic_institutions, _synthetic_fred
from models.stress.stress_model import compute_composite_stress_score, train_early_warning_model


class TestDataIngestion:
    def test_bank_panel_shape(self):
        df = build_bank_panel(n_banks=10, quarters=8)
        assert len(df) > 0
        assert "bank_id" in df.columns
        assert "npa_ratio" in df.columns

    def test_stress_banks_present(self):
        df = build_bank_panel(n_banks=20, quarters=15)
        assert df["is_stress_bank"].sum() > 0

    def test_npa_ratio_bounded(self):
        df = build_bank_panel(n_banks=10, quarters=8)
        assert (df["npa_ratio"] >= 0).all()
        assert (df["npa_ratio"] < 0.5).all()

    def test_synthetic_institutions(self):
        df = _synthetic_institutions(20)
        assert len(df) == 20
        assert "cert" in df.columns

    def test_synthetic_fred(self):
        df = _synthetic_fred("FEDFUNDS")
        assert len(df) > 0
        assert "value" in df.columns


class TestStressModel:
    def setup_method(self):
        self.df = build_bank_panel(n_banks=20, quarters=16)

    def test_composite_score_computed(self):
        result = compute_composite_stress_score(self.df)
        assert "stress_score" in result.columns
        assert (result["stress_score"] >= 0).all()
        assert (result["stress_score"] <= 100).all()

    def test_stress_tier_categories(self):
        result = compute_composite_stress_score(self.df)
        assert "stress_tier" in result.columns
        valid_tiers = {"Green", "Amber", "Red"}
        actual_tiers = set(result["stress_tier"].dropna().unique())
        assert actual_tiers.issubset(valid_tiers)

    def test_model_returns_auc(self):
        model_results = train_early_warning_model(self.df)
        assert "mean_cv_auc" in model_results
        assert 0 <= model_results["mean_cv_auc"] <= 1

    def test_stress_banks_score_higher(self):
        df = compute_composite_stress_score(self.df)
        stress_scores = df[df["is_stress_bank"] == 1]["stress_score"].mean()
        stable_scores = df[df["is_stress_bank"] == 0]["stress_score"].mean()
        assert stress_scores > stable_scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
