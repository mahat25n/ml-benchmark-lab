"""Tests for src/benchmark.py — run_benchmark and build_results_table."""

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier


# ================================================================
# Module-scoped fixture: run benchmark once, reuse across tests
# ================================================================

@pytest.fixture(scope="module")
def bench(tmp_path_factory, tmp_csv):
    """One benchmark run, shared across all tests in this module."""
    from src.benchmark import run_benchmark
    out = tmp_path_factory.mktemp("bench_out")
    results_df, pp = run_benchmark(
        tmp_csv, "target",
        output_dir=str(out),
        models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
        verbose=False,
    )
    return results_df, pp, out


# ================================================================
# TestBuildResultsTable
# ================================================================

class TestBuildResultsTable:

    def test_returns_dataframe(self):
        from src.benchmark import build_results_table
        rows = [{"Model": "A", "ROC AUC": 0.9}, {"Model": "B", "ROC AUC": 0.7}]
        df = build_results_table(rows)
        assert isinstance(df, pd.DataFrame)

    def test_sorted_descending_by_sort_col(self):
        from src.benchmark import build_results_table
        rows = [{"Model": "A", "ROC AUC": 0.7}, {"Model": "B", "ROC AUC": 0.9}]
        df = build_results_table(rows, sort_by="ROC AUC")
        assert df.iloc[0]["Model"] == "B"

    def test_missing_sort_col_does_not_raise(self):
        from src.benchmark import build_results_table
        rows = [{"Model": "A", "Accuracy": 0.8}]
        df = build_results_table(rows, sort_by="ROC AUC")
        assert isinstance(df, pd.DataFrame)

    def test_numeric_rounded(self):
        from src.benchmark import build_results_table
        rows = [{"Model": "A", "ROC AUC": 0.123456789}]
        df = build_results_table(rows, round_digits=3)
        assert df.iloc[0]["ROC AUC"] == round(0.123456789, 3)

    def test_reset_integer_index(self):
        from src.benchmark import build_results_table
        rows = [{"Model": "A", "ROC AUC": 0.9}, {"Model": "B", "ROC AUC": 0.7}]
        df = build_results_table(rows)
        assert list(df.index) == list(range(len(df)))


# ================================================================
# TestRunBenchmark
# ================================================================

class TestRunBenchmark:

    def test_returns_tuple_of_two(self, tmp_csv, tmp_path):
        from src.benchmark import run_benchmark
        result = run_benchmark(
            tmp_csv, "target",
            output_dir=str(tmp_path),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            verbose=False,
        )
        assert len(result) == 2

    def test_results_is_dataframe(self, bench):
        results_df, _, _ = bench
        assert isinstance(results_df, pd.DataFrame)

    def test_model_column_present(self, bench):
        results_df, _, _ = bench
        assert "Model" in results_df.columns

    def test_expected_metric_columns_present(self, bench):
        results_df, _, _ = bench
        for col in ("Accuracy", "F1 Score", "ROC AUC"):
            assert col in results_df.columns

    def test_preprocessor_has_expected_keys(self, bench):
        _, pp, _ = bench
        for key in ("scaler", "feature_encoders", "target_encoder",
                    "feature_names", "target_col", "n_classes"):
            assert key in pp

    def test_accuracy_in_valid_range(self, bench):
        results_df, _, _ = bench
        acc = results_df["Accuracy"].dropna()
        assert (acc >= 0.0).all() and (acc <= 1.0).all()

    def test_one_row_per_model(self, bench):
        results_df, _, _ = bench
        assert len(results_df) == 1  # single "DT" model

    def test_confusion_matrix_png_created(self, bench):
        _, _, out = bench
        cm_files = list(out.glob("cm_*.png"))
        assert len(cm_files) >= 1

    def test_unsupported_task_raises(self, tmp_csv, tmp_path):
        from src.benchmark import run_benchmark
        with pytest.raises(NotImplementedError):
            run_benchmark(
                tmp_csv, "target",
                task="forecasting",
                models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
                output_dir=str(tmp_path),
                verbose=False,
            )

    def test_export_csv_creates_file(self, tmp_csv, tmp_path):
        from src.benchmark import run_benchmark
        run_benchmark(
            tmp_csv, "target",
            output_dir=str(tmp_path),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            export_formats=["csv"],
            verbose=False,
        )
        assert (tmp_path / "results.csv").exists()

    def test_compute_importance_stored_in_preprocessor(self, tmp_csv, tmp_path):
        from src.benchmark import run_benchmark
        _, pp = run_benchmark(
            tmp_csv, "target",
            output_dir=str(tmp_path),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            compute_importance=True,
            verbose=False,
        )
        assert "importance" in pp
        assert "DT" in pp["importance"]

    @pytest.mark.integration
    def test_imbalance_strategy_does_not_raise(self, tmp_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            tmp_csv, "target",
            output_dir=str(tmp_path),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            imbalance_strategy="random_over",
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)

    @pytest.mark.integration
    def test_cv_strategy_string_accepted(self, tmp_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            tmp_csv, "target",
            output_dir=str(tmp_path),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            cv_strategy="stratified_kfold",
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)
