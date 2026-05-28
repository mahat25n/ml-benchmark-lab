"""Tests for src/reporting.py — CSV, Excel, and Word export functions."""

import warnings

import pandas as pd
import pytest


# ================================================================
# Shared fixture: small results DataFrame
# ================================================================

@pytest.fixture
def results_df():
    return pd.DataFrame({
        "Model":    ["Ridge", "Lasso", "Linear Regression"],
        "MAE":      [0.321, 0.415, 0.289],
        "RMSE":     [0.443, 0.531, 0.401],
        "R2":       [0.871, 0.762, 0.903],
    })


# ================================================================
# TestExportCSV
# ================================================================

class TestExportCSV:

    def test_file_is_created(self, tmp_path, results_df):
        from src.reporting import export_results_csv
        p = tmp_path / "results.csv"
        export_results_csv(results_df, p)
        assert p.exists()

    def test_file_is_non_empty(self, tmp_path, results_df):
        from src.reporting import export_results_csv
        p = tmp_path / "results.csv"
        export_results_csv(results_df, p)
        assert p.stat().st_size > 0

    def test_roundtrip_columns(self, tmp_path, results_df):
        from src.reporting import export_results_csv
        p = tmp_path / "results.csv"
        export_results_csv(results_df, p)
        loaded = pd.read_csv(p)
        assert list(loaded.columns) == list(results_df.columns)

    def test_roundtrip_row_count(self, tmp_path, results_df):
        from src.reporting import export_results_csv
        p = tmp_path / "results.csv"
        export_results_csv(results_df, p)
        loaded = pd.read_csv(p)
        assert len(loaded) == len(results_df)

    def test_creates_parent_dir_if_absent(self, tmp_path, results_df):
        from src.reporting import export_results_csv
        p = tmp_path / "subdir" / "results.csv"
        export_results_csv(results_df, p)
        assert p.exists()


# ================================================================
# TestExportExcel
# ================================================================

class TestExportExcel:

    def test_file_is_created(self, tmp_path, results_df):
        from src.reporting import export_results_excel
        p = tmp_path / "results.xlsx"
        export_results_excel(results_df, p)
        assert p.exists()

    def test_file_is_non_empty(self, tmp_path, results_df):
        from src.reporting import export_results_excel
        p = tmp_path / "results.xlsx"
        export_results_excel(results_df, p)
        assert p.stat().st_size > 0

    def test_roundtrip_columns(self, tmp_path, results_df):
        from src.reporting import export_results_excel
        p = tmp_path / "results.xlsx"
        export_results_excel(results_df, p)
        loaded = pd.read_excel(p)
        assert list(loaded.columns) == list(results_df.columns)

    def test_roundtrip_row_count(self, tmp_path, results_df):
        from src.reporting import export_results_excel
        p = tmp_path / "results.xlsx"
        export_results_excel(results_df, p)
        loaded = pd.read_excel(p)
        assert len(loaded) == len(results_df)

    def test_custom_sheet_name(self, tmp_path, results_df):
        from src.reporting import export_results_excel
        p = tmp_path / "results.xlsx"
        export_results_excel(results_df, p, sheet_name="Benchmark")
        loaded = pd.read_excel(p, sheet_name="Benchmark")
        assert len(loaded) == len(results_df)

    def test_creates_parent_dir_if_absent(self, tmp_path, results_df):
        from src.reporting import export_results_excel
        p = tmp_path / "nested" / "results.xlsx"
        export_results_excel(results_df, p)
        assert p.exists()


# ================================================================
# TestExportWord
# ================================================================

class TestExportWord:

    def test_file_is_created(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "report.docx"
        export_results_word(results_df, p)
        assert p.exists()

    def test_file_is_non_empty(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "report.docx"
        export_results_word(results_df, p)
        assert p.stat().st_size > 0

    def test_custom_title_does_not_raise(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "report.docx"
        export_results_word(results_df, p, title="Custom Report Title")
        assert p.exists()

    def test_description_paragraph_does_not_raise(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "report.docx"
        export_results_word(results_df, p, description="A short description.")
        assert p.exists()

    def test_missing_roc_path_warns_not_raises(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "report.docx"
        missing = tmp_path / "nonexistent_roc.png"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            export_results_word(results_df, p, roc_path=missing)
        assert p.exists()
        assert any("nonexistent_roc" in str(w.message) for w in caught)

    def test_missing_pr_path_warns_not_raises(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "report.docx"
        missing = tmp_path / "nonexistent_pr.png"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            export_results_word(results_df, p, pr_path=missing)
        assert p.exists()
        assert any("nonexistent_pr" in str(w.message) for w in caught)

    def test_missing_cm_path_warns_not_raises(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "report.docx"
        cm_paths = {"ModelA": tmp_path / "missing_cm.png"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            export_results_word(results_df, p, cm_paths=cm_paths)
        assert p.exists()
        assert any("missing_cm" in str(w.message) for w in caught)

    def test_existing_roc_image_embedded(self, tmp_path, results_df):
        from src.reporting import export_results_word
        from src.plots import init_roc_figure, finalize_roc_plot
        roc_png = tmp_path / "roc.png"
        fig, ax = init_roc_figure()
        finalize_roc_plot(fig, ax, roc_png)
        p = tmp_path / "report.docx"
        export_results_word(results_df, p, roc_path=roc_png)
        assert p.exists() and p.stat().st_size > 0

    def test_scatter_and_residual_paths_do_not_raise(self, tmp_path, results_df):
        """Passing missing scatter/residual paths only warns — never raises."""
        from src.reporting import export_results_word
        scatter  = {"Ridge": tmp_path / "missing_avp.png"}
        residual = {"Ridge": tmp_path / "missing_res.png"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            export_results_word(results_df, tmp_path / "r.docx",
                                scatter_paths=scatter, residual_paths=residual)
        assert (tmp_path / "r.docx").exists()

    def test_cluster_and_pca_paths_do_not_raise(self, tmp_path, results_df):
        """Passing missing cluster/pca paths only warns — never raises."""
        from src.reporting import export_results_word
        cluster = {"KMeans": tmp_path / "missing_cluster.png"}
        pca     = {"PCA":    tmp_path / "missing_pca.png"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            export_results_word(results_df, tmp_path / "r2.docx",
                                cluster_paths=cluster, pca_paths=pca)
        assert (tmp_path / "r2.docx").exists()

    def test_creates_parent_dir_if_absent(self, tmp_path, results_df):
        from src.reporting import export_results_word
        p = tmp_path / "nested" / "report.docx"
        export_results_word(results_df, p)
        assert p.exists()
