import warnings
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.config import DEFAULT_REPORTING


# ================================================================
# INTERNAL HELPERS
# ================================================================


def _resolve_image(path):
    """
    Return a Path if the file exists, else warn and return None.

    Centralises the missing-figure check so callers do not need to
    guard every image insertion individually.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        warnings.warn(f"Image not found, skipping: {p}")
        return None
    return p


def _format_excel_sheet(ws, df):
    """
    Apply publication-style formatting to an openpyxl worksheet.

    - Bold dark-blue header row with white text
    - Alternating light-grey fill on data rows
    - Auto-fit column widths (capped at 40)
    - Freeze pane below header row
    - Center-align all numeric cells
    """
    HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
    HEADER_FONT  = Font(bold=True, color="FFFFFF", size=11)
    ALT_FILL     = PatternFill("solid", fgColor="EEF2F7")
    CENTER       = Alignment(horizontal="center", vertical="center")

    n_cols = len(df.columns)

    # Header row
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER

    # Data rows
    for row_idx in range(2, ws.max_row + 1):
        fill = ALT_FILL if row_idx % 2 == 0 else None
        for col_idx in range(1, n_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            if fill:
                cell.fill = fill
            if col_idx > 1:  # numeric columns after the Model name
                cell.alignment = CENTER

    # Column widths
    for col_idx, col_name in enumerate(df.columns, start=1):
        max_len = max(
            len(str(col_name)),
            df.iloc[:, col_idx - 1].astype(str).str.len().max(),
        )
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 40)

    # Freeze pane: keep header visible when scrolling
    ws.freeze_panes = ws["A2"]


def _add_word_table(doc, df):
    """
    Insert a formatted Word table from a DataFrame.

    Header row uses bold text. All cells are centred. Column widths
    are distributed evenly across the page.
    """
    n_rows, n_cols = df.shape
    table = doc.add_table(rows=n_rows + 1, cols=n_cols)
    table.style = "Table Grid"

    # Header
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        hdr_cells[i].text = str(col_name)
        run = hdr_cells[i].paragraphs[0].runs[0]
        run.bold = True
        hdr_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Data rows
    for row_idx, row in enumerate(df.itertuples(index=False), start=1):
        row_cells = table.rows[row_idx].cells
        for col_idx, val in enumerate(row):
            row_cells[col_idx].text = str(val)
            row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_cm_pairs(doc, cm_paths, cm_width, heading="Confusion Matrices"):
    """
    Insert per-model images two per row under a section heading.

    cm_paths : dict {model_name: path_str_or_Path}
    cm_width : Inches width per image.
    heading  : Section heading text.
    """
    if not cm_paths:
        return

    items = [(name, _resolve_image(p)) for name, p in cm_paths.items()]
    items = [(name, p) for name, p in items if p is not None]

    if not items:
        return

    doc.add_heading(heading, level=2)

    # Pair up images — two per paragraph to approximate a two-column layout
    for i in range(0, len(items), 2):
        left_name,  left_path  = items[i]
        right_pair = items[i + 1] if i + 1 < len(items) else None

        para = doc.add_paragraph()
        run = para.add_run(f"{left_name}:  ")
        run.bold = True
        para.add_run().add_picture(str(left_path), width=Inches(cm_width))

        if right_pair:
            right_name, right_path = right_pair
            para.add_run(f"    {right_name}:  ").bold = True
            para.add_run().add_picture(str(right_path), width=Inches(cm_width))


# ================================================================
# PUBLIC API
# ================================================================


def export_results_csv(results_df, output_path):
    """
    Save the results DataFrame to a CSV file.

    Parameters
    ----------
    results_df  : pd.DataFrame   Per-model benchmark results.
    output_path : str or Path    Destination path (created if absent).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)


def export_results_excel(
    results_df,
    output_path,
    *,
    sheet_name=DEFAULT_REPORTING["sheet_name"],
):
    """
    Save the results DataFrame to a formatted Excel workbook.

    Applies bold dark-blue headers, alternating row fills, auto-fit
    column widths, and a frozen header row.

    Parameters
    ----------
    results_df  : pd.DataFrame   Per-model benchmark results.
    output_path : str or Path    Destination .xlsx path.
    sheet_name  : str            Worksheet tab label. Default "Results".
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        _format_excel_sheet(ws, results_df)


def export_results_word(
    results_df,
    output_path,
    *,
    title=DEFAULT_REPORTING["word_title"],
    description=None,
    roc_path=None,
    pr_path=None,
    cm_paths=None,
    scatter_paths=None,
    residual_paths=None,
    cluster_paths=None,
    pca_paths=None,
    forecast_paths=None,
    optimization_results=None,
    stats_summary=None,
    figure_width=DEFAULT_REPORTING["figure_width"],
    cm_width=DEFAULT_REPORTING["cm_width"],
):
    """
    Export a publication-oriented Word report (.docx).

    Structure
    ---------
    1. Title heading
    2. Optional description paragraph
    3. Results table (all models × all metrics)
    4. Page break + ROC curve figure (if roc_path provided and file exists)
    5. Page break + PR curve figure  (if pr_path  provided and file exists)
    6. Page break + confusion matrix pairs (if cm_paths provided)

    Parameters
    ----------
    results_df   : pd.DataFrame
        Per-model benchmark results — the DataFrame returned by
        benchmark.build_results_table() or run_benchmark().
    output_path  : str or Path
        Destination .docx file path.
    title        : str
        Report heading. Default "Benchmark Report".
    description  : str or None
        Optional text paragraph inserted before the results table.
    roc_path      : str, Path, or None
        Path to the combined ROC curve PNG generated by plots.py.
    pr_path       : str, Path, or None
        Path to the combined PR curve PNG generated by plots.py.
    cm_paths      : dict {model_name: path} or None
        Per-model confusion matrix paths. Images are laid out two per row.
    scatter_paths  : dict {model_name: path} or None
        Per-model actual-vs-predicted plot paths (regression). Two per row.
    residual_paths : dict {model_name: path} or None
        Per-model residual plot paths (regression). Two per row.
    cluster_paths  : dict {model_name: path} or None
        Per-model cluster scatter plot paths (unsupervised). Two per row.
    pca_paths      : dict {model_name: path} or None
        Per-model PCA variance plot paths (unsupervised). Two per row.
    forecast_paths : dict {model_name: path} or None
        Per-model forecast plot paths (time series). Two per row.
    optimization_results : dict or None
        {model_name: result_dict} from optimization.optimize_model().
        When provided, adds a hyperparameter optimization summary table:
        Model | Method | Best Score | # Evals | Duration (s) | Best Params.
    stats_summary : dict or None
        Output of the compute_stats block in run_benchmark. When provided,
        adds a Statistical Analysis section with a bootstrap CI table:
        Model | Metric | Mean | 95% CI Lower | 95% CI Upper.
    figure_width  : float
        Inches width for ROC and PR figures. Default 6.0.
    cm_width     : float
        Inches width for each confusion matrix thumbnail. Default 3.0.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # ── Title ─────────────────────────────────────────────────────────────
    heading = doc.add_heading(title, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── Optional description ───────────────────────────────────────────────
    if description:
        para = doc.add_paragraph(description)
        para.style.font.size = Pt(11)

    # ── Results table ──────────────────────────────────────────────────────
    doc.add_heading("Model Comparison Results", level=2)
    _add_word_table(doc, results_df)

    # ── ROC curve ─────────────────────────────────────────────────────────
    roc = _resolve_image(roc_path)
    if roc:
        doc.add_page_break()
        doc.add_heading("ROC Curves", level=2)
        doc.add_picture(str(roc), width=Inches(figure_width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── PR curve ──────────────────────────────────────────────────────────
    pr = _resolve_image(pr_path)
    if pr:
        doc.add_page_break()
        doc.add_heading("Precision-Recall Curves", level=2)
        doc.add_picture(str(pr), width=Inches(figure_width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── Confusion matrices (classification) ──────────────────────────────
    if cm_paths:
        doc.add_page_break()
        _add_cm_pairs(doc, cm_paths, cm_width)

    # ── Actual vs Predicted (regression) ─────────────────────────────────
    if scatter_paths:
        doc.add_page_break()
        _add_cm_pairs(doc, scatter_paths, cm_width,
                      heading="Actual vs Predicted")

    # ── Residual Plots (regression) ───────────────────────────────────────
    if residual_paths:
        doc.add_page_break()
        _add_cm_pairs(doc, residual_paths, cm_width,
                      heading="Residual Plots")

    # ── Cluster Scatter (unsupervised) ────────────────────────────────────
    if cluster_paths:
        doc.add_page_break()
        _add_cm_pairs(doc, cluster_paths, cm_width,
                      heading="Cluster Scatter Plots")

    # ── PCA Variance (unsupervised) ───────────────────────────────────────
    if pca_paths:
        doc.add_page_break()
        _add_cm_pairs(doc, pca_paths, cm_width,
                      heading="PCA Explained Variance")

    # ── Forecast Plots (time series) ──────────────────────────────────────
    if forecast_paths:
        doc.add_page_break()
        _add_cm_pairs(doc, forecast_paths, cm_width,
                      heading="Forecast Plots")

    # ── Hyperparameter Optimization Summary ───────────────────────────────
    if optimization_results:
        doc.add_page_break()
        doc.add_heading("Hyperparameter Optimization Summary", level=2)
        rows = []
        for model_name, res in optimization_results.items():
            params_str = ", ".join(
                f"{k}={v}" for k, v in (res.get("best_params") or {}).items()
            )
            rows.append({
                "Model":       model_name,
                "Method":      res.get("method", ""),
                "Best Score":  round(float(res.get("best_score", float("nan"))), 4),
                "# Evals":     res.get("n_evaluations", ""),
                "Duration (s)": round(float(res.get("search_duration", 0.0)), 1),
                "Best Params": params_str,
            })
        opt_df = pd.DataFrame(rows)
        _add_word_table(doc, opt_df)

    # ── Diebold-Mariano Forecast Comparison ──────────────────────────────────
    if stats_summary:
        for loss_key in ("dm_mae", "dm_mse"):
            records = stats_summary.get(loss_key)
            if not records:
                continue
            loss_label = loss_key.replace("dm_", "").upper()
            doc.add_page_break()
            doc.add_heading("Diebold-Mariano Forecast Comparison", level=2)
            doc.add_heading(f"Loss function: {loss_label}", level=3)
            rows = []
            for rec in records:
                rows.append({
                    "Model A":    rec.get("Model_A", ""),
                    "Model B":    rec.get("Model_B", ""),
                    "DM Stat":    round(float(rec.get("DM_Statistic", float("nan"))), 4),
                    "p-value":    round(float(rec.get("p_value", float("nan"))), 4),
                    "Loss Diff":  round(float(rec.get("Mean_Loss_Diff", float("nan"))), 6),
                    "Sig.":       "Yes" if rec.get("Significant") else "No",
                    "Favored":    rec.get("Favored", ""),
                })
            if rows:
                _add_word_table(doc, pd.DataFrame(rows))
            break  # Only add one section header; both loss tables follow together

        # Add second loss table without a new page break if first was added
        added_first = any(stats_summary.get(k) for k in ("dm_mae", "dm_mse"))
        loss_keys_remaining = [k for k in ("dm_mae", "dm_mse")
                               if stats_summary.get(k)]
        if len(loss_keys_remaining) > 1:
            second_key = loss_keys_remaining[1]
            loss_label2 = second_key.replace("dm_", "").upper()
            doc.add_heading(f"Loss function: {loss_label2}", level=3)
            rows2 = []
            for rec in stats_summary[second_key]:
                rows2.append({
                    "Model A":    rec.get("Model_A", ""),
                    "Model B":    rec.get("Model_B", ""),
                    "DM Stat":    round(float(rec.get("DM_Statistic", float("nan"))), 4),
                    "p-value":    round(float(rec.get("p_value", float("nan"))), 4),
                    "Loss Diff":  round(float(rec.get("Mean_Loss_Diff", float("nan"))), 6),
                    "Sig.":       "Yes" if rec.get("Significant") else "No",
                    "Favored":    rec.get("Favored", ""),
                })
            if rows2:
                _add_word_table(doc, pd.DataFrame(rows2))

    # ── Statistical Analysis ──────────────────────────────────────────────
    if stats_summary and "bootstrap_ci" in stats_summary:
        doc.add_page_break()
        doc.add_heading("Statistical Analysis", level=2)
        doc.add_heading("Bootstrap Confidence Intervals (95%)", level=3)
        rows = []
        for model_name, ci in stats_summary["bootstrap_ci"].items():
            rows.append({
                "Model":          model_name,
                "Metric":         ci.get("metric", ""),
                "Mean":           round(float(ci.get("mean", float("nan"))), 4),
                "95% CI Lower":   round(float(ci.get("lower", float("nan"))), 4),
                "95% CI Upper":   round(float(ci.get("upper", float("nan"))), 4),
            })
        if rows:
            _add_word_table(doc, pd.DataFrame(rows))

        if "mcnemar_pairs" in stats_summary and stats_summary["mcnemar_pairs"]:
            doc.add_heading("Pairwise McNemar Tests", level=3)
            pair_rows = []
            for pair_name, res in stats_summary["mcnemar_pairs"].items():
                pair_rows.append({
                    "Comparison":    pair_name,
                    "chi2":          round(float(res.get("statistic", float("nan"))), 4),
                    "p-value":       round(float(res.get("p_value", float("nan"))), 4),
                    "n_discordant":  res.get("n_discordant", ""),
                })
            if pair_rows:
                _add_word_table(doc, pd.DataFrame(pair_rows))

    doc.save(out)


# ================================================================
# EXPERIMENT ANALYSIS EXPORT
# ================================================================


def export_analysis_word(summary, save_path, *, title="Experiment Analysis Report"):
    """
    Export experiment history analysis to a Word document.

    Writes a formatted ``.docx`` with three sections:
    1. Experiment Summary (key stats)
    2. Aggregate Results (per-model table)
    3. Run Timeline (best-model-per-run table, up to 30 rows)

    Parameters
    ----------
    summary   : dict  Output of summarize_experiment_history().
    save_path : str or Path
    title     : str   Report title heading.
    """
    import pandas as _pd

    doc = Document()

    # ── Title ──────────────────────────────────────────────────────────────
    doc.add_heading(title, level=0)

    # ── Experiment Summary ──────────────────────────────────────────────────
    doc.add_heading("Experiment Summary", level=1)
    p = doc.add_paragraph()
    p.add_run(f"Total runs: ").bold = True
    p.add_run(f"{summary.get('n_runs', 0)}\n")
    p.add_run("Experiments: ").bold = True
    p.add_run(f"{summary.get('n_experiments', 0)}\n")
    if summary.get("experiment_names"):
        p.add_run("Experiment names: ").bold = True
        p.add_run(f"{', '.join(summary['experiment_names'])}\n")
    p.add_run("Metric: ").bold = True
    p.add_run(f"{summary.get('metric', '')}\n")
    if summary.get("best_model"):
        p.add_run("Best overall model: ").bold = True
        p.add_run(f"{summary['best_model']}\n")

    # ── Aggregate Results ──────────────────────────────────────────────────
    agg = summary.get("aggregate", _pd.DataFrame())
    if not agg.empty:
        doc.add_heading("Aggregate Results", level=1)
        _add_word_table(doc, agg)

    # ── Experiment Comparison ──────────────────────────────────────────────
    comp = summary.get("comparison", _pd.DataFrame())
    if not comp.empty:
        doc.add_heading("Experiment Comparison", level=1)
        comp_reset = comp.reset_index()
        _add_word_table(doc, comp_reset)

    # ── Run Timeline ───────────────────────────────────────────────────────
    tl = summary.get("timeline", _pd.DataFrame())
    if not tl.empty:
        doc.add_heading("Run Timeline", level=1)
        cols = ["run_id", "experiment_name", "best_model", "best_score"]
        present_cols = [c for c in cols if c in tl.columns]
        _add_word_table(doc, tl[present_cols].head(30))

    doc.save(str(save_path))


# ================================================================
# FUTURE: ADDITIONAL EXPORT FORMATS  (not yet implemented)
# ================================================================
#
# All planned export functions follow the same signature convention:
#   export_results_<format>(results_df, output_path, **kwargs)
#
# LaTeX export (publication-ready table):
#   export_results_latex(results_df, output_path, *,
#                        caption="Model Comparison Results",
#                        label="tab:results", bold_best=True)
#   Outputs a standalone .tex file with \begin{table} ... \end{table}.
#   bold_best=True highlights the top metric value per column.
#   Requires: no additional dependencies (stdlib only).
#
# PDF report:
#   export_results_pdf(results_df, output_path, *,
#                      title="Benchmark Report", roc_path=None,
#                      pr_path=None, cm_paths=None)
#   Generates a PDF via reportlab or WeasyHTML→PDF.
#   Requires: pip install reportlab  OR  pip install weasyprint
#
# HTML report:
#   export_results_html(results_df, output_path, *,
#                       title="Benchmark Report", roc_path=None,
#                       pr_path=None, cm_paths=None,
#                       embed_images=True)
#   Self-contained HTML with an interactive sortable table (via DataTables CDN).
#   embed_images=True base64-encodes all figures so the file is portable.
#   Requires: no additional dependencies (stdlib only).
