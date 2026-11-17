"""Generate a compact in-memory PDF audit summary."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


def build_pdf(result: dict, title: str, dataset_name: str, source: str, positive_description: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.62*inch, leftMargin=0.62*inch, topMargin=0.62*inch, bottomMargin=0.62*inch, title=title)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#52616b")))
    story = [Paragraph(escape(title.replace("—", "-")), styles["Title"]), Paragraph("AI Bias Detector - Group disparity summary", styles["Heading2"]),
             Paragraph(f"Dataset: {escape(dataset_name)} | Source: {escape(source)}", styles["BodyText"]),
             Paragraph(f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}", styles["SmallNote"]), Spacer(1, 12),
             Paragraph("Analysis configuration", styles["Heading2"])]
    settings = result["metrics"]["settings"]
    config = [["Demographic field", settings["group_column"]], ["Prediction field", settings["prediction_column"]],
              ["True outcome", settings["true_column"] or "Not supplied"], ["Positive class means", positive_description],
              ["Prediction mode", "Score threshold" if settings["score_mode"] else "Binary class"],
              ["Score threshold", f">= {settings['score_threshold']}" if settings["score_mode"] else "Not applicable"],
              ["Prediction labels", f"positive: {settings['positive_label']}; negative: {settings['negative_label']}"],
              ["True labels", f"positive: {settings['true_positive_label']}; negative: {settings['true_negative_label']}" if settings["true_column"] else "Not supplied"],
              ["Rows analysed", f"{result['metrics']['rows_analysed']:,}"], ["DPD screen", f">= {settings['dpd_threshold']:.3f}"],
              ["DIR screen", f"< {settings['dir_threshold']:.3f}"], ["Minimum group size", str(settings["minimum_group_size"])]]
    story.append(_table(config, [2.05*inch, 4.95*inch]))
    story += [Spacer(1, 14), Paragraph("Group results", styles["Heading2"])]
    headers = ["Group", "n", "Selection rate", "TPR", "FPR", "FNR"]
    rows = [headers]
    for row in result["groups"]:
        rows.append([escape(str(row["group"])), str(row["n"]), f"{row['selection_rate']:.4f}", _fmt(row.get("tpr")), _fmt(row.get("fpr")), _fmt(row.get("fnr"))])
    story.append(_table(rows, [2.4*inch, .55*inch, 1.0*inch, .8*inch, .8*inch, .8*inch], font=8))
    m = result["metrics"]
    story += [Spacer(1, 14), Paragraph("Summary metrics", styles["Heading2"])]
    metric_rows = [["Metric", "Value", "Interpretation"]]
    metric_rows.extend([
        ["Demographic parity difference", _fmt(m["demographic_parity_difference"]), m["decision_dpd"]],
        ["Disparate impact ratio", _fmt(m["disparate_impact_ratio"]), m["decision_dir"]],
        ["Equal opportunity difference", _fmt(m["equal_opportunity_difference"]), "Gap between highest and lowest group TPR."],
        ["Equalised odds difference", _fmt(m["equalised_odds_difference"]), "Larger of the TPR and FPR gaps."],
    ])
    story.append(_table(metric_rows, [1.65*inch, .72*inch, 4.55*inch], font=7.5))
    comp = m["selection_rate_comparison"]
    story += [Spacer(1, 10), Paragraph(f"Two-sided selection-rate test: highest group {escape(str(comp['high_group']))}; lowest group {escape(str(comp['low_group']))}; z={_fmt(comp['z'])}; p={_fmt(comp['p_value_two_sided'])}.", styles["BodyText"])]
    if m["undefined"]:
        story += [Spacer(1, 8), Paragraph("Unavailable measures", styles["Heading2"])]
        story += [Paragraph(escape(item), styles["SmallNote"]) for item in m["undefined"]]
    story += [Spacer(1, 14), Paragraph("Interpretation and limitations", styles["Heading2"]),
              Paragraph("These metrics are screening indicators for further investigation, not findings of legal discrimination or definitive judgements of fairness. Selection-rate comparisons do not assess prediction correctness. Error-rate comparisons depend on the quality and meaning of recorded labels. Different fairness definitions may disagree, and results depend on the selected groups, positive class, threshold, sample size and data preparation. COMPAS positive means a higher risk flag (adverse).", styles["SmallNote"])]
    doc.build(story)
    return buffer.getvalue()


def _fmt(value) -> str:
    return "Unavailable" if value is None else f"{float(value):.6f}"


def _table(data, widths, font=9):
    wrapped = [[Paragraph(str(v), getSampleStyleSheet()["BodyText"]) for v in row] for row in data]
    table = Table(wrapped, colWidths=widths, repeatRows=1 if len(data) > 1 else 0, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#eaf0ed")),
        ("TEXTCOLOR", (0,0),(-1,0), colors.HexColor("#173b34")),
        ("FONTNAME", (0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0),(-1,-1), font),
        ("GRID", (0,0),(-1,-1), .35, colors.HexColor("#d4ded9")),
        ("VALIGN", (0,0),(-1,-1), "TOP"), ("LEFTPADDING", (0,0),(-1,-1), 5),
        ("RIGHTPADDING", (0,0),(-1,-1), 5), ("TOPPADDING", (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    return table
