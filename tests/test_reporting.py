import pymupdf

from bias_detector.metrics import AuditSettings, calculate_audit
from bias_detector.reporting import build_pdf
from bias_detector.synthetic import generate_scenario


def test_pdf_contains_report_values_without_data_rows():
    data=generate_scenario("S3 — Moderate disparity")
    result=calculate_audit(data,AuditSettings(group_column="group",prediction_column="prediction",true_column="truth",minimum_group_size=30))
    pdf=build_pdf(result,"AI Bias Detector — Summary Report","Synthetic sample","Seeded synthetic reference data","positive prediction")
    with pymupdf.open(stream=pdf,filetype="pdf") as doc:
        text="\n".join(page.get_text() for page in doc)
    compact=" ".join(text.split())
    assert "All-group DPD" in compact
    assert "Eligible-group screening" in compact
    assert "Sample status" in compact
    assert "not proof of discrimination" in text
    assert "Group A" in text
    assert "Generated (UTC)" in text
