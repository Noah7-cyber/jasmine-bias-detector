import numpy as np
import pandas as pd
import pytest

from bias_detector.metrics import AuditSettings, calculate_audit
from bias_detector.data_loader import read_uploaded_csv
from bias_detector.synthetic import SCENARIOS, generate_scenario


def example_frame():
    return pd.DataFrame([
        *([{"group":"A","pred":1,"truth":1}]*80), *([{"group":"A","pred":0,"truth":1}]*20),
        *([{"group":"A","pred":1,"truth":0}]*40), *([{"group":"A","pred":0,"truth":0}]*60),
        *([{"group":"B","pred":1,"truth":1}]*30), *([{"group":"B","pred":0,"truth":1}]*20),
        *([{"group":"B","pred":1,"truth":0}]*30), *([{"group":"B","pred":0,"truth":0}]*70),
    ])


def config(**kwargs):
    base = dict(group_column="group", prediction_column="pred", true_column="truth", minimum_group_size=30)
    base.update(kwargs)
    return AuditSettings(**base)


def test_proposal_worked_example():
    out=calculate_audit(example_frame(),config())
    m=out["metrics"]
    assert out["groups"][0]["selection_rate"] == pytest.approx(.60)
    assert out["groups"][1]["selection_rate"] == pytest.approx(.40)
    assert m["demographic_parity_difference"] == pytest.approx(.20)
    assert m["disparate_impact_ratio"] == pytest.approx(2/3)
    assert m["true_positive_rate"]["A"] == pytest.approx(.80)
    assert m["false_positive_rate"]["A"] == pytest.approx(.40)
    assert m["equal_opportunity_difference"] == pytest.approx(.20)
    assert m["equalised_odds_difference"] == pytest.approx(.20)
    assert m["selection_rate_comparison"]["p_value_two_sided"] == pytest.approx(.0002, abs=1e-4)


def test_score_threshold_is_inclusive_and_labels_are_independent():
    frame=pd.DataFrame({"group":["A"]*40+["B"]*40,"score":[5]*20+[4]*20+[5]*10+[4]*30,"outcome":["yes"]*20+["no"]*20+["no"]*40})
    out=calculate_audit(frame,AuditSettings(group_column="group",prediction_column="score",score_mode=True,score_threshold=5,true_column="outcome",true_positive_label="yes",true_negative_label="no",minimum_group_size=1))
    assert out["groups"][0]["selection_rate"] == pytest.approx(.5)
    assert out["groups"][1]["selection_rate"] == pytest.approx(.25)


@pytest.mark.parametrize("name",list(SCENARIOS))
def test_fixed_seed_scenarios_are_reproducible(name):
    pd.testing.assert_frame_equal(generate_scenario(name,7),generate_scenario(name,7))


def test_small_groups_keep_metrics_but_report_insufficient():
    frame=generate_scenario("S5 — Small group")
    m=calculate_audit(frame,config(prediction_column="prediction"))["metrics"]
    assert m["demographic_parity_difference"] == pytest.approx(.30)
    assert m["reliable"] is False
    assert m["decision_dpd"].startswith("Insufficient data")


@pytest.mark.parametrize("column",["pred","truth"])
def test_invalid_binary_values_block_analysis(column):
    frame=example_frame()
    frame[column]=frame[column].astype("string")
    frame.loc[0,column]="unknown"
    with pytest.raises(ValueError): calculate_audit(frame,config())


def test_uploaded_csv_duplicate_headers_detected_before_pandas_renames_them():
    frame,duplicates=read_uploaded_csv(b"group,group,pred\nA,B,1\n")
    assert duplicates==["group"]


def test_uploaded_csv_is_read_directly_from_bytes():
    frame,duplicates=read_uploaded_csv(b"group,pred\nA,1\nB,0\n")
    assert not duplicates
    assert frame.to_dict(orient="list")=={"group":["A","B"],"pred":[1,0]}


def test_all_zero_predictions_define_dpd_but_not_dir_or_test():
    frame=pd.DataFrame({"group":["A"]*40+["B"]*40,"pred":[0]*80})
    out=calculate_audit(frame,config(true_column=None,minimum_group_size=1))
    m=out["metrics"]
    assert m["demographic_parity_difference"]==0
    assert m["disparate_impact_ratio"] is None
    assert m["selection_rate_comparison"]["p_value_two_sided"] is None


def test_threshold_boundaries_use_proposal_inclusive_and_strict_inequalities():
    frame=pd.DataFrame({"group":["A"]*100+["B"]*100,"pred":[1]*60+[0]*40+[1]*50+[0]*50})
    m=calculate_audit(frame,config(true_column=None,minimum_group_size=1))["metrics"]
    assert m["demographic_parity_difference"]==pytest.approx(.1)
    assert m["demographic_parity_flag"] is True
    assert m["disparate_impact_ratio"]==pytest.approx(5/6)
    exact_dir=pd.DataFrame({"group":["A"]*100+["B"]*100,"pred":[1]*80+[0]*20+[1]*64+[0]*36})
    m2=calculate_audit(exact_dir,config(true_column=None,minimum_group_size=1))["metrics"]
    assert m2["disparate_impact_ratio"]==pytest.approx(.8)
    assert m2["disparate_impact_flag"] is False


def test_fairlearn_reference_agreement():
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference
    frame=example_frame()
    out=calculate_audit(frame,config())
    groups=frame["group"].to_numpy(); y=frame["truth"].to_numpy(); pred=frame["pred"].to_numpy()
    m=out["metrics"]
    assert m["demographic_parity_difference"] == pytest.approx(demographic_parity_difference(y,pred,sensitive_features=groups,method="between_groups"),abs=1e-6)
    assert m["disparate_impact_ratio"] == pytest.approx(demographic_parity_ratio(y,pred,sensitive_features=groups,method="between_groups"),abs=1e-6)
    assert m["equalised_odds_difference"] == pytest.approx(equalized_odds_difference(y,pred,sensitive_features=groups,method="between_groups"),abs=1e-6)
    reference=MetricFrame(metrics={"selection_rate":selection_rate,"tpr":true_positive_rate,"fpr":false_positive_rate},y_true=y,y_pred=pred,sensitive_features=groups)
    for group in groups.astype(str).tolist()[::10]:
        app_group=next(row for row in out["groups"] if row["group"]==group)
        assert app_group["selection_rate"]==pytest.approx(reference.by_group.loc[group,"selection_rate"],abs=1e-6)
        assert app_group["tpr"]==pytest.approx(reference.by_group.loc[group,"tpr"],abs=1e-6)
        assert app_group["fpr"]==pytest.approx(reference.by_group.loc[group,"fpr"],abs=1e-6)


def test_eligible_screening_keeps_small_group_visible():
    frame = pd.DataFrame({"group": ["A"]*100+["B"]*100+["C"]*5,
                          "pred": [1]*70+[0]*30+[1]*40+[0]*60+[1]*5})
    out = calculate_audit(frame, config(true_column=None))
    e = out["metrics"]["eligible_screening"]
    assert out["metrics"]["demographic_parity_difference"] == pytest.approx(.6)
    assert e["excluded_groups"] == ["C"]
    assert out["groups"][2]["sample_status"] == "Insufficient group data"
    assert e["dpd"] == pytest.approx(.3)
    assert e["dir"] == pytest.approx(4/7)
    assert e["high_group"] == "A" and e["low_group"] == "B"
    assert e["pairs"][0]["adjusted_p"] < .05

def test_eligible_screening_requires_two_groups():
    frame = pd.DataFrame({"group": ["A"]*40+["B"]*5, "pred": [1]*45})
    e = calculate_audit(frame, config(true_column=None))["metrics"]["eligible_screening"]
    assert e["dpd"] is None and e["pairs"] == []
