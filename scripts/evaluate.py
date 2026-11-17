"""Evaluate the six known-answer scenarios and independent metric references."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bias_detector.metrics import AuditSettings, calculate_audit
from bias_detector.synthetic import SCENARIOS, generate_scenario, scenario_expected_status


def compare_fairlearn(frame, group_col, pred, truth):
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference
    settings=AuditSettings(group_column=group_col,prediction_column="_pred",true_column="_truth",minimum_group_size=1)
    work=frame[[group_col]].copy()
    work["_pred"]=np.asarray(pred,dtype=int)
    work["_truth"]=np.asarray(truth,dtype=int)
    result=calculate_audit(work,settings)
    sensitive=work[group_col].astype(str).to_numpy(); y=work["_truth"].to_numpy(); yhat=work["_pred"].to_numpy()
    mf=MetricFrame(metrics={"selection_rate":selection_rate,"tpr":true_positive_rate,"fpr":false_positive_rate},y_true=y,y_pred=yhat,sensitive_features=sensitive)
    pairs=[]
    for row in result["groups"]:
        group=row["group"]
        for key,ref_key in (("selection_rate","selection_rate"),("tpr","tpr"),("fpr","fpr")):
            app_value=row.get(key); ref_value=mf.by_group.loc[group,ref_key]
            if app_value is not None and np.isfinite(ref_value):
                pairs.append({"metric":key,"group":group,"app":float(app_value),"reference":float(ref_value),"absolute_error":abs(float(app_value)-float(ref_value))})
    m=result["metrics"]
    aggregate=[
        ("demographic_parity_difference",m["demographic_parity_difference"],demographic_parity_difference(y,yhat,sensitive_features=sensitive,method="between_groups")),
        ("disparate_impact_ratio",m["disparate_impact_ratio"],demographic_parity_ratio(y,yhat,sensitive_features=sensitive,method="between_groups")),
        ("equalised_odds_difference",m["equalised_odds_difference"],equalized_odds_difference(y,yhat,sensitive_features=sensitive,method="between_groups")),
    ]
    for name,app_value,ref_value in aggregate:
        if app_value is not None and ref_value is not None and np.isfinite(ref_value):
            pairs.append({"metric":name,"group":"all groups","app":float(app_value),"reference":float(ref_value),"absolute_error":abs(float(app_value)-float(ref_value))})
    errors=[p["absolute_error"] for p in pairs]
    epsilon=1e-6
    return {"comparisons":len(pairs),"within_tolerance":sum(x<=epsilon for x in errors),"accuracy_pct":100*sum(x<=epsilon for x in errors)/len(errors) if errors else None,"mae":float(np.mean(errors)) if errors else None,"maximum_absolute_error":max(errors,default=None),"details":pairs}


def main():
    results=[]
    binary_detection=[]
    for name in SCENARIOS:
        expected=scenario_expected_status(name)
        for seed in range(100):
            frame=generate_scenario(name,seed=seed)
            start=time.perf_counter()
            result=calculate_audit(frame,AuditSettings(group_column="group",prediction_column="prediction",true_column="truth",minimum_group_size=30))
            elapsed=time.perf_counter()-start
            metrics=result["metrics"]
            if not metrics["reliable"]:
                observed="insufficient_data"
            elif metrics["demographic_parity_flag"] and metrics["selection_rate_comparison"]["p_value_two_sided"] is not None and metrics["selection_rate_comparison"]["p_value_two_sided"] < .05:
                observed="supported_dp_flag"
            elif metrics["equal_opportunity_difference"] and metrics["equal_opportunity_difference"] >= .20:
                observed="equal_opportunity_flag_only"
            else:
                observed="no_dp_flag"
            results.append({"scenario":name,"seed":seed,"expected":expected,"observed":observed,"correct":expected==observed,"seconds":elapsed})
            if expected in ("no_dp_flag","supported_dp_flag"):
                binary_detection.append({"y_true":int(expected=="supported_dp_flag"),"y_pred":int(observed=="supported_dp_flag")})
    groups={}
    for name in SCENARIOS:
        rows=[r for r in results if r["scenario"]==name]
        groups[name]={"n":len(rows),"expected_status":rows[0]["expected"],"correct":sum(r["correct"] for r in rows),"accuracy_pct":100*sum(r["correct"] for r in rows)/len(rows),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))}
    tp=sum(x["y_true"]==1 and x["y_pred"]==1 for x in binary_detection)
    tn=sum(x["y_true"]==0 and x["y_pred"]==0 for x in binary_detection)
    fp=sum(x["y_true"]==0 and x["y_pred"]==1 for x in binary_detection)
    fn=sum(x["y_true"]==1 and x["y_pred"]==0 for x in binary_detection)
    binary_scores={"n":len(binary_detection),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"accuracy_pct":100*(tp+tn)/len(binary_detection),"precision_pct":100*tp/(tp+fp) if tp+fp else None,"recall_pct":100*tp/(tp+fn) if tp+fn else None}
    output={"cases":len(results),"overall_expected_status_accuracy_pct":100*sum(r["correct"] for r in results)/len(results),"binary_dp_detection_s1_s4":binary_scores,"scenario_results":groups,"interpretation":"Expected-status accuracy spans the binary DP cases, insufficient-data abstentions, and the equal-opportunity case. Binary DP precision/recall/accuracy are separately calculated over S1-S4 (400 cases)."}
    evaluation=ROOT/"evaluation"
    evaluation.mkdir(exist_ok=True)
    benchmark={}
    fairlearn_references={}
    sample_paths={"COMPAS":ROOT/"data/samples/compas.csv","Adult held-out":ROOT/"data/samples/adult_holdout.csv"}
    for label,path in sample_paths.items():
        if not path.exists():
            benchmark[label]={"status":"not prepared"}
            continue
        frame=pd.read_csv(path)
        if label=="COMPAS":
            cfg=AuditSettings(group_column="race",prediction_column="risk_score",score_mode=True,score_threshold=5,true_column="recidivism_2yr",minimum_group_size=30)
            fairlearn_references[label]=compare_fairlearn(frame,"race",frame["risk_score"].ge(5).astype(int),frame["recidivism_2yr"])
        else:
            cfg=AuditSettings(group_column="race",prediction_column="income_prediction",true_column="income_true",minimum_group_size=30)
            fairlearn_references[label]=compare_fairlearn(frame,"race",frame["income_prediction"],frame["income_true"])
        durations=[]
        for run in range(11):
            started=time.perf_counter()
            measured=calculate_audit(frame,cfg)
            group_data=pd.DataFrame(measured["groups"])
            px.bar(group_data,x="group",y="selection_rate")
            elapsed=time.perf_counter()-started
            if run: durations.append(elapsed)
        benchmark[label]={"rows":len(frame),"runs":len(durations),"mean_analysis_and_chart_build_seconds":float(np.mean(durations)),"max_seconds":float(max(durations))}
    adult_path=ROOT/"data/samples/adult_holdout.csv"
    if adult_path.exists():
        adult=pd.read_csv(adult_path)
        stress=adult.iloc[np.arange(48_842)%len(adult)].reset_index(drop=True)
        durations=[]
        cfg=AuditSettings(group_column="race",prediction_column="income_prediction",true_column="income_true",minimum_group_size=30)
        for run in range(11):
            started=time.perf_counter()
            calculate_audit(stress,cfg)
            elapsed=time.perf_counter()-started
            if run: durations.append(elapsed)
        benchmark["Adult 48,842-row stress fixture"]={"rows":len(stress),"runs":len(durations),"mean_metric_seconds":float(np.mean(durations)),"max_metric_seconds":float(max(durations)),"note":"Rows are repeated from the held-out Adult sample for speed testing only, not independent people; excluded from fairness evaluation."}
    output["performance_benchmarks"]=benchmark
    output["fairlearn_reference_checks"]=fairlearn_references
    (evaluation/"synthetic_scenarios.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
