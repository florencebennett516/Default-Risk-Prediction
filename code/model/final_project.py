from pathlib import Path
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score, roc_curve,
)
from xgboost import DMatrix
from model_optimization_features import build_features


MODEL_NAMES = ["基线_XGBoost", "调优_XGBoost"]
TARGET = "isDefault"
SEED = 42
MAX_REJECTION_RATE = 0.30
BUSINESS_POLICY = "业务约束阈值"


def save_table(frame, output_dir, name):
    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def markdown_table(frame, digits=4):
    displayed = frame.copy()
    for column in displayed.select_dtypes(include="number"):
        displayed[column] = displayed[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    rows = ["| " + " | ".join(map(str, displayed.columns)) + " |"]
    rows.append("| " + " | ".join(["---"] * len(displayed.columns)) + " |")
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in displayed.itertuples(index=False, name=None))
    return "\n".join(rows)


def clean_raw(raw):
    cleaned = raw.copy()
    employment = cleaned["employmentLength"].astype("string").str.strip()
    employment = employment.replace({"10+ years": "10 years", "< 1 year": "0 years"})
    years = employment.str.extract(r"^(\d+)\s*years?$", expand=False)
    if (employment.notna() & years.isna()).any():
        raise ValueError("工龄存在不能解析的非空值。")
    cleaned["employmentLength"] = pd.to_numeric(years).astype("float64")
    for column, format_string in {"issueDate": "%Y-%m-%d", "earliesCreditLine": "%b-%Y"}.items():
        cleaned[column] = pd.to_datetime(cleaned[column], format=format_string, errors="raise")
        if cleaned[column].isna().any():
            raise ValueError(f"{column} 存在空日期。")
    numeric = cleaned.columns.difference(["grade", "subGrade", "issueDate", "earliesCreditLine"])
    for column in numeric:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    cleaned.loc[(cleaned["dti"] < 0) | (cleaned["dti"] == 999), "dti"] = np.nan
    cleaned["issueYear"] = cleaned["issueDate"].dt.year
    cleaned["earliesCreditYear"] = cleaned["earliesCreditLine"].dt.year
    if not cleaned["id"].is_unique or cleaned["id"].isna().any():
        raise ValueError("贷款 id 重复或缺失。")
    if not cleaned[TARGET].isin([0, 1]).all():
        raise ValueError("标签必须为完整的 0/1。")
    return cleaned


def verify_cleaned(cleaned, snapshot_path, output_dir):
    snapshot = pd.read_csv(snapshot_path, parse_dates=["issueDate", "earliesCreditLine"])
    if len(cleaned) != len(snapshot) or set(cleaned.columns) != set(snapshot.columns):
        raise AssertionError("清洗重放的形状或列名与历史快照不一致。")
    rows = []
    for column in snapshot:
        left, right = cleaned[column], snapshot[column]
        if pd.api.types.is_numeric_dtype(right):
            matched = np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True, rtol=1e-12, atol=1e-12)
        else:
            matched = left.eq(right).to_numpy()
        rows.append({"字段": column, "不一致行数": int((~matched).sum())})
    audit = pd.DataFrame(rows)
    save_table(audit, output_dir, "cleaning_replay_audit.csv")
    if audit["不一致行数"].sum():
        raise AssertionError("清洗重放与历史快照不一致，详见 cleaning_replay_audit.csv。")
    return snapshot, audit


def evaluate(target, probability, threshold=0.5):
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("预测概率不在 [0, 1] 内。")
    prediction = probability >= threshold
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probability)
    return {
        "阈值": float(threshold), "样本数": len(target), "违约率": float(target.mean()),
        "AUC": roc_auc_score(target, probability), "KS": float((true_positive_rate - false_positive_rate).max()),
        "PR-AUC": average_precision_score(target, probability),
        "Precision": precision_score(target, prediction, zero_division=0),
        "Recall": recall_score(target, prediction, zero_division=0), "F1": f1_score(target, prediction, zero_division=0),
        "标记比例": float(prediction.mean()), "准确率": float((prediction == target).mean()),
        "通过组违约率": float(target[~prediction].mean()) if (~prediction).any() else np.nan,
        "Brier": brier_score_loss(target, probability), "LogLoss": log_loss(target, probability, labels=[0, 1]),
        "TN": int(true_negative), "FP": int(false_positive), "FN": int(false_negative), "TP": int(true_positive),
    }


def validate_saved_results(predictions, source, output_dir):
    rows = [{"集合": split_name, "模型": name, **evaluate(frame[TARGET], frame[name])}
            for split_name, frame in predictions.items() for name in MODEL_NAMES]
    metrics = pd.DataFrame(rows)
    for split_name, filename in [("验证集", "validation_metrics.csv"), ("测试集", "test_metrics.csv")]:
        historical = pd.read_csv(source / filename).set_index("模型")
        recalculated = metrics.loc[metrics["集合"] == split_name].set_index("模型")
        for name in MODEL_NAMES:
            for column in ["AUC", "KS", "PR-AUC", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP", "样本数"]:
                np.testing.assert_allclose(recalculated.loc[name, column], historical.loc[name, column], atol=1e-10)
    save_table(metrics, output_dir, "recomputed_metrics.csv")
    return metrics


def threshold_candidates(validation, selected_model, output_dir,
                         max_rejection_rate=MAX_REJECTION_RATE):
    if not 0 <= max_rejection_rate < 1:
        raise ValueError("拒绝率上限须在 [0, 1) 内。")
    target = validation[TARGET].to_numpy()
    scores = validation[selected_model].to_numpy()
    if len(target) == 0 or not np.isin(target, [0, 1]).all() or np.unique(target).size != 2:
        raise ValueError("阈值选择需要同时包含正常和违约标签的非空验证集。")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("验证集预测概率必须为 [0, 1] 内的有限值。")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(target, scores, drop_intermediate=False)
    thresholds[0] = np.nextafter(scores.max(), np.inf)
    true_positive = np.rint(true_positive_rate * target.sum()).astype(int)
    false_positive = np.rint(false_positive_rate * (len(target) - target.sum())).astype(int)
    flagged = true_positive + false_positive
    passed = len(target) - flagged
    pass_defaults = target.sum() - true_positive
    pass_bad_rate = np.divide(pass_defaults, passed, out=np.full_like(passed, np.nan, dtype=float), where=passed > 0)
    precision = np.divide(true_positive, flagged, out=np.zeros_like(true_positive, dtype=float), where=flagged > 0)
    denominator = precision + true_positive_rate
    f1_values = np.divide(2 * precision * true_positive_rate, denominator, out=np.zeros_like(precision), where=denominator > 0)
    search = pd.DataFrame({"阈值": thresholds, "KS": true_positive_rate - false_positive_rate,
                           "Recall": true_positive_rate, "Precision": precision, "F1": f1_values,
                            "标记比例": flagged / len(target), "拒绝率": flagged / len(target),
                            "通过组违约率": pass_bad_rate, "FPR": false_positive_rate})
    search = search.loc[np.isfinite(search["阈值"]) & search["阈值"].ge(0)].copy()
    search["满足业务约束"] = search["拒绝率"].le(max_rejection_rate) & np.isfinite(search["通过组违约率"])
    save_table(search, output_dir, "validation_threshold_search.csv")
    feasible = search.loc[search["满足业务约束"]]
    if feasible.empty:
        raise ValueError(
            f"验证集不存在满足拒绝率≤{max_rejection_rate:.1%}且通过组坏率有效的阈值。"
            "未应用阈值，也不会回退到0.5。"
            "完整搜索见 validation_threshold_search.csv。"
        )
    business_candidate = feasible.sort_values(
        ["通过组违约率", "拒绝率", "阈值"], ascending=[True, True, False],
    ).iloc[0]
    definitions = [
        ("验证集最大KS", search.sort_values(["KS", "阈值"], ascending=[False, False]).iloc[0]),
        ("验证集最大F1", search.sort_values(["F1", "阈值"], ascending=[False, False]).iloc[0]),
        ("召回至少60%示例", search.loc[search["Recall"] >= 0.6].sort_values("阈值", ascending=False).iloc[0]),
        ("复核容量20%示例", search.loc[search["标记比例"] <= 0.2].sort_values(["Recall", "阈值"], ascending=[False, False]).iloc[0]),
    ]
    policies = [{"策略": "原固定阈值", "阈值": 0.5}]
    policies.extend({"策略": name, "阈值": float(row["阈值"])} for name, row in definitions)
    policies.append({"策略": BUSINESS_POLICY, "阈值": float(business_candidate["阈值"]),
                     "拒绝率上限": float(max_rejection_rate), "优化目标": "最小化验证集通过组坏率",
                     "验证集通过组坏率": float(business_candidate["通过组违约率"])})
    save_table(pd.DataFrame(policies), output_dir, "threshold_policies.csv")
    return policies, search


def evaluate_policies(predictions, selected_model, policies, output_dir):
    rows = [{"策略": policy["策略"], "集合": split_name, **evaluate(frame[TARGET], frame[selected_model], policy["阈值"])}
            for policy in policies for split_name, frame in predictions.items()]
    table = pd.DataFrame(rows)
    save_table(table, output_dir, "threshold_scenarios.csv")
    return table


def apply_business_threshold(bundle, predictions, policies, output_dir):
    policy = next(policy for policy in policies if policy["策略"] == BUSINESS_POLICY)
    threshold = policy["阈值"]
    selected_model = bundle["selected_model"]
    validation = predictions["验证集"]
    validation_metrics = evaluate(validation[TARGET], validation[selected_model], threshold)
    if (validation_metrics["标记比例"] > policy["拒绝率上限"]
            or not np.isfinite(validation_metrics["通过组违约率"])):
        raise ValueError("候选阈值未通过验证集业务约束复核，未保存应用模型。")
    applied_bundle = dict(bundle)
    applied_bundle["original_threshold"] = bundle.get("original_threshold", bundle["threshold"])
    applied_bundle["threshold"] = threshold
    applied_bundle["threshold_policy"] = dict(policy)
    rows = []
    for split_name, frame in predictions.items():
        metrics = evaluate(frame[TARGET], frame[selected_model], threshold)
        rows.append({"集合": split_name, "模型": selected_model, **metrics,
                     "拒绝率达标": metrics["标记比例"] <= policy["拒绝率上限"],
                     "优化目标": policy["优化目标"]})
    test_predictions = predictions["测试集"].copy()
    test_predictions["最终预测"] = (test_predictions[selected_model] >= threshold).astype(int)
    test_predictions["应用阈值"] = threshold
    save_table(test_predictions, output_dir, "test_predictions_with_threshold.csv")
    save_table(pd.DataFrame(rows), output_dir, "applied_threshold_metrics.csv")
    joblib.dump(applied_bundle, Path(output_dir) / "final_model.joblib")
    return applied_bundle


def paired_bootstrap(test, selected_model, output_dir, repetitions=300):
    target = test[TARGET].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    structures = {}
    for name in MODEL_NAMES:
        scores = test[name].to_numpy()
        order = np.argsort(scores, kind="stable")
        structures[name] = (order, np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1])

    def weighted_auc(weights, name):
        order, starts = structures[name]
        positive = np.add.reduceat((weights * target)[order], starts)
        negative = np.add.reduceat((weights * (1 - target))[order], starts)
        return float((positive * (np.cumsum(negative) - negative / 2)).sum() / (positive.sum() * negative.sum()))

    for name in MODEL_NAMES:
        np.testing.assert_allclose(weighted_auc(np.ones(len(test)), name), roc_auc_score(target, test[name]), atol=1e-12)
    month_codes, months = pd.factorize(test["issueDate"].dt.to_period("M"))
    rows = []
    selected_point = roc_auc_score(target, test[selected_model])
    baseline_point = roc_auc_score(target, test[MODEL_NAMES[0]])
    for method in ["逐笔配对bootstrap", "整月分块配对bootstrap"]:
        auc_values, difference_values = [], []
        for _ in range(repetitions):
            if method == "逐笔配对bootstrap":
                weights = np.bincount(rng.integers(0, len(test), len(test)), minlength=len(test))
            else:
                month_weights = np.bincount(rng.integers(0, len(months), len(months)), minlength=len(months))
                weights = month_weights[month_codes]
            if (weights * target).sum() == 0 or (weights * (1 - target)).sum() == 0:
                continue
            selected_auc = weighted_auc(weights, selected_model)
            auc_values.append(selected_auc)
            difference_values.append(selected_auc - weighted_auc(weights, MODEL_NAMES[0]))
        for name, point, values in [("最终模型AUC", selected_point, auc_values),
                                     ("最终减基线AUC", selected_point - baseline_point, difference_values)]:
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append({"方法": method, "指标": name, "点估计": point,
                         "95%下限": lower, "95%上限": upper, "有效重采样次数": len(values)})
    intervals = pd.DataFrame(rows)
    save_table(intervals, output_dir, "auc_bootstrap_intervals.csv")
    return intervals


def stability_tables(test, selected_model, output_dir):
    frame = test.copy()
    frame["季度"] = frame["issueDate"].dt.to_period("Q").astype(str)
    quarterly = pd.DataFrame([
        {"季度": quarter, **evaluate(group[TARGET], group[selected_model])}
        for quarter, group in frame.groupby("季度", sort=True) if group[TARGET].nunique() == 2
    ])
    frame["风险档"] = pd.qcut(frame[selected_model], q=10, labels=False, duplicates="drop") + 1
    risk = frame.groupby("风险档", observed=True).agg(
        样本数=(TARGET, "size"), 违约数=(TARGET, "sum"), 违约率=(TARGET, "mean"),
        平均预测概率=(selected_model, "mean"), 最低分=(selected_model, "min"), 最高分=(selected_model, "max"),
    ).reset_index().sort_values("风险档", ascending=False)
    risk["累计覆盖率"] = risk["样本数"].cumsum() / len(frame)
    risk["累计违约捕获率"] = risk["违约数"].cumsum() / frame[TARGET].sum()
    risk["Lift"] = risk["违约率"] / frame[TARGET].mean()
    frame["概率区间"] = pd.cut(frame[selected_model], bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = frame.groupby("概率区间", observed=True).agg(
        样本数=(TARGET, "size"), 平均预测概率=(selected_model, "mean"), 实际违约率=(TARGET, "mean"),
    ).reset_index()
    calibration["绝对误差"] = (calibration["平均预测概率"] - calibration["实际违约率"]).abs()
    for table, filename in [(quarterly, "test_quarterly_metrics.csv"), (risk, "test_risk_deciles.csv"), (calibration, "test_calibration.csv")]:
        save_table(table, output_dir, filename)
    return quarterly, risk, calibration


def drift_table(development, test_data, output_dir):
    rows = []
    for column in ["grade", "subGrade", "term", "dti", "ficoRangeLow", "annualIncome", "revolUtil", "n14"]:
        reference, current = development[column], test_data[column]
        if pd.api.types.is_numeric_dtype(reference) and reference.nunique() > 10:
            edges = np.unique(reference.quantile(np.linspace(0, 1, 11)).dropna())
            edges[0], edges[-1] = -np.inf, np.inf
            reference_bins = pd.cut(reference, edges, include_lowest=True).astype("string").fillna("缺失")
            current_bins = pd.cut(current, edges, include_lowest=True).astype("string").fillna("缺失")
        else:
            reference_bins = reference.astype("string").fillna("缺失")
            current_bins = current.astype("string").fillna("缺失")
        levels = pd.Index(reference_bins.unique().tolist() + current_bins.unique().tolist()).unique()
        reference_counts = reference_bins.value_counts().reindex(levels, fill_value=0) + 0.5
        current_counts = current_bins.value_counts().reindex(levels, fill_value=0) + 0.5
        reference_share, current_share = reference_counts / reference_counts.sum(), current_counts / current_counts.sum()
        psi = ((current_share - reference_share) * np.log(current_share / reference_share)).sum()
        rows.append({"字段": column, "PSI": psi, "开发集缺失率": reference.isna().mean(), "测试集缺失率": current.isna().mean()})
    result = pd.DataFrame(rows).sort_values("PSI", ascending=False)
    save_table(result, output_dir, "raw_feature_drift.csv")
    return result


def transform_batch(bundle, data):
    return bundle["preprocessor"].transform(build_features(data)).loc[:, bundle["selected_features"]].astype("float32")


def replay_predictions(bundle, test_data, saved_test, output_dir, batch_size=4000):
    expected = test_data[["id", "issueDate", TARGET]].merge(saved_test, on=["id", "issueDate", TARGET], validate="one_to_one", how="left")
    if len(expected) != len(saved_test) or expected[MODEL_NAMES].isna().any().any():
        raise AssertionError("测试样本与已保存预测不能一一对应。")
    probabilities = {name: [] for name in MODEL_NAMES}
    for offset in range(0, len(test_data), batch_size):
        ready = transform_batch(bundle, test_data.iloc[offset:offset + batch_size])
        for name in MODEL_NAMES:
            probabilities[name].append(bundle["models"][name].predict_proba(ready)[:, 1])
    rows = []
    for name in MODEL_NAMES:
        actual = np.concatenate(probabilities[name])
        np.testing.assert_allclose(actual, expected[name], atol=1e-7, rtol=1e-6)
        rows.append({"模型": name, "核验样本数": len(actual), "最大概率差": np.max(np.abs(actual - expected[name]))})
    audit = pd.DataFrame(rows)
    save_table(audit, output_dir, "prediction_replay_audit.csv")
    return audit


def feature_group(name):
    if name.startswith(("grade_", "subGrade_")) or name in ["grade", "subGrade"]:
        return "既有评级及其交叉统计"
    if name.startswith("anonymous_") or re.fullmatch(r"n\d+", name):
        return "匿名行为特征"
    if name.startswith(("employmentTitle", "title", "postCode", "regionCode")):
        return "职业标题地区及交叉统计"
    if name.startswith("fico"):
        return "FICO信用水平"
    if name in ["term", "loanAmnt", "annual_loan_income_ratio", "installment"]:
        return "借款合同与偿债安排"
    return "其他财务与征信特征"


def explain_model(bundle, development, output_dir, sample_size=2000):
    sample = development.sample(min(sample_size, len(development)), random_state=SEED)
    ready = transform_batch(bundle, sample)
    model = bundle["models"][bundle["selected_model"]]
    booster = model.get_booster()
    contribution = booster.predict(DMatrix(ready), pred_contribs=True)
    np.testing.assert_allclose(1 / (1 + np.exp(-contribution.sum(axis=1))), model.predict_proba(ready)[:, 1], atol=1e-6)
    gain = booster.get_score(importance_type="total_gain")
    importance = pd.DataFrame({"特征": ready.columns, "平均绝对SHAP": np.abs(contribution[:, :-1]).mean(axis=0),
                               "total_gain": [gain.get(name, 0) for name in ready.columns]})
    importance["SHAP占比"] = importance["平均绝对SHAP"] / importance["平均绝对SHAP"].sum()
    importance["分组"] = importance["特征"].map(feature_group)
    importance = importance.sort_values("平均绝对SHAP", ascending=False)
    group = importance.groupby("分组", sort=False)["SHAP占比"].sum().sort_values(ascending=False).reset_index()
    directions = []
    for column in ["term", "ficoRange_mean", "dti", "n14", "anonymous_nonzero_count", "annualIncome"]:
        if column in ready:
            position = ready.columns.get_loc(column)
            correlation = pd.Series(ready[column].to_numpy()).corr(pd.Series(contribution[:, position]), method="spearman")
            directions.append({"特征": column, "取值与SHAP的Spearman相关": correlation, "解释样本数": len(ready)})
    direction_table = pd.DataFrame(directions)
    for table, name in [(importance, "final_feature_importance.csv"), (group, "feature_group_importance.csv"), (direction_table, "feature_shap_direction.csv")]:
        save_table(table, output_dir, name)
    return importance, group, direction_table


def make_figures(test, selected_model, quarterly, risk, calibration, importance, search, output_dir):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in MODEL_NAMES:
        false_positive_rate, true_positive_rate, _ = roc_curve(test[TARGET], test[name])
        axes[0].plot(false_positive_rate, true_positive_rate, label=f"{name}: {roc_auc_score(test[TARGET], test[name]):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="时间外测试 ROC")
    axes[0].legend()
    axes[1].plot(calibration["平均预测概率"], calibration["实际违约率"], "o-")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(xlabel="平均预测概率", ylabel="实际违约率", title="固定0.1宽度概率校准图")
    axes[2].plot(quarterly["季度"], quarterly["AUC"], "o-")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].set(ylabel="AUC", title="分季度区分能力")
    figure.tight_layout()
    figure.savefig(output_dir / "test_evaluation.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1, 2, 1]})
    ordered = risk.sort_values("风险档")
    axes[0].bar(ordered["风险档"], ordered["违约率"])
    axes[0].axhline(test[TARGET].mean(), linestyle="--", color="red")
    axes[0].set(xlabel="风险档：1最低，10最高", ylabel="观察违约率", title="风险分层")
    top = importance.head(12).iloc[::-1]
    axes[1].barh(top["特征"], top["平均绝对SHAP"])
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set(xlabel="mean absolute SHAP（log-odds）", title="最终模型关键特征")
    sampled = search.iloc[::max(1, len(search) // 1500)].sort_values("阈值")
    for column in ["Precision", "Recall", "标记比例"]:
        axes[2].plot(sampled["阈值"], sampled[column], label=column)
    axes[2].axvline(0.5, color="gray", linestyle="--")
    axes[2].set(xlabel="概率阈值", title="仅验证集：阈值权衡")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "business_analysis.png", dpi=160)
    plt.close(figure)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(project_dir, source, output_dir, bundle, policies, repetitions):
    files = [project_dir / "data" / "credit_data(1).csv", project_dir / "data" / "data_processing.csv",
             source / "final_model.joblib", source / "validation_predictions.csv", source / "test_predictions.csv",
             source / "best_parameters.csv", project_dir / "code" / "model" / "xgboost_optimization.py",
             project_dir / "code" / "model" / "model_optimization_features.py", Path(__file__),
              Path(__file__).with_name("final_report_template.md"),
              Path(__file__).with_name("final_report_template_boss.md")]
    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(), "project_dir": str(project_dir), "source_dir": str(source),
        "selected_model": bundle["selected_model"], "original_threshold": bundle.get("original_threshold", bundle["threshold"]),
        "applied_threshold": bundle["threshold"], "threshold_policy": bundle.get("threshold_policy"),
        "feature_count": len(bundle["selected_features"]), "seed": SEED,
        "bootstrap_repetitions": repetitions, "shap_sample_size": 2000,
        "threshold_status": "在验证集拒绝率上限内最小化通过组坏率并应用；另存模型及分类结果，跨时期容量须另行验证",
        "threshold_policies": policies, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "xgboost", "optuna", "joblib", "matplotlib", "ipykernel"]},
        "source_files": [{"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size} for path in files],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def render_html(markdown, destination, title):
    def inline(text):
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)

    lines, blocks, position = markdown.splitlines(), [], 0
    while position < len(lines):
        line = lines[position]
        if line.startswith("|"):
            table_lines = []
            while position < len(lines) and lines[position].startswith("|"):
                table_lines.append(lines[position])
                position += 1
            rows = [[part.strip() for part in row.strip("|").split("|")] for row in table_lines]
            table = "<div class='table-scroll'><table><thead><tr>" + "".join(f"<th>{inline(value)}</th>" for value in rows[0]) + "</tr></thead><tbody>"
            table += "".join("<tr>" + "".join(f"<td>{inline(value)}</td>" for value in row) + "</tr>" for row in rows[2:])
            blocks.append(table + "</tbody></table></div>")
            continue
        image = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if image:
            blocks.append(f'<figure><img src="{html.escape(image[2], quote=True)}" alt="{html.escape(image[1], quote=True)}"></figure>')
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 6)
            blocks.append(f"<h{level}>{inline(line.lstrip('#').strip())}</h{level}>")
        elif line.startswith("- "):
            blocks.append(f"<p>• {inline(line[2:])}</p>")
        elif line.strip():
            blocks.append(f"<p>{inline(line)}</p>")
        position += 1
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title)
    document += """</title><style>
body{font-family:'Microsoft YaHei',sans-serif;color:#183047;background:#f4f7fa;margin:0;line-height:1.85}
main{max-width:1180px;margin:30px auto;padding:36px;background:white;border-top:6px solid #087e8b}
h1,h2,h3{color:#123b56;line-height:1.4}h2{margin-top:38px;border-bottom:1px solid #d8e5ed;padding-bottom:10px}
table{border-collapse:collapse;min-width:650px;font-size:13px;width:100%}th{background:#123b56;color:white}td,th{padding:8px 12px;border:1px solid #d8e5ed;text-align:left}tr:nth-child(even){background:#f3f8fa}
.table-scroll{overflow:auto;margin:20px 0}img{max-width:100%;height:auto}figure{margin:24px 0}code{background:#eef4f7;padding:2px 4px;border-radius:3px}
@media print{body{background:white}main{margin:0;padding:0;border:0}h2{break-after:avoid}tr,img{break-inside:avoid}}
</style><main>""" + "\n".join(blocks) + "</main></html>"
    Path(destination).write_text(document, encoding="utf-8")


def write_report(project_dir, source, output_dir, bundle, metrics, policies_table,
                 intervals, quarterly, risk, calibration, importance, groups, directions, drift, manifest):
    selected = bundle["selected_model"]
    final = metrics.loc[(metrics["集合"] == "测试集") & (metrics["模型"] == selected)].iloc[0]
    validation = metrics.loc[(metrics["集合"] == "验证集") & (metrics["模型"] == selected)].iloc[0]
    baseline = metrics.loc[(metrics["集合"] == "测试集") & (metrics["模型"] == MODEL_NAMES[0])].iloc[0]
    validation_baseline = metrics.loc[(metrics["集合"] == "验证集") & (metrics["模型"] == MODEL_NAMES[0])].iloc[0]
    candidate = policies_table.loc[(policies_table["策略"] == "验证集最大KS") & (policies_table["集合"] == "测试集")].iloc[0]
    applied = policies_table.loc[(policies_table["策略"] == BUSINESS_POLICY) & (policies_table["集合"] == "测试集")].iloc[0]
    summary = pd.read_csv(source / "sample_summary.csv")
    cv = pd.read_csv(source / "cv_summary.csv")
    folds = pd.read_csv(source / "cv_fold_metrics.csv")
    cv_values = folds.loc[folds["模型"] == selected, "AUC"]
    top, bottom = risk.iloc[0], risk.iloc[-1]
    interval = intervals.loc[(intervals["方法"] == "整月分块配对bootstrap") & (intervals["指标"] == "最终减基线AUC")].iloc[0]
    if interval["95%下限"] <= 0 <= interval["95%上限"]:
        interpretation = "整月分块重采样的模型差值区间包含零，不支持稳定优于基线的结论。"
    elif interval["95%上限"] < 0:
        interpretation = "整月分块重采样的模型差值区间整体为负，在该重采样假设下支持调优模型略低于基线；不能据此声称其具有泛化优势。"
    else:
        interpretation = "整月分块重采样的模型差值区间整体为正，但该结论仅适用于固定模型及当前样本，不能替代未来样本验证。"
    report_dir = project_dir / "docs" / "最终报告" if source.name == "xgboost_optimization" else output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_prefix = Path(os.path.relpath(output_dir, report_dir)).as_posix()
    sample_display = summary.copy()
    sample_display["违约率"] = sample_display["违约率"].map(lambda value: f"{value:.2%}")
    sample_display["样本数"] = sample_display["样本数"].astype(str)
    matrix = pd.DataFrame({"实际类别": ["正常", "违约"], "预测正常": [int(final["TN"]), int(final["FN"])],
                           "预测违约": [int(final["FP"]), int(final["TP"])]}).astype(str)
    test = pd.read_csv(source / "test_predictions.csv", parse_dates=["issueDate"])
    fold_summary = pd.read_csv(source / "fold_summary.csv")
    latest = quarterly.iloc[-1]
    capacity = policies_table.loc[policies_table["策略"] == "复核容量20%示例"].set_index("集合")
    drift_index = drift.set_index("字段")
    values = {
        "cv_folds": len(fold_summary), "n_trials": len(pd.read_csv(source / "trials_XGBoost.csv")),
        "first_fold_size": int(fold_summary.iloc[0]["训练样本数"]), "last_fold_size": int(fold_summary.iloc[-1]["训练样本数"]),
        "latest_quarter": latest["季度"], "latest_quarter_size": int(latest["样本数"]),
        "latest_quarter_defaults": int(latest["TP"] + latest["FN"]), "latest_quarter_bad_rate": latest["违约率"],
        "latest_quarter_auc": latest["AUC"], "n14_psi": drift_index.loc["n14", "PSI"],
        "n14_development_missing": drift_index.loc["n14", "开发集缺失率"], "n14_test_missing": drift_index.loc["n14", "测试集缺失率"],
        "revol_psi": drift_index.loc["revolUtil", "PSI"], "capacity_threshold": capacity.loc["验证集", "阈值"],
        "capacity_validation_flagged": capacity.loc["验证集", "标记比例"], "capacity_test_flagged": capacity.loc["测试集", "标记比例"],
        "capacity_test_recall": capacity.loc["测试集", "Recall"], "capacity_test_precision": capacity.loc["测试集", "Precision"],
        "execution_scope": "本次执行了训练分支与报告分析。" if manifest.get("training_executed_this_run") else "本次执行了完整默认重放与报告分析，没有重新执行调参或训练分支。",
        "date": manifest["generated_at"][:10], "selected_model": selected, "feature_count": len(bundle["selected_features"]),
        "sample_table": markdown_table(sample_display), "total_size": int(summary["样本数"].sum()),
        "test_size": int(final["样本数"]), "normal_count": int(final["TN"] + final["FP"]),
        "default_count": int(final["TP"] + final["FN"]), "auc": final["AUC"], "ks": final["KS"],
        "ap": final["PR-AUC"], "test_bad_rate": final["违约率"], "ap_lift": final["PR-AUC"] / final["违约率"],
        "validation_auc": validation["AUC"], "validation_baseline_auc": validation_baseline["AUC"],
        "baseline_auc": baseline["AUC"], "auc_shortfall": baseline["AUC"] - final["AUC"],
        "confusion_table": markdown_table(matrix), "flagged_count": int(final["TP"] + final["FP"]),
        "flagged_rate": final["标记比例"], "true_positive": int(final["TP"]), "precision": final["Precision"],
        "recall": final["Recall"], "false_negative": int(final["FN"]), "miss_rate": 1 - final["Recall"],
        "accuracy": final["准确率"], "majority_accuracy": 1 - final["违约率"], "cv_table": markdown_table(cv),
        "cv_min": cv_values.min(), "cv_max": cv_values.max(), "bootstrap_repetitions": manifest["bootstrap_repetitions"],
        "interval_table": markdown_table(intervals.drop(columns="有效重采样次数")), "interval_interpretation": interpretation,
        "test_months": test["issueDate"].dt.to_period("M").nunique(), "brier": final["Brier"], "logloss": final["LogLoss"],
        "ece": (calibration["绝对误差"] * calibration["样本数"]).sum() / len(test), "mean_probability": test[selected].mean(),
        "figure_prefix": figure_prefix, "drift_table": markdown_table(drift),
        "importance_table": markdown_table(importance[["特征", "平均绝对SHAP", "SHAP占比"]].head(12)),
        "feature_group_table": markdown_table(groups), "rating_share": groups.set_index("分组")["SHAP占比"].get("既有评级及其交叉统计", 0),
        "direction_table": markdown_table(directions), "top_bad_rate": top["违约率"], "bottom_bad_rate": bottom["违约率"],
        "top_lift": top["Lift"], "top_coverage": top["累计覆盖率"], "top_capture": top["累计违约捕获率"],
        "ks_threshold": candidate["阈值"], "ks_recall": candidate["Recall"], "ks_precision": candidate["Precision"],
        "ks_flagged": candidate["标记比例"], "ks_pass_bad_rate": candidate["通过组违约率"],
        "applied_threshold": bundle["threshold"], "applied_recall": applied["Recall"],
        "applied_rejection_rate": applied["标记比例"], "applied_pass_bad_rate": applied["通过组违约率"],
        "max_rejection_rate": bundle["threshold_policy"]["拒绝率上限"],
        "entropy_count": sum(name.endswith("_y_entropy") for name in bundle["selected_features"]),
        "source_relative": source.relative_to(project_dir).as_posix(), "output_relative": output_dir.relative_to(project_dir).as_posix(),
    }
    values["test_metrics_table"] = markdown_table(metrics.loc[metrics["集合"] == "测试集", ["模型", "AUC", "KS", "PR-AUC", "Precision", "Recall", "F1", "标记比例", "Brier"]])
    values["quarterly_table"] = markdown_table(quarterly[["季度", "样本数", "违约率", "AUC", "KS", "PR-AUC"]])
    values["risk_table"] = markdown_table(risk[["风险档", "样本数", "违约率", "平均预测概率", "累计违约捕获率", "Lift"]])
    values["threshold_table"] = markdown_table(policies_table[["策略", "集合", "阈值", "Precision", "Recall", "标记比例", "通过组违约率", "F1"]])
    report = Path(__file__).with_name("final_report_template.md").read_text(encoding="utf-8").format(**values)
    path = report_dir / "最终项目报告.md"
    path.write_text(report, encoding="utf-8")
    render_html(report, path.with_suffix(".html"), "信贷违约预测项目最终报告")
    boss_report = Path(__file__).with_name("final_report_template_boss.md").read_text(encoding="utf-8").format(**values)
    boss_path = report_dir / "项目汇报_管理层版.md"
    boss_path.write_text(boss_report, encoding="utf-8")
    render_html(boss_report, boss_path.with_suffix(".html"), "信贷风险模型项目阶段汇报")
    return path
