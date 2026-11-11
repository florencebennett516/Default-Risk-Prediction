from functools import partial
from pathlib import Path
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from xgboost import XGBClassifier

from model_optimization_features import (
    AnonymousFeatureAggregator, CrossEntropy2DTransformer, CrossFeatureTransformer, IVFeatureSelector,
    MissingFlagImputer, build_features, select_features,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "data_processing.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "xgboost_optimization"
REPORT_PATH = PROJECT_DIR / "docs" / "阶段报告" / "第5阶段_XGBoost模型优化.md"
TARGET = "isDefault"
TRAIN_END = "2016-03-31"
VALIDATION_END = "2017-06-30"
N_TRIALS = 20
N_SPLITS = 5
N_JOBS = 4
SEED = 42
THRESHOLD = 0.5
IV_THRESHOLD = 0.02
CORRELATION_THRESHOLD = 0.8
VIF_THRESHOLD = 10
SELECTION_SAMPLE_SIZE = 20000
ANONYMOUS_VARIABLES = [f"n{index}" for index in range(15)]
CONTINUOUS_VARIABLES = [
    "loanAmnt", "interestRate", "installment", "annualIncome", "dti",
    "ficoRange_mean", "revolBal", "revolUtil",
]
CATEGORICAL_VARIABLES = [
    "grade", "subGrade", "employmentTitle", "homeOwnership", "verificationStatus",
    "purpose", "postCode", "regionCode", "initialListStatus", "applicationType", "title",
]
CATEGORY_PAIRS = [
    ("grade", "homeOwnership"), ("grade", "verificationStatus"),
    ("grade", "applicationType"), ("grade", "purpose"),
    ("subGrade", "homeOwnership"), ("subGrade", "verificationStatus"),
    ("subGrade", "applicationType"), ("subGrade", "purpose"),
]
BASELINE_PARAMETERS = {
    "n_estimators": 1000, "max_depth": 5, "learning_rate": 0.05,
    "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.0, "reg_lambda": 1.0,
}
SEARCH_SPACE = {
    "n_estimators": (100, 1000), "max_depth": (3, 8),
    "learning_rate": (0.01, 0.15), "min_child_weight": (1, 20),
    "subsample": (0.6, 1.0), "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (1e-8, 10.0), "reg_lambda": (0.001, 10.0),
}


def fit_features(train_features, train_target, other_features):
    encoder = ColumnTransformer([
        ("ordinal", OrdinalEncoder(
            categories=[list("ABCDEFG"), [f"{grade}{level}" for grade in "ABCDEFG" for level in range(1, 6)]],
            handle_unknown="use_encoded_value", unknown_value=-1,
        ), ["grade", "subGrade"]),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["homeOwnership", "verificationStatus", "purpose", "initialListStatus", "applicationType"]),
        ("target", TargetEncoder(
            target_type="binary", smooth="auto",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        ), ["employmentTitle", "postCode", "regionCode", "title"]),
    ], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    preprocessor = make_pipeline(
        AnonymousFeatureAggregator(ANONYMOUS_VARIABLES),
        MissingFlagImputer(), CrossFeatureTransformer(CONTINUOUS_VARIABLES, CATEGORICAL_VARIABLES),
        CrossEntropy2DTransformer(CATEGORY_PAIRS, CONTINUOUS_VARIABLES),
        encoder, IVFeatureSelector(iv_threshold=IV_THRESHOLD),
    )
    train_ready = preprocessor.fit_transform(train_features, train_target).astype("float32")
    iv_scores = preprocessor.steps[-1][1].iv_result_.set_index("feature")["IV"]
    selected = select_features(
        train_ready, iv_scores, SELECTION_SAMPLE_SIZE, SEED, CORRELATION_THRESHOLD, VIF_THRESHOLD,
    )
    train_ready = train_ready.loc[:, selected]
    other_ready = preprocessor.transform(other_features).loc[:, selected].astype("float32")
    return preprocessor, selected, train_ready, other_ready


def prepare_folds(features, target, dates):
    months = dates.dt.to_period("M")
    unique_months = months.sort_values().unique()
    folds, summary = [], []
    for fold_number, (train_months, valid_months) in enumerate(
        TimeSeriesSplit(n_splits=N_SPLITS).split(unique_months), start=1,
    ):
        train_index = np.flatnonzero(months.isin(unique_months[train_months]))
        valid_index = np.flatnonzero(months.isin(unique_months[valid_months]))
        train_target, valid_target = target.iloc[train_index], target.iloc[valid_index]
        _, selected, train_ready, valid_ready = fit_features(
            features.iloc[train_index], train_target, features.iloc[valid_index],
        )
        folds.append((train_ready, train_target, valid_ready, valid_target))
        summary.append({
            "折": fold_number, "训练开始": str(dates.iloc[train_index].min().date()),
            "训练结束": str(dates.iloc[train_index].max().date()),
            "验证开始": str(dates.iloc[valid_index].min().date()),
            "验证结束": str(dates.iloc[valid_index].max().date()),
            "训练样本数": len(train_index), "验证样本数": len(valid_index), "特征数": len(selected),
        })
    return folds, pd.DataFrame(summary)


def make_model(parameters):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        n_jobs=N_JOBS, random_state=SEED, **parameters,
    )


def evaluate(model_name, target, probability):
    prediction = (np.asarray(probability) >= THRESHOLD).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probability)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target, prediction, labels=[0, 1],
    ).ravel()
    return {
        "模型": model_name, "AUC": roc_auc_score(target, probability),
        "KS": float((true_positive_rate - false_positive_rate).max()),
        "PR-AUC": average_precision_score(target, probability),
        "Precision": precision_score(target, prediction, zero_division=0),
        "Recall": recall_score(target, prediction), "F1": f1_score(target, prediction, zero_division=0),
        "TN": int(true_negative), "FP": int(false_positive),
        "FN": int(false_negative), "TP": int(true_positive), "样本数": len(target),
    }


def cross_validate(model_name, parameters, folds):
    rows, probabilities = [], []
    for fold_number, (train_ready, train_target, valid_ready, valid_target) in enumerate(folds, start=1):
        model = make_model(parameters)
        model.fit(train_ready, train_target)
        probability = model.predict_proba(valid_ready)[:, 1]
        rows.append({"折": fold_number, **evaluate(model_name, valid_target, probability)})
        probabilities.append(pd.Series(probability, index=valid_target.index))
    return pd.DataFrame(rows), pd.concat(probabilities).sort_index()


def suggest_parameters(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", *SEARCH_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *SEARCH_SPACE["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", *SEARCH_SPACE["min_child_weight"]),
        "subsample": trial.suggest_float("subsample", *SEARCH_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *SEARCH_SPACE["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *SEARCH_SPACE["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *SEARCH_SPACE["reg_lambda"], log=True),
    }


def objective(trial, folds):
    parameters = suggest_parameters(trial)
    metrics, _ = cross_validate("调优_XGBoost", parameters, folds)
    trial.set_user_attr("fold_auc", metrics["AUC"].tolist())
    trial.set_user_attr("auc_std", float(metrics["AUC"].std()))
    return float(metrics["AUC"].mean())


def fit_models(features, target, best_parameters):
    models = {}
    for model_name, parameters in [
        ("基线_XGBoost", BASELINE_PARAMETERS), ("调优_XGBoost", best_parameters),
    ]:
        model = make_model(parameters)
        model.fit(features, target)
        models[model_name] = model
    return models


def predict_models(models, features):
    return pd.DataFrame({
        name: model.predict_proba(features)[:, 1] for name, model in models.items()
    }, index=features.index)


def predict_final(bundle, data):
    features = build_features(data)
    ready = bundle["preprocessor"].transform(features)[bundle["selected_features"]].astype("float32")
    probability = bundle["models"][bundle["selected_model"]].predict_proba(ready)[:, 1]
    return pd.Series(probability, index=ready.index, name=bundle["selected_model"])


def save_table(frame, filename):
    frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    print(f"\n{filename}\n{frame.round(4).to_string(index=False)}", flush=True)


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.round(4).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(sample_summary, fold_summary, cv_summary, best_table,
                 validation_metrics, test_metrics, selected_model, selected):
    search_table = pd.DataFrame([
        {"参数": parameter, "范围": str(values)} for parameter, values in SEARCH_SPACE.items()
    ])
    settings = pd.DataFrame({
        "项目": ["调优试验次数", "时间交叉验证折数", "随机种子", "模型线程数", "分类阈值", "最终特征数"],
        "数值": [N_TRIALS, N_SPLITS, SEED, N_JOBS, THRESHOLD, len(selected)],
    })
    metric_columns = ["模型", "AUC", "KS", "PR-AUC", "Precision", "Recall", "F1"]
    final_result = test_metrics.set_index("模型").loc[selected_model]
    confusion = pd.DataFrame({
        "实际类别": ["正常", "违约"],
        "预测正常": [int(final_result["TN"]), int(final_result["FN"])],
        "预测违约": [int(final_result["FP"]), int(final_result["TP"])],
    })
    retained_aggregates = [name for name in selected if name.startswith("anonymous_")]
    report = f"""# 第5阶段：XGBoost模型优化

运行时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}
本报告只包含XGBoost基线与调优模型，不包含逻辑回归、随机森林、投票或堆叠。

## 1. 数据划分与运行参数

按issueDate、id排序，以完整月份划分训练、验证、测试集。

{markdown_table(sample_summary)}

{markdown_table(settings)}

## 2. 特征流程

业务衍生 → 匿名字段逐行聚合 → 缺失标记与众数填补 → 单类别与双类别分组统计及标签熵
→ 有序/One-Hot/目标编码 → IV筛选 → Spearman相关性筛选 → VIF筛选。
对n0–n14计算和、均值、中位数、最小值、最大值、总体标准差、非零数量和缺失数量。
聚合在填补前执行，只使用当前记录，保留原匿名字段；全缺失行的前6项保留缺失。
非零计数不含缺失值；总体标准差采用ddof=0。IV阈值{IV_THRESHOLD}，
Spearman阈值{CORRELATION_THRESHOLD}，VIF阈值{VIF_THRESHOLD}，抽样数{SELECTION_SAMPLE_SIZE}。
最终保留的匿名聚合相关特征：{json.dumps(retained_aggregates, ensure_ascii=False)}。

## 3. 贝叶斯搜索范围与最佳参数

使用Optuna TPE，前5次随机启动，以训练期5折时间交叉验证平均ROC-AUC最大化为目标。
learning_rate、reg_alpha、reg_lambda使用对数采样，其余按表中范围采样。
XGBoost基线参数：{json.dumps(BASELINE_PARAMETERS, ensure_ascii=False)}。
两套模型使用相同划分及特征流程，不使用验证集早停。

{markdown_table(search_table)}

{markdown_table(best_table)}

## 4. 时间交叉验证

每折仅用训练月份重新拟合缺失处理、分组统计、编码、IV和共线性筛选，随后转换验证月份。
目标编码内部5折交叉拟合与模型时间交叉验证分开执行。

{markdown_table(fold_summary)}

{markdown_table(cv_summary)}

## 5. 验证集对比与模型选择

按验证集AUC、KS依次排序，选定模型：**{selected_model}**。

{markdown_table(validation_metrics[metric_columns])}

## 6. 再训练与测试结果

合并训练集和验证集，重新拟合特征流程及两套XGBoost；保持验证集选定的模型不变，
不根据测试集结果重新选模型。PR-AUC采用average precision，分类阈值固定为{THRESHOLD}。

{markdown_table(test_metrics[metric_columns + ["相对基线AUC差"]])}

最终模型测试AUC：**{final_result['AUC']:.4f}**，KS：**{final_result['KS']:.4f}**，
相对同口径XGBoost基线AUC差：**{final_result['相对基线AUC差']:+.4f}**。

{markdown_table(confusion)}

## 7. 输出与预测

输出目录：results/xgboost_optimization；不会覆盖原三模型优化结果。
逐次搜索、最佳参数、逐折指标、折外预测、验证/测试预测分别写入CSV。
模型：results/xgboost_optimization/final_model.joblib；特征：final_features.csv。
模型文件包含预处理器、筛选后的特征、基线与调优模型以及验证集选定模型名称。
在本脚本和model_optimization_features模块可导入的环境下，使用joblib.load加载模型，
再调用predict_final(bundle, data)获得违约概率；data为清洗后字段结构，标签isDefault可省略。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n报告已保存：{REPORT_PATH}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["issueDate"])
    data = data.sort_values(["issueDate", "id"]).reset_index(drop=True)
    train_data = data.loc[data["issueDate"] <= TRAIN_END]
    validation_data = data.loc[(data["issueDate"] > TRAIN_END) & (data["issueDate"] <= VALIDATION_END)]
    test_data = data.loc[data["issueDate"] > VALIDATION_END]
    split_assignment = data[["id", "issueDate", TARGET]].assign(集合=np.select(
        [data["issueDate"] <= TRAIN_END, data["issueDate"] <= VALIDATION_END],
        ["训练集", "验证集"], default="测试集",
    ))
    split_assignment.to_csv(OUTPUT_DIR / "split_assignment.csv", index=False, encoding="utf-8-sig")
    sample_summary = pd.DataFrame([{
        "样本": name, "样本数": len(frame), "违约率": frame[TARGET].mean(),
        "开始日期": str(frame["issueDate"].min().date()), "结束日期": str(frame["issueDate"].max().date()),
    } for name, frame in [("训练集", train_data), ("验证集", validation_data), ("测试集", test_data)]])
    save_table(sample_summary, "sample_summary.csv")

    train_features, train_target = build_features(train_data), train_data[TARGET]
    validation_features, validation_target = build_features(validation_data), validation_data[TARGET]
    folds, fold_summary = prepare_folds(train_features, train_target, train_data["issueDate"])
    save_table(fold_summary, "fold_summary.csv")
    baseline_metrics, baseline_oof = cross_validate("基线_XGBoost", BASELINE_PARAMETERS, folds)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
    )
    study.optimize(partial(objective, folds=folds), n_trials=N_TRIALS)
    trial_rows = [{
        "试验": trial.number, **trial.params, "平均AUC": trial.value,
        "AUC标准差": trial.user_attrs["auc_std"],
        **{f"第{number}折AUC": score for number, score in enumerate(trial.user_attrs["fold_auc"], start=1)},
        "耗时秒": trial.duration.total_seconds(),
    } for trial in study.trials]
    save_table(pd.DataFrame(trial_rows), "trials_XGBoost.csv")
    best_parameters = study.best_params
    best_table = pd.DataFrame([{
        "模型": "XGBoost", "最佳参数": json.dumps(best_parameters, ensure_ascii=False),
        "CV平均AUC": study.best_value, "CV标准差": study.best_trial.user_attrs["auc_std"],
    }])
    optimized_metrics, optimized_oof = cross_validate("调优_XGBoost", best_parameters, folds)
    cv_metrics = pd.concat([baseline_metrics, optimized_metrics], ignore_index=True)
    cv_summary = cv_metrics.groupby("模型", sort=False)["AUC"].agg(["mean", "std"]).reset_index()
    cv_summary.columns = ["模型", "CV平均AUC", "CV标准差"]
    save_table(best_table, "best_parameters.csv")
    save_table(cv_metrics, "cv_fold_metrics.csv")
    save_table(cv_summary, "cv_summary.csv")
    train_data[["id", "issueDate", TARGET]].join(pd.DataFrame({
        "基线_XGBoost": baseline_oof, "调优_XGBoost": optimized_oof,
    })).to_csv(OUTPUT_DIR / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    del folds

    _, _, train_ready, validation_ready = fit_features(train_features, train_target, validation_features)
    models = fit_models(train_ready, train_target, best_parameters)
    validation_probabilities = predict_models(models, validation_ready)
    validation_metrics = pd.DataFrame([
        evaluate(name, validation_target, validation_probabilities[name]) for name in validation_probabilities
    ]).sort_values(["AUC", "KS"], ascending=False, kind="stable")
    selected_model = validation_metrics.iloc[0]["模型"]
    save_table(validation_metrics, "validation_metrics.csv")
    validation_data[["id", "issueDate", TARGET]].join(validation_probabilities).to_csv(
        OUTPUT_DIR / "validation_predictions.csv", index=False, encoding="utf-8-sig",
    )
    print(f"\n验证集选定模型：{selected_model}", flush=True)
    del train_ready, validation_ready, models

    development_features = pd.concat([train_features, validation_features])
    development_target = pd.concat([train_target, validation_target])
    preprocessor, selected, development_ready, test_ready = fit_features(
        development_features, development_target, build_features(test_data),
    )
    models = fit_models(development_ready, development_target, best_parameters)
    test_probabilities = predict_models(models, test_ready)
    test_metrics = pd.DataFrame([
        evaluate(name, test_data[TARGET], test_probabilities[name]) for name in test_probabilities
    ])
    baseline_auc = test_metrics.set_index("模型").loc["基线_XGBoost", "AUC"]
    test_metrics["相对基线AUC差"] = test_metrics["AUC"] - baseline_auc
    save_table(test_metrics, "test_metrics.csv")
    test_data[["id", "issueDate", TARGET]].join(test_probabilities).assign(
        最终预测=(test_probabilities[selected_model] >= THRESHOLD).astype(int),
    ).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    bundle = {
        "preprocessor": preprocessor, "selected_features": selected, "models": models,
        "selected_model": selected_model, "threshold": THRESHOLD, "parameters": best_parameters,
    }
    joblib.dump(bundle, OUTPUT_DIR / "final_model.joblib")
    save_table(pd.DataFrame({"特征": selected}), "final_features.csv")
    write_report(
        sample_summary, fold_summary, cv_summary, best_table,
        validation_metrics, test_metrics, selected_model, selected,
    )
    return bundle


if __name__ == "__main__":
    main()
