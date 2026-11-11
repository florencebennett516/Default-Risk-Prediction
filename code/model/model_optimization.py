from functools import partial
from pathlib import Path
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from model_optimization_features import (
    AnonymousFeatureAggregator, CrossEntropy2DTransformer, CrossFeatureTransformer, IVFeatureSelector,
    MissingFlagImputer, build_features, select_features,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "data_processing.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "model_optimization"
REPORT_PATH = PROJECT_DIR / "docs" / "阶段报告" / "第5阶段_模型优化.md"
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
MODEL_NAMES = ["逻辑回归", "随机森林", "XGBoost"]
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
    "逻辑回归": {"C": 1.0, "class_weight": None},
    "随机森林": {
        "n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20,
        "max_features": "sqrt", "class_weight": None,
    },
    "XGBoost": {
        "n_estimators": 1000, "max_depth": 5, "learning_rate": 0.05,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
    },
}
SEARCH_SPACE = {
    "逻辑回归": {"C": (0.001, 100.0), "class_weight": [None, "balanced"]},
    "随机森林": {
        "n_estimators": (100, 500), "max_depth": (5, 20),
        "min_samples_leaf": (10, 100), "max_features": ["sqrt", 0.5, 1.0],
        "class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "n_estimators": (100, 1000), "max_depth": (3, 8),
        "learning_rate": (0.01, 0.15), "min_child_weight": (1, 20),
        "subsample": (0.6, 1.0), "colsample_bytree": (0.6, 1.0),
        "reg_alpha": (1e-8, 10.0), "reg_lambda": (0.001, 10.0),
    },
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


def make_model(model_name, parameters):
    if model_name == "逻辑回归":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, **parameters),
        )
    if model_name == "随机森林":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, **parameters),
        )
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
        model = make_model(model_name, parameters)
        model.fit(train_ready, train_target)
        probability = model.predict_proba(valid_ready)[:, 1]
        rows.append({"折": fold_number, **evaluate(model_name, valid_target, probability)})
        probabilities.append(pd.Series(probability, index=valid_target.index))
    return pd.DataFrame(rows), pd.concat(probabilities).sort_index()


def suggest_parameters(trial, model_name):
    space = SEARCH_SPACE[model_name]
    if model_name == "逻辑回归":
        return {
            "C": trial.suggest_float("C", *space["C"], log=True),
            "class_weight": trial.suggest_categorical("class_weight", space["class_weight"]),
        }
    if model_name == "随机森林":
        return {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *space["min_samples_leaf"]),
            "max_features": trial.suggest_categorical("max_features", space["max_features"]),
            "class_weight": trial.suggest_categorical("class_weight", space["class_weight"]),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
    }


def objective(trial, model_name, folds):
    parameters = suggest_parameters(trial, model_name)
    metrics, _ = cross_validate(model_name, parameters, folds)
    trial.set_user_attr("fold_auc", metrics["AUC"].tolist())
    trial.set_user_attr("auc_std", float(metrics["AUC"].std()))
    return float(metrics["AUC"].mean())


def fit_models(features, target, best_parameters, oof_probabilities):
    models = {}
    for model_name in MODEL_NAMES:
        baseline = make_model(model_name, BASELINE_PARAMETERS[model_name])
        baseline.fit(features, target)
        models[f"基线_{model_name}"] = baseline
        optimized = make_model(model_name, best_parameters[model_name])
        optimized.fit(features, target)
        models[f"调优_{model_name}"] = optimized
    meta_features = pd.DataFrame(oof_probabilities).dropna()
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta_model.fit(meta_features[MODEL_NAMES], target.loc[meta_features.index])
    models["堆叠"] = meta_model
    return models


def predict_models(models, features):
    probabilities = pd.DataFrame(index=features.index)
    for model_name in MODEL_NAMES:
        probabilities[f"基线_{model_name}"] = models[f"基线_{model_name}"].predict_proba(features)[:, 1]
        probabilities[f"调优_{model_name}"] = models[f"调优_{model_name}"].predict_proba(features)[:, 1]
    meta_features = probabilities[[f"调优_{name}" for name in MODEL_NAMES]].copy()
    meta_features.columns = MODEL_NAMES
    probabilities["软投票"] = meta_features.mean(axis=1)
    probabilities["堆叠"] = models["堆叠"].predict_proba(meta_features)[:, 1]
    return probabilities


def predict_final(bundle, data):
    features = build_features(data)
    ready = bundle["preprocessor"].transform(features)[bundle["selected_features"]].astype("float32")
    probabilities = predict_models(bundle["models"], ready)
    return probabilities[bundle["selected_model"]]


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

    best_parameters, optimized_oof = {}, {}
    cv_frames, best_rows = [], []
    for model_name in MODEL_NAMES:
        baseline_metrics, _ = cross_validate(model_name, BASELINE_PARAMETERS[model_name], folds)
        baseline_metrics["模型"] = f"基线_{model_name}"
        cv_frames.append(baseline_metrics)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        )
        study.optimize(partial(objective, model_name=model_name, folds=folds), n_trials=N_TRIALS)
        trial_rows = [{
            "试验": trial.number, **trial.params, "平均AUC": trial.value,
            "AUC标准差": trial.user_attrs["auc_std"],
            **{f"第{number}折AUC": score for number, score in enumerate(trial.user_attrs["fold_auc"], start=1)},
            "耗时秒": trial.duration.total_seconds(),
        } for trial in study.trials]
        save_table(pd.DataFrame(trial_rows), f"trials_{model_name}.csv")
        best_parameters[model_name] = study.best_params
        best_rows.append({
            "模型": model_name, "最佳参数": json.dumps(study.best_params, ensure_ascii=False),
            "CV平均AUC": study.best_value, "CV标准差": study.best_trial.user_attrs["auc_std"],
        })
        optimized_metrics, optimized_oof[model_name] = cross_validate(model_name, study.best_params, folds)
        optimized_metrics["模型"] = f"调优_{model_name}"
        cv_frames.append(optimized_metrics)

    best_table = pd.DataFrame(best_rows)
    cv_metrics = pd.concat(cv_frames, ignore_index=True)
    cv_summary = cv_metrics.groupby("模型", sort=False)["AUC"].agg(["mean", "std"]).reset_index()
    cv_summary.columns = ["模型", "CV平均AUC", "CV标准差"]
    save_table(best_table, "best_parameters.csv")
    save_table(cv_metrics, "cv_fold_metrics.csv")
    save_table(cv_summary, "cv_summary.csv")
    train_data[["id", "issueDate", TARGET]].join(pd.DataFrame(optimized_oof)).to_csv(
        OUTPUT_DIR / "train_oof_predictions.csv", index=False, encoding="utf-8-sig",
    )

    _, _, train_ready, validation_ready = fit_features(train_features, train_target, validation_features)
    models = fit_models(train_ready, train_target, best_parameters, optimized_oof)
    validation_probabilities = predict_models(models, validation_ready)
    validation_metrics = pd.DataFrame([
        evaluate(name, validation_target, validation_probabilities[name]) for name in validation_probabilities
    ]).sort_values(["AUC", "KS"], ascending=False)
    selected_model = validation_metrics.iloc[0]["模型"]
    baseline_name = validation_metrics.loc[validation_metrics["模型"].str.startswith("基线_")].iloc[0]["模型"]
    save_table(validation_metrics, "validation_metrics.csv")
    validation_data[["id", "issueDate", TARGET]].join(validation_probabilities).to_csv(
        OUTPUT_DIR / "validation_predictions.csv", index=False, encoding="utf-8-sig",
    )
    print(f"\n验证集选定模型：{selected_model}", flush=True)
    del folds, train_ready, validation_ready, models

    development_features = pd.concat([train_features, validation_features])
    development_target = pd.concat([train_target, validation_target])
    development_dates = pd.concat([train_data["issueDate"], validation_data["issueDate"]])
    development_folds, development_fold_summary = prepare_folds(
        development_features, development_target, development_dates,
    )
    save_table(development_fold_summary, "retrain_fold_summary.csv")
    development_oof = {}
    for model_name in MODEL_NAMES:
        _, development_oof[model_name] = cross_validate(model_name, best_parameters[model_name], development_folds)
    data.loc[development_features.index, ["id", "issueDate", TARGET]].join(pd.DataFrame(development_oof)).to_csv(
        OUTPUT_DIR / "retrain_oof_predictions.csv", index=False, encoding="utf-8-sig",
    )
    del development_folds
    preprocessor, selected, development_ready, test_ready = fit_features(
        development_features, development_target, build_features(test_data),
    )
    models = fit_models(development_ready, development_target, best_parameters, development_oof)
    test_probabilities = predict_models(models, test_ready)
    test_metrics = pd.DataFrame([
        evaluate(name, test_data[TARGET], test_probabilities[name]) for name in test_probabilities
    ])
    reference_auc = test_metrics.set_index("模型").loc[baseline_name, "AUC"]
    test_metrics["相对参考基线AUC差"] = test_metrics["AUC"] - reference_auc
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

    search_table = pd.DataFrame([
        {"模型": name, "参数": parameter, "范围": str(values)}
        for name, space in SEARCH_SPACE.items() for parameter, values in space.items()
    ])
    settings = pd.DataFrame({
        "项目": ["每模型试验次数", "时间交叉验证折数", "随机种子", "模型线程数", "分类阈值", "最终特征数"],
        "数值": [N_TRIALS, N_SPLITS, SEED, N_JOBS, THRESHOLD, len(selected)],
    })
    metric_columns = ["模型", "AUC", "KS", "PR-AUC", "Precision", "Recall", "F1"]
    final_result = test_metrics.set_index("模型").loc[selected_model]
    confusion = pd.DataFrame({
        "实际类别": ["正常", "违约"],
        "预测正常": [int(final_result["TN"]), int(final_result["FN"])],
        "预测违约": [int(final_result["FP"]), int(final_result["TP"])],
    })
    report = f"""# 第5阶段：模型优化

运行时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}

## 1. 数据划分与运行参数

按 issueDate、id 排序，以完整月份划分样本。

{markdown_table(sample_summary)}

{markdown_table(settings)}

特征流程新增 n0–n14 逐行聚合，在缺失填补前计算和、均值、中位数、最小值、最大值、
总体标准差（ddof=0）、非零数量和缺失数量，共8个特征，保留原匿名字段。
统计量跳过缺失值，全缺失行的前6项保留缺失；非零数量不计缺失值。
新增特征与原特征一起参与后续缺失处理、IV和共线性筛选，不使用标签或跨样本统计。

运行：`D:\\credit\\venv\\Scripts\\python.exe D:\\credit\\code\\model\\model_optimization.py`

## 2. 贝叶斯搜索范围与最佳参数

使用 Optuna TPE，前5次随机启动，按时间交叉验证平均 AUC 最大化选参数。
C、learning_rate、reg_alpha、reg_lambda 使用对数采样，其余按表中范围采样。
三套基线和调优模型使用相同划分及特征流程；XGBoost 基线固定1000棵树，不使用验证集早停。

{markdown_table(search_table)}

{markdown_table(best_table)}

逐次试验记录：`results/model_optimization/trials_模型名.csv`。

## 3. 时间交叉验证结果

每折仅在训练月份拟合缺失处理、分组统计、编码、IV和共线性筛选，再转换验证月份。
目标编码内部使用5折交叉拟合；下表为模型搜索及同口径基线的时间交叉验证成绩。

{markdown_table(fold_summary)}

{markdown_table(cv_summary)}

逐折指标：`results/model_optimization/cv_fold_metrics.csv`。

## 4. 基线、调优、投票和堆叠对比

软投票为三个调优模型的概率等权平均；堆叠以时间折外概率训练逻辑回归元模型。
最早没有折外预测的月份不参与元模型训练。模型按验证集 AUC、KS 依次排序选择。

{markdown_table(validation_metrics[metric_columns])}

验证集选定模型：**{selected_model}**。参考基线：**{baseline_name}**。

## 5. 再训练与测试结果

合并训练集和验证集，重新拟合特征流程与模型；堆叠重新生成开发集时间折外预测。
测试集结果如下，PR-AUC 使用 average precision，分类阈值固定为 {THRESHOLD}。

{markdown_table(test_metrics[metric_columns + ["相对参考基线AUC差"]])}

最终模型 **{selected_model}**：测试 AUC **{final_result['AUC']:.4f}**，
KS **{final_result['KS']:.4f}**，相对参考基线 AUC 差 **{final_result['相对参考基线AUC差']:+.4f}**。

{markdown_table(confusion)}

模型：`results/model_optimization/final_model.joblib`。
预测：`results/model_optimization/test_predictions.csv`。
模型文件保存预处理器、最终特征、各模型和验证集选定的模型名称。
加载后调用 `predict_final(bundle, data)` 得到该模型的违约概率。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n报告已保存：{REPORT_PATH}", flush=True)
    return bundle


if __name__ == "__main__":
    main()
