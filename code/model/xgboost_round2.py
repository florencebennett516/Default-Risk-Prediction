import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

import xgboost_optimization as legacy
from model_optimization_features import (
    CrossEntropy2DTransformer, CrossFeatureTransformer, IVFeatureSelector,
    MissingFlagImputer, select_features,
)
from xgboost_round2_features import (
    ANONYMOUS_COLUMNS, FEATURE_VERSION, LEGACY_NAMES, NEW_FEATURES,
    AnonymousFeatureAggregatorV2, build_features_v2, check_feature_boundaries,
    feature_checklist,
)
from xgboost_round2_report_features import build_final_feature_dictionary


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_DIR / "results" / "xgboost_optimization"
REPORT_PATH = PROJECT_DIR / "docs" / "周报" / "XGBOOST第二轮优化及两轮对比报告.md"
FINAL_REPORT_PATH = PROJECT_DIR / "docs" / "周报" / "XGBOOST第二轮优化及两轮对比报告_最终版.md"
OLD_NAMES = {"基线_XGBoost": "第一轮_基线", "调优_XGBoost": "第一轮_调优"}
NEW_NAMES = ["第二轮_基线参数", "第二轮_上轮最优参数", "第二轮_重新调优"]
METRICS = ["AUC", "KS", "PR-AUC", "Precision", "Recall", "F1"]


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def write_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_files():
    paths = [legacy.DATA_PATH, Path(legacy.__file__), Path(legacy.__file__).with_name("model_optimization_features.py")]
    paths.extend(sorted(SOURCE_DIR.glob("*")))
    paths.extend([
        PROJECT_DIR / "docs" / "阶段报告" / "第5阶段_XGBoost模型优化.md",
        PROJECT_DIR / "docs" / "周报" / "optimization.md",
        PROJECT_DIR / "docs" / "周报" / "XGBOOST优化.pdf",
    ])
    return [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in paths if path.is_file()]


def verify_protected(records):
    changed = [record["path"] for record in records if file_hash(Path(record["path"])) != record["sha256"]]
    if changed:
        raise ValueError(f"输入或历史成果发生变化：{changed}")


def load_inputs():
    data = pd.read_csv(legacy.DATA_PATH, parse_dates=["issueDate", "earliesCreditLine"])
    data = data.sort_values(["issueDate", "id"]).reset_index(drop=True)
    if data["id"].duplicated().any() or not data[legacy.TARGET].isin([0, 1]).all():
        raise ValueError("ID不唯一或标签不是二分类")
    assignment = data[["id", "issueDate", legacy.TARGET]].assign(集合=np.select(
        [data["issueDate"] <= legacy.TRAIN_END, data["issueDate"] <= legacy.VALIDATION_END],
        ["训练集", "验证集"], default="测试集",
    ))
    old_assignment = pd.read_csv(SOURCE_DIR / "split_assignment.csv", parse_dates=["issueDate"])
    pd.testing.assert_frame_equal(assignment, old_assignment, check_dtype=False)
    summary = pd.DataFrame([{
        "样本": name, "样本数": len(frame), "违约率": frame[legacy.TARGET].mean(),
        "开始日期": str(frame["issueDate"].min().date()), "结束日期": str(frame["issueDate"].max().date()),
    } for name in ["训练集", "验证集", "测试集"] for frame in [data.loc[assignment["集合"] == name]]])
    if summary["样本数"].tolist() != [558371, 161731, 79898]:
        raise ValueError("样本量与锁定的上一轮划分不一致")
    audit = check_feature_boundaries(data.head(1))
    audit["样本ID日期标签划分一致"] = True
    audit["匿名字段负数个数"] = int(data[ANONYMOUS_COLUMNS].lt(0).sum().sum())
    audit["非正收入样本数"] = int(data["annualIncome"].le(0).sum())
    if audit["匿名字段负数个数"]:
        raise ValueError("匿名字段出现负数，无法沿用本次已确认的计数等价结论")
    historical_manifest = PROJECT_DIR / "results" / "final_project" / "run_manifest.json"
    if historical_manifest.exists():
        records = json.loads(historical_manifest.read_text(encoding="utf-8"))["source_files"]
        recorded = next(record for record in records if Path(record["path"]) == legacy.DATA_PATH)
        if file_hash(legacy.DATA_PATH) != recorded["sha256"]:
            raise ValueError("清洗数据与上一轮历史文件哈希不一致")
        audit["清洗数据历史哈希一致"] = True
    return data, assignment, summary, audit


def fit_features(train_features, train_target, other_features, output_dir, phase):
    log(f"拟合特征流程：{phase}，{len(train_features)}条训练样本")
    encoder = ColumnTransformer([
        ("ordinal", OrdinalEncoder(
            categories=[list("ABCDEFG"), [f"{grade}{level}" for grade in "ABCDEFG" for level in range(1, 6)]],
            handle_unknown="use_encoded_value", unknown_value=-1,
        ), ["grade", "subGrade"]),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["homeOwnership", "verificationStatus", "purpose", "initialListStatus", "applicationType"]),
        ("target", TargetEncoder(
            target_type="binary", smooth="auto",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=legacy.SEED),
        ), ["employmentTitle", "postCode", "regionCode", "title"]),
    ], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    preprocessor = make_pipeline(
        AnonymousFeatureAggregatorV2(ANONYMOUS_COLUMNS), MissingFlagImputer(),
        CrossFeatureTransformer(legacy.CONTINUOUS_VARIABLES, legacy.CATEGORICAL_VARIABLES),
        CrossEntropy2DTransformer(legacy.CATEGORY_PAIRS, legacy.CONTINUOUS_VARIABLES),
        encoder, IVFeatureSelector(iv_threshold=legacy.IV_THRESHOLD),
    )
    train_ready = preprocessor.fit_transform(train_features, train_target).astype("float32")
    iv = preprocessor.steps[-1][1].iv_result_.set_index("feature")["IV"]
    selected = select_features(
        train_ready, iv, legacy.SELECTION_SAMPLE_SIZE, legacy.SEED,
        legacy.CORRELATION_THRESHOLD, legacy.VIF_THRESHOLD,
    )
    audit = iv.rename("IV").reset_index()
    audit["筛选状态"] = np.select(
        [audit["feature"].isin(selected), audit["IV"] < legacy.IV_THRESHOLD],
        ["保留", "IV筛除"], default="相关性或VIF筛除",
    )
    save_csv(audit, output_dir / f"feature_selection_{phase}.csv")
    train_ready = train_ready.loc[:, selected]
    other_ready = preprocessor.transform(other_features).loc[:, selected].astype("float32")
    for ready in [train_ready, other_ready]:
        if ready.columns.has_duplicates or np.isinf(ready.to_numpy()).any():
            raise ValueError("筛选后特征存在重复列或无穷值")
    log(f"特征流程完成：{phase}，保留{len(selected)}列")
    return preprocessor, selected, train_ready, other_ready


def prepare_folds(features, target, dates, output_dir):
    cache = output_dir / "fold_cache.joblib"
    if cache.exists():
        log("加载已验证的时间折缓存")
        return joblib.load(cache)
    months = dates.dt.to_period("M")
    unique_months = months.sort_values().unique()
    folds, summaries = [], []
    for number, (train_months, valid_months) in enumerate(TimeSeriesSplit(n_splits=5).split(unique_months), 1):
        train_index = np.flatnonzero(months.isin(unique_months[train_months]))
        valid_index = np.flatnonzero(months.isin(unique_months[valid_months]))
        if dates.iloc[train_index].max() >= dates.iloc[valid_index].min():
            raise ValueError("时间交叉验证发生月份重叠")
        fold_cache = output_dir / f"fold_{number}_cache.joblib"
        if fold_cache.exists():
            fold = joblib.load(fold_cache)
            selected = fold[0].columns.tolist()
        else:
            _, selected, train_ready, valid_ready = fit_features(
                features.iloc[train_index], target.iloc[train_index], features.iloc[valid_index], output_dir, f"fold{number}",
            )
            fold = (train_ready, target.iloc[train_index], valid_ready, target.iloc[valid_index])
            joblib.dump(fold, fold_cache)
        folds.append(fold)
        summaries.append({
            "折": number, "训练开始": str(dates.iloc[train_index].min().date()),
            "训练结束": str(dates.iloc[train_index].max().date()), "验证开始": str(dates.iloc[valid_index].min().date()),
            "验证结束": str(dates.iloc[valid_index].max().date()), "训练样本数": len(train_index),
            "验证样本数": len(valid_index), "特征数": len(selected),
        })
        gc.collect()
    summary = pd.DataFrame(summaries)
    old_summary = pd.read_csv(SOURCE_DIR / "fold_summary.csv")
    columns = [column for column in summary.columns if column != "特征数"]
    pd.testing.assert_frame_equal(summary[columns], old_summary[columns], check_dtype=False)
    save_csv(summary, output_dir / "fold_summary.csv")
    joblib.dump(folds, cache)
    return folds


def save_trials(study, output_dir):
    rows, fold_rows = [], []
    for trial in study.trials:
        rows.append({
            "试验": trial.number, "状态": trial.state.name, **trial.params,
            "平均AUC": trial.value, "AUC标准差": trial.user_attrs.get("auc_std"),
            **{f"第{number}折AUC": score for number, score in enumerate(trial.user_attrs.get("fold_auc", []), 1)},
            "耗时秒": trial.duration.total_seconds() if trial.duration else None,
            "失败原因": trial.user_attrs.get("error"),
        })
        fold_rows.extend({"试验": trial.number, **row} for row in trial.user_attrs.get("fold_metrics", []))
    current = pd.DataFrame(rows)
    save_csv(current, output_dir / "trials_XGBoost.csv")
    if fold_rows:
        save_csv(pd.DataFrame(fold_rows), output_dir / "trial_fold_metrics.csv")
    historical = pd.read_csv(SOURCE_DIR / "trials_XGBoost.csv").assign(状态="COMPLETE")
    tables = []
    for round_number, feature_version, table, source in [
        (1, "business_v1", historical, SOURCE_DIR / "trials_XGBoost.csv"),
        (2, FEATURE_VERSION, current, output_dir / "trials_XGBoost.csv"),
    ]:
        tables.append(table.assign(轮次=round_number, 特征版本=feature_version, 原始试验编号=table["试验"], 记录来源=str(source)))
    save_csv(pd.concat(tables, ignore_index=True), output_dir / "trials_merged.csv")


def search(folds, output_dir, n_trials):
    sampler_path = output_dir / "sampler.joblib"
    sampler = joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=legacy.SEED, n_startup_trials=5)
    study = optuna.create_study(
        study_name="xgboost_round2", direction="maximize", sampler=sampler,
        storage="sqlite:///" + (output_dir / "study.sqlite3").as_posix(), load_if_exists=True,
    )
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

    def objective(trial):
        log(f"开始搜索试验{trial.number}")
        rows = []
        try:
            parameters = legacy.suggest_parameters(trial)
            for number, (train_ready, train_target, valid_ready, valid_target) in enumerate(folds, 1):
                model = legacy.make_model(parameters)
                model.fit(train_ready, train_target)
                probability = model.predict_proba(valid_ready)[:, 1]
                rows.append({"折": number, **legacy.evaluate(NEW_NAMES[2], valid_target, probability)})
            scores = [row["AUC"] for row in rows]
            trial.set_user_attr("fold_metrics", rows)
            trial.set_user_attr("fold_auc", scores)
            trial.set_user_attr("auc_std", float(np.std(scores, ddof=1)))
            return float(np.mean(scores))
        except Exception as error:
            trial.set_user_attr("fold_metrics", rows)
            trial.set_user_attr("error", repr(error))
            raise

    def callback(current_study, trial):
        save_trials(current_study, output_dir)
        joblib.dump(current_study.sampler, sampler_path)
        log(f"试验{trial.number}：{trial.state.name}，AUC={trial.value}，当前最佳={current_study.best_value:.6f}")

    completed = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    try:
        study.optimize(objective, n_trials=max(0, n_trials - completed), callbacks=[callback])
    finally:
        save_trials(study, output_dir)
        joblib.dump(study.sampler, sampler_path)
    completed = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    if completed != n_trials:
        raise RuntimeError(f"仅完成{completed}/{n_trials}次搜索，不能生成完成报告")
    write_json({
        "参数": study.best_params, "CV平均AUC": study.best_value,
        "CV标准差": study.best_trial.user_attrs["auc_std"], "试验编号": study.best_trial.number,
    }, output_dir / "search_best.json")
    return study.best_params


def historical_metrics(filename):
    table = pd.read_csv(SOURCE_DIR / filename)
    table["模型"] = table["模型"].map(OLD_NAMES)
    return table


def historical_probabilities(frame, filename):
    stored = pd.read_csv(SOURCE_DIR / filename, parse_dates=["issueDate"])
    pd.testing.assert_frame_equal(
        frame[["id", "issueDate", legacy.TARGET]].reset_index(drop=True),
        stored[["id", "issueDate", legacy.TARGET]], check_dtype=False,
    )
    return stored.rename(columns=OLD_NAMES).set_axis(frame.index)[list(OLD_NAMES.values())]


def predict_round2(bundle, data, model_name=None):
    if bundle.get("format_version") != "round2_v1":
        raise ValueError("请使用第二轮版本化模型包")
    name = model_name or bundle["selected_model"]
    if name in OLD_NAMES.values():
        old_bundle = dict(bundle["historical_bundle"])
        old_bundle["selected_model"] = next(key for key, value in OLD_NAMES.items() if value == name)
        return legacy.predict_final(old_bundle, data).rename(name)
    if bundle["feature_versions"].get(name) != FEATURE_VERSION:
        raise ValueError(f"不支持的模型或特征版本：{name}")
    features = build_features_v2(data)
    ready = bundle["preprocessor"].transform(features)[bundle["selected_features"]].astype("float32")
    return pd.Series(bundle["models"][name].predict_proba(ready)[:, 1], index=data.index, name=name)


def fit_candidates(train_ready, target, parameters):
    models = {}
    for name, values in parameters.items():
        log(f"训练模型：{name}，样本{len(train_ready)}，特征{train_ready.shape[1]}")
        model = legacy.make_model(values)
        model.fit(train_ready, target)
        models[name] = model
    return models


def evaluate_candidates(models, ready, data, filename):
    probabilities = historical_probabilities(data, filename)
    for name, model in models.items():
        probabilities[name] = model.predict_proba(ready)[:, 1]
    metrics = pd.concat([
        historical_metrics(filename.replace("predictions", "metrics")),
        pd.DataFrame([legacy.evaluate(name, data[legacy.TARGET], probabilities[name]) for name in models]),
    ], ignore_index=True)
    return probabilities, metrics


def add_differences(frame, group_columns):
    result = frame.copy()
    groups = result.groupby(group_columns, sort=False) if group_columns else [(None, result)]
    for _, group in groups:
        indexed = group.set_index("模型")
        for metric in METRICS:
            if metric not in group:
                continue
            for name, suffix in [(OLD_NAMES["基线_XGBoost"], "上轮基线"), (OLD_NAMES["调优_XGBoost"], "上轮调优")]:
                result.loc[group.index, f"{metric}差_相对{suffix}"] = group[metric] - indexed.loc[name, metric]
    return result


def update_checklist(output_dir, selected_model, selected_features, legacy_features):
    checklist = pd.read_csv(output_dir / "feature_checklist.csv")
    selection = pd.read_csv(output_dir / "feature_selection_development.csv").set_index("feature")
    checklist["本轮开发集IV"] = checklist["实际特征"].map(selection["IV"])
    checklist["本轮开发集筛选状态"] = checklist["实际特征"].map(selection["筛选状态"]).fillna("暂缓")
    checklist["本轮流程保留"] = checklist["实际特征"].isin(selected_features)
    final_features = legacy_features if selected_model in OLD_NAMES.values() else selected_features
    actual = checklist["实际特征"].replace(LEGACY_NAMES) if selected_model in OLD_NAMES.values() else checklist["实际特征"]
    checklist["最终选定模型入模"] = actual.isin(final_features)
    save_csv(checklist, output_dir / "feature_checklist.csv")
    save_csv(pd.DataFrame({"特征": selected_features}), output_dir / "round2_features.csv")
    save_csv(pd.DataFrame({"特征": final_features}), output_dir / "final_features.csv")


def merge_metrics(output_dir):
    fold_metrics = pd.read_csv(output_dir / "cv_fold_metrics.csv")
    cv = fold_metrics.groupby("模型", sort=False)[METRICS].mean().reset_index()
    cv["CV标准差"] = fold_metrics.groupby("模型", sort=False)["AUC"].std().to_numpy()
    cv["集合"] = "训练期时间CV"
    validation = pd.read_csv(output_dir / "validation_metrics.csv").assign(集合="验证集")
    test = pd.read_csv(output_dir / "test_metrics.csv").assign(集合="测试集")
    combined = pd.concat([cv, validation, test], ignore_index=True)
    combined["轮次"] = np.where(combined["模型"].isin(OLD_NAMES.values()), 1, 2)
    combined["特征版本"] = np.where(combined["轮次"] == 1, "business_v1", FEATURE_VERSION)
    save_csv(add_differences(combined, ["集合"]), output_dir / "metrics_merged.csv")


def run_experiment(data, assignment, output_dir, manifest, n_trials):
    def stage(name):
        manifest.update(stage=name, updated_at=datetime.now().isoformat())
        write_json(manifest, output_dir / "run_manifest.json")
        log(name)

    train = data.loc[assignment["集合"] == "训练集"]
    validation = data.loc[assignment["集合"] == "验证集"]
    test = data.loc[assignment["集合"] == "测试集"]
    old_best = json.loads(pd.read_csv(SOURCE_DIR / "best_parameters.csv").iloc[0]["最佳参数"])
    if not (output_dir / "feature_checklist.csv").exists():
        stage("训练集候选特征审计")
        save_csv(feature_checklist(train), output_dir / "feature_checklist.csv")
    train_features = build_features_v2(train)
    validation_features = build_features_v2(validation)
    stage("构建五折时间交叉验证特征")
    folds = prepare_folds(train_features, train[legacy.TARGET], train["issueDate"], output_dir)
    parameters = {NEW_NAMES[0]: legacy.BASELINE_PARAMETERS, NEW_NAMES[1]: old_best}
    save_csv(pd.DataFrame([{"模型": name, "参数": json.dumps(values, ensure_ascii=False)} for name, values in parameters.items()]), output_dir / "control_parameters.csv")
    stage("固定参数特征对照")
    for number, (name, values) in enumerate(parameters.items()):
        cache = output_dir / f"cv_candidate_{number}.joblib"
        if not cache.exists():
            log(f"五折对照：{name}")
            started = time.perf_counter()
            metrics, probabilities = legacy.cross_validate(name, values, folds)
            joblib.dump((metrics, probabilities), cache)
            write_json({"模型": name, "耗时秒": time.perf_counter() - started}, output_dir / f"control_timing_{number}.json")
    stage("50次独立参数搜索")
    parameters[NEW_NAMES[2]] = search(folds, output_dir, n_trials)
    best_cache = output_dir / "cv_candidate_2.joblib"
    if not best_cache.exists():
        joblib.dump(legacy.cross_validate(NEW_NAMES[2], parameters[NEW_NAMES[2]], folds), best_cache)
    cv_tables = [historical_metrics("cv_fold_metrics.csv")]
    oof = historical_probabilities(train, "train_oof_predictions.csv")
    for number, name in enumerate(NEW_NAMES):
        metrics, probabilities = joblib.load(output_dir / f"cv_candidate_{number}.joblib")
        cv_tables.append(metrics)
        oof[name] = probabilities
    save_csv(train[["id", "issueDate", legacy.TARGET]].join(oof), output_dir / "train_oof_predictions.csv")
    cv_metrics = pd.concat(cv_tables, ignore_index=True)
    save_csv(cv_metrics, output_dir / "cv_fold_metrics.csv")
    cv_summary = cv_metrics.groupby("模型", sort=False)["AUC"].agg(["mean", "std"]).reset_index()
    cv_summary.columns = ["模型", "CV平均AUC", "CV标准差"]
    save_csv(cv_summary, output_dir / "cv_summary.csv")
    all_parameters = {OLD_NAMES["基线_XGBoost"]: legacy.BASELINE_PARAMETERS, OLD_NAMES["调优_XGBoost"]: old_best, **parameters}
    parameter_table = pd.DataFrame([{"模型": name, "参数": json.dumps(values, ensure_ascii=False)} for name, values in all_parameters.items()])
    save_csv(parameter_table.merge(cv_summary, on="模型", validate="one_to_one"), output_dir / "candidate_parameters.csv")
    del folds, oof
    gc.collect()

    stage("验证集评估与模型锁定")
    selection_path = output_dir / "selection.json"
    if selection_path.exists():
        selected_model = json.loads(selection_path.read_text(encoding="utf-8"))["selected_model"]
    else:
        transform_cache = output_dir / "training_transform_cache.joblib"
        if transform_cache.exists():
            _, _, train_ready, validation_ready = joblib.load(transform_cache)
        else:
            transformed = fit_features(train_features, train[legacy.TARGET], validation_features, output_dir, "training")
            joblib.dump(transformed, transform_cache)
            _, _, train_ready, validation_ready = transformed
            del transformed
        models = fit_candidates(train_ready, train[legacy.TARGET], parameters)
        probabilities, metrics = evaluate_candidates(models, validation_ready, validation, "validation_predictions.csv")
        metrics["并列优先级"] = metrics["模型"].map({name: priority for priority, name in enumerate([
            OLD_NAMES["调优_XGBoost"], OLD_NAMES["基线_XGBoost"], *NEW_NAMES,
        ])})
        metrics = metrics.sort_values(["AUC", "KS", "并列优先级"], ascending=[False, False, True], kind="stable")
        selected_model = metrics.iloc[0]["模型"]
        save_csv(metrics.drop(columns="并列优先级"), output_dir / "validation_metrics.csv")
        save_csv(validation[["id", "issueDate", legacy.TARGET]].join(probabilities), output_dir / "validation_predictions.csv")
        write_json({
            "selected_model": selected_model, "selected_at": datetime.now().isoformat(),
            "rule": "验证集AUC降序、KS降序、完全并列时上轮选定模型优先", "test_metrics_used": False,
        }, selection_path)
        del models, train_ready, validation_ready, probabilities
    manifest["selected_model"] = selected_model
    log(f"已锁定模型：{selected_model}；此后不按测试结果更换")
    del train_features, validation_features
    gc.collect()

    stage("开发集重训与固定测试集评估")
    development = pd.concat([train, validation])
    transform_cache = output_dir / "development_transform_cache.joblib"
    if transform_cache.exists():
        preprocessor, selected, development_ready, test_ready = joblib.load(transform_cache)
    else:
        transformed = fit_features(build_features_v2(development), development[legacy.TARGET], build_features_v2(test), output_dir, "development")
        joblib.dump(transformed, transform_cache)
        preprocessor, selected, development_ready, test_ready = transformed
        del transformed
    models = fit_candidates(development_ready, development[legacy.TARGET], parameters)
    probabilities, metrics = evaluate_candidates(models, test_ready, test, "test_predictions.csv")
    save_csv(add_differences(metrics, []), output_dir / "test_metrics.csv")
    save_csv(test[["id", "issueDate", legacy.TARGET]].join(probabilities).assign(
        最终预测=probabilities[selected_model].ge(legacy.THRESHOLD).astype("int8"),
    ), output_dir / "test_predictions.csv")
    historical_bundle = joblib.load(SOURCE_DIR / "final_model.joblib")
    bundle = {
        "format_version": "round2_v1", "feature_versions": {
            **{name: "business_v1" for name in OLD_NAMES.values()},
            **{name: FEATURE_VERSION for name in NEW_NAMES},
        },
        "preprocessor": preprocessor, "selected_features": selected, "models": models,
        "historical_bundle": historical_bundle, "selected_model": selected_model,
        "threshold": legacy.THRESHOLD, "parameters": all_parameters,
    }
    joblib.dump(bundle, output_dir / "final_model.joblib")
    update_checklist(output_dir, selected_model, selected, historical_bundle["selected_features"])
    manifest.update(round2_feature_count=len(selected), final_feature_count=len(pd.read_csv(output_dir / "final_features.csv")))
    del bundle, models, preprocessor, development_ready, test_ready, historical_bundle
    gc.collect()

    stage("模型重载与历史文件保护检查")
    loaded = joblib.load(output_dir / "final_model.joblib")
    sample = test.sample(n=min(512, len(test)), random_state=legacy.SEED)
    audit = []
    for name in list(OLD_NAMES.values()) + NEW_NAMES:
        replay = predict_round2(loaded, sample.drop(columns=legacy.TARGET), name)
        expected = probabilities.loc[sample.index, name]
        difference = float(np.abs(replay - expected).max())
        if not np.allclose(replay, expected, atol=2e-7, rtol=0):
            raise ValueError(f"模型重载预测不一致：{name}，误差{difference}")
        audit.append({"模型": name, "样本数": len(sample), "最大概率误差": difference, "标签可省略": True, "通过": True})
    save_csv(pd.DataFrame(audit), output_dir / "prediction_replay_audit.csv")
    verify_protected(manifest["protected_files"])
    manifest["historical_files_unchanged"] = True
    merge_metrics(output_dir)
    manifest.update(status="complete", stage="全部实验与校验完成", completed_at=datetime.now().isoformat())
    write_json(manifest, output_dir / "run_manifest.json")
    write_report(output_dir)
    log(f"全部完成：{REPORT_PATH}")


def markdown_table(frame):
    def display(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "／").replace(chr(10), " ")

    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(display(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return chr(10).join(lines)


def write_report(output_dir):
    output_dir = Path(output_dir).resolve()
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("实验未完成，不生成完成版报告")
    merge_metrics(output_dir)
    trials = pd.read_csv(output_dir / "trials_merged.csv")
    current = trials.loc[trials["轮次"] == 2]
    if current["状态"].eq("COMPLETE").sum() != manifest["n_trials"]:
        raise ValueError("搜索完成数与计划不一致")
    samples = pd.read_csv(output_dir / "sample_summary.csv")
    checklist = pd.read_csv(output_dir / "feature_checklist.csv")
    cv = pd.read_csv(output_dir / "cv_summary.csv")
    validation = pd.read_csv(output_dir / "validation_metrics.csv")
    test = pd.read_csv(output_dir / "test_metrics.csv")
    selected = manifest["selected_model"]
    final = test.set_index("模型").loc[selected]
    feature_names = pd.read_csv(output_dir / "final_features.csv")["特征"].tolist()
    if len(feature_names) != manifest["final_feature_count"]:
        raise ValueError("最终特征清单与模型记录的特征数不一致")
    feature_dictionary = build_final_feature_dictionary(feature_names)
    save_csv(feature_dictionary, output_dir / "final_feature_dictionary.csv")
    feature_summary = feature_dictionary.groupby("特征类型", sort=False).size().rename("特征数").reset_index()
    parameters = pd.read_csv(output_dir / "candidate_parameters.csv")
    parameter_maps = {row["模型"]: json.loads(row["参数"]) for row in parameters.to_dict("records")}
    parameter_comparison = pd.DataFrame([{
        "参数": name, "搜索范围": str(bounds),
        "原基线": parameter_maps[OLD_NAMES["基线_XGBoost"]][name],
        "上轮最优": parameter_maps[OLD_NAMES["调优_XGBoost"]][name],
        "本轮搜索最优": parameter_maps[NEW_NAMES[2]][name],
    } for name, bounds in legacy.SEARCH_SPACE.items()])
    merged = pd.read_csv(output_dir / "metrics_merged.csv")
    effects = []
    for group in ["训练期时间CV", "验证集", "测试集"]:
        rows = merged.loc[merged["集合"] == group].set_index("模型")
        for description, first, second in [
            ("固定原基线参数：特征流程变化", OLD_NAMES["基线_XGBoost"], NEW_NAMES[0]),
            ("固定上轮最优参数：特征流程变化", OLD_NAMES["调优_XGBoost"], NEW_NAMES[1]),
            ("同一新特征流程：重新调参变化", NEW_NAMES[1], NEW_NAMES[2]),
        ]:
            effects.append({"集合": group, "比较": description, "参照模型": first, "比较模型": second,
                           **{f"{metric}差": rows.loc[second, metric] - rows.loc[first, metric] for metric in METRICS}})
    effects = pd.DataFrame(effects)
    save_csv(effects, output_dir / "effect_comparisons.csv")
    old_auc = test.set_index("模型").loc[OLD_NAMES["调优_XGBoost"], "AUC"]
    old_baseline_auc = test.set_index("模型").loc[OLD_NAMES["基线_XGBoost"], "AUC"]
    old_final = test.set_index("模型").loc[OLD_NAMES["调优_XGBoost"]]
    optimized_auc = test.set_index("模型").loc[NEW_NAMES[2], "AUC"]
    improvement = optimized_auc - old_auc
    conclusion = "提高" if improvement > 0 else "下降" if improvement < 0 else "持平"
    selected_new = checklist.loc[(checklist["处理方式"] == "新增") & checklist["本轮流程保留"], "表格变量"].tolist()
    fold_summary = pd.read_csv(output_dir / "fold_summary.csv")
    old_fold = pd.read_csv(SOURCE_DIR / "fold_summary.csv")
    fold_summary["上轮特征数"] = old_fold["特征数"]
    replay = pd.read_csv(output_dir / "prediction_replay_audit.csv")
    report = f"""# XGBoost第二轮优化及两轮对比报告（最终版）

实验完成时间：{manifest['completed_at']}。最终版报告生成时间：{datetime.now().isoformat(timespec='seconds')}。
本报告对应已实际执行的实验，不是调优方案；本次补全最终特征及业务含义，不重新训练或修改已完成的调优结果。

## 1. 结论摘要

- 本轮核对24项需求：复用6项、新增16项候选、暂缓2项类别标记。暂缓项保留原始applicationType与purpose，不猜测类别含义。
- 新特征流程在开发集筛选后共保留{manifest['round2_feature_count']}列，其中直接保留的新增业务特征为：{', '.join(selected_new) if selected_new else '无'}。
- 本轮完成{int(current['状态'].eq('COMPLETE').sum())}次五折搜索；两轮合并共{len(trials)}条搜索记录。固定参数对照另行记录，不占搜索次数。
- 按验证集AUC、KS锁定的最终模型为 **{selected}**，使用对应版本的{manifest['final_feature_count']}个特征。测试AUC **{final['AUC']:.6f}**，KS **{final['KS']:.6f}**，PR-AUC **{final['PR-AUC']:.6f}**。
- 本轮重新调优候选测试AUC为{optimized_auc:.6f}，相对上轮调优模型{old_auc:.6f}，{conclusion}{abs(improvement):.6f}；这是测试集描述性比较，不用于换模型。
- 最终选定模型相对上轮基线测试AUC {old_baseline_auc:.6f} 的差值为{final['AUC'] - old_baseline_auc:+.6f}。新增特征和扩大搜索不预设一定提升。
- 最终选定模型相对上轮调优模型的测试AUC差为{final['AUC'] - old_auc:+.6f}，Recall差为{final['Recall'] - old_final['Recall']:+.6f}，F1差为{final['F1'] - old_final['F1']:+.6f}；排序能力没有提升，固定阈值下召回与F1有所变化。

## 2. 数据、时间划分与可比性

输入为原清洗快照data/data_processing.csv。样本按issueDate、id排序，逐行核对上轮split_assignment.csv的ID、日期、标签和集合；清洗数据同时通过历史哈希校验。
训练截止2016-03-31，验证期2016-04-01至2017-06-30，测试期2017-07-01之后。两轮使用完全相同的样本和时间窗口。

{markdown_table(samples)}

本轮未改动原清洗、原始匿名字段、原有分组统计或编码逻辑。表中“复用”指原候选流程已有，不代表上轮一定通过筛选。原流程中已被筛除的等价特征不重复构造。

## 3. 业务特征对照与计算口径

{markdown_table(checklist[['表格变量', '处理方式', '实际特征', '训练集缺失率', '本轮开发集筛选状态', '最终选定模型入模']])}

### 3.1 实际公式与解释

{markdown_table(checklist[['表格变量', '实际公式', '业务解释与限制']])}

- dti为百分数，新增年供收入比为小数，故total_dti使用dti/100后相加；term为3或5年，全周期还款额使用term×12个月。
- employmentLength已转换成年数，直接用于short_emp_flag；employmentTitle是岗位编码，不能替代就业年限。
- 比率遇到非正收入设为缺失，保留income_nonpositive；输入缺失的普通业务标记也保留缺失，再由折内原有缺失流程处理，不提前误判为0。
- 信用账龄按年月差计算，负值设缺失；匿名有效值、缺失数量和缺失标记均在填补前逐行计算，不使用标签。
- n_valid_count按>0计算；当前原数据匿名字段负数为0，与原非零计数等价。新流程用规范名替换两项旧匿名计数，历史流程仍保留旧名。
- term_int_burden是交互项而非精确利息；uncleared_negative只表示两类记录的数量差，不能支持“历史污点已修复／仍存续”的判断。
- 阈值特征属于业务假设，名称和阈值本身不证明风险方向或因果作用。

### 3.2 筛选流程

业务衍生→匿名逐行聚合→缺失标记与众数填补→原一维／二维分组统计和标签熵→有序、One-Hot与目标编码→IV→Spearman→VIF。
IV阈值0.02、Spearman阈值0.8、VIF阈值10、筛选样本20,000，均与上轮一致。每折仅用该折训练月份拟合，之后转换验证月份；训练集和开发集另行拟合。
全部16项新候选进入相同筛选流程，不强制保留；各阶段完整IV及筛选状态见feature_selection_*.csv。相关性与VIF剔除合并标注为“相关性或VIF筛除”。

### 3.3 最终入模特征与业务含义（完整清单）

最终选定模型为 **{selected}**。下表严格依据该模型对应的final_features.csv，完整列示开发集筛选后实际使用的 **{len(feature_dictionary)}个特征**，不是前述24项业务候选，也不是某一时间验证折的临时特征。
“模型列序”保留最终输入列的原始顺序，仅用于核对，不代表特征重要性排名；本轮没有据此重新估计SHAP或风险方向。

{markdown_table(feature_summary)}

{markdown_table(feature_dictionary)}

**解释口径：** 单类别及双类别统计均先在对应拟合样本中按原流程完成缺失处理，再计算分组统计并映射至个体；最终模型的拟合样本为训练集与验证集合并的开发集。最大值、最小值、均值和中位数反映同组客户的分布背景，不能当作申请人自身数值；本轮未新增“个体值减组均值”等相对位置变量。
就业岗位、贷款用途、申请类型、贷款标题和邮编等字段仅按原编码区分群组，未根据数值臆测具体类别含义。直接入模的employmentTitle、title、postCode为目标编码后的数值，不是原始类别编号。
匿名字段n3、n14及匿名聚合只解释统计口径，未获得字段字典前不将其解释为某类账户或逾期行为。所有业务说明用于解释信息来源和经济直觉，不构成经验证的单调风险方向或因果结论。
完整可复用字典另存final_feature_dictionary.csv；本轮新增且最终保留的业务特征仍为total_dti和total_payment_income。

## 4. 调优设计、参数和时间交叉验证

随机种子42，模型线程4，tree_method=hist，分类阈值0.5。使用Optuna TPE，前5次随机启动，以训练期五折平均ROC-AUC最大化为目标，不使用验证集或测试集早停。
本轮50次独立搜索不导入上轮试验分数；两轮20+50条记录仅合并展示。旧、新特征下的试验分数不能当作同一个搜索空间内的连续试验。

{markdown_table(parameter_comparison)}

learning_rate、reg_alpha、reg_lambda采用对数采样，其余沿用原采样方式。固定参数对照为“第二轮_基线参数”和“第二轮_上轮最优参数”；它们用于与第一轮对应模型比较。

{markdown_table(fold_summary)}

{markdown_table(cv)}

各折指标的算术平均如下；PR-AUC、Precision、Recall和F1为逐折计算后求均值，不是将折外样本合并后重新计算。

{markdown_table(merged.loc[merged['集合'] == '训练期时间CV', ['模型', *METRICS]])}

五折样本按月份递进，早期折样本量小，各折样本量也不相等；目标仍为各折AUC的算术平均，CV标准差为样本标准差。目标编码的内部五折交叉拟合与外层时间验证分开执行，沿用原设计。

## 5. 验证集比较与模型锁定

{markdown_table(validation[['模型', *METRICS]])}

在测试评估前写入selection.json，按验证集AUC降序、KS降序选定 **{selected}**；完全并列时优先保留上轮选定模型。
第一轮指标和预测直接读取历史成果；第二轮三个候选在原训练集上拟合后评估同一验证集。测试结果不参与此选择。

## 6. 开发集重训与测试结果

第二轮三个候选均在训练集与验证集合并后重新拟合特征流程和模型；第一轮使用原开发集模型及原测试预测。
分类指标固定阈值0.5，PR-AUC采用average precision。

{markdown_table(test[['模型', *METRICS, 'AUC差_相对上轮基线', 'AUC差_相对上轮调优']])}

最终选定模型混淆矩阵：

{markdown_table(pd.DataFrame([{'实际类别': '正常', '预测正常': int(final['TN']), '预测违约': int(final['FP'])}, {'实际类别': '违约', '预测正常': int(final['FN']), '预测违约': int(final['TP'])}]))}

## 7. 特征变化与重新调参的分解比较

下表均为“比较模型－参照模型”的绝对指标差，不是相对增长率。固定参数比较刻画加入特征并重新筛选后的整体流程变化，不能归因于某一个变量；重新调参比较则固定新特征流程。

{markdown_table(effects[['集合', '比较', 'AUC差', 'KS差', 'PR-AUC差', 'Precision差', 'Recall差', 'F1差']])}

判断是否改善应同时查看验证集与测试集、排序指标与固定阈值指标。未做显著性检验，较小差值不能直接宣称统计显著或稳定业务收益。

## 8. 校验、局限与交付

边界检查覆盖零收入、缺失输入、利用率30／70、FICO 660、账龄24个月、就业2年、月供比0.05、匿名缺失2／3项、全缺失和负数计数语义。断言通过；旧业务特征的构造值保持一致。

{markdown_table(replay)}

模型包保存重载后，五个候选各抽取512条测试记录回放，允许省略标签，概率最大容差2e-7；第一轮CSV的概率截断误差也纳入该容差。历史输入、模型、预测、旧报告和旧脚本的哈希保护检查通过。

**限制：** 测试集已在上一轮评估，本轮是同一测试集上的回顾性比较，不是新的盲测；验证集也被重复用于候选选择，仍需新的时间外样本检验。原标签熵及目标编码等监督特征设计未在本轮重构；本轮不把既有设计说成新增的严格逐时点交叉拟合。
两个类别标记因缺少编码字典暂缓；不根据模型结果反推类别含义。不调整分类阈值，也不将特征重要性或单次AUC变化解释为因果结论。

结果目录：{output_dir.relative_to(PROJECT_DIR).as_posix()}。
主要记录包括trials_merged.csv、trial_fold_metrics.csv、metrics_merged.csv、effect_comparisons.csv、feature_checklist.csv、final_feature_dictionary.csv、candidate_parameters.csv、验证／测试／折外预测、final_model.joblib、final_features.csv及run_manifest.json。
清单中的“本轮流程保留”指新流程开发集筛选结果，“最终选定模型入模”按最终选择的实际特征版本判断；若选中历史模型，两列可能不同。

运行入口：code/model/xgboost_round2.py；特征模块：code/model/xgboost_round2_features.py。使用项目venv运行；传入--output-dir可恢复同一目录，--report-only仅从完成产物重建报告，--check-only只检查输入和边界。
预测时将code/model加入Python模块搜索路径，joblib.load加载模型包后调用xgboost_round2.predict_round2(bundle, cleaned_data)，自动按最终选定模型分派旧／新特征版本。
本轮没有覆盖第一轮成果。环境版本和文件SHA256见run_manifest.json；搜索状态、失败信息与逐步进度见逐试验表和execution.log。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    FINAL_REPORT_PATH.write_text(report, encoding="utf-8")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest.update(
        report_generated_at=datetime.now().isoformat(), report_renderer_sha256=file_hash(Path(__file__)),
        feature_dictionary_renderer_sha256=file_hash(Path(__file__).with_name("xgboost_round2_report_features.py")),
        final_report_path=str(FINAL_REPORT_PATH),
    )
    write_json(manifest, output_dir / "run_manifest.json")


class Tee:
    def __init__(self, original, stream):
        self.original = original
        self.stream = stream

    def write(self, text):
        self.original.write(text)
        self.stream.write(text)
        self.stream.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.stream.flush()


def main():
    parser = argparse.ArgumentParser(description="第二轮XGBoost业务特征优化与两轮合并报告")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        if args.output_dir is None:
            parser.error("--report-only需要--output-dir")
        write_report(args.output_dir.resolve())
        return
    data, assignment, summary, audit = load_inputs()
    if args.check_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(summary.to_string(index=False))
        return
    if args.n_trials != 50:
        parser.error("本轮锁定50次搜索；固定参数对照不占用搜索次数")
    root = PROJECT_DIR / "results" / "xgboost_optimization_round2"
    output_dir = (args.output_dir or root / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    if not output_dir.is_relative_to(root.resolve()) or output_dir == root.resolve():
        parser.error("输出目录必须是独立的results/xgboost_optimization_round2子目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_protected(manifest["protected_files"])
        if manifest["status"] == "complete":
            write_report(output_dir)
            log(f"实验已完成，仅重建报告：{REPORT_PATH}")
            return
        if file_hash(Path(__file__).with_name("xgboost_round2_features.py")) != manifest["feature_module_sha256"]:
            raise ValueError("特征代码变更，不能沿用旧缓存，请创建新运行目录")
        manifest.update(status="running", resumed_at=datetime.now().isoformat())
    else:
        manifest = {
            "status": "running", "started_at": datetime.now().isoformat(), "project_dir": str(PROJECT_DIR),
            "output_dir": str(output_dir), "source_dir": str(SOURCE_DIR), "n_trials": 50,
            "n_splits": 5, "seed": legacy.SEED, "n_jobs": legacy.N_JOBS, "threshold": legacy.THRESHOLD,
            "feature_version": FEATURE_VERSION, "python": platform.python_version(),
            "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost", "optuna", "joblib"]},
            "protected_files": protected_files(),
            "feature_module_sha256": file_hash(Path(__file__).with_name("xgboost_round2_features.py")),
            "runner_sha256": file_hash(Path(__file__)),
            "selection_uses_test": False, "test_status": "与上轮相同测试集，回顾性比较，非全新盲测",
        }
    write_json(manifest, manifest_path)
    write_json(audit, output_dir / "input_boundary_audit.json")
    save_csv(summary, output_dir / "sample_summary.csv")
    save_csv(assignment, output_dir / "split_assignment.csv")
    with (output_dir / "execution.log").open("a", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            log(f"输出目录：{output_dir}")
            try:
                run_experiment(data, assignment, output_dir, manifest, args.n_trials)
            except BaseException as error:
                manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error), failed_at=datetime.now().isoformat())
                write_json(manifest, manifest_path)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
