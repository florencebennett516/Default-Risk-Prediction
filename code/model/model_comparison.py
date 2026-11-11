from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "docs" / "model_comparison"
MODEL_NAMES = {"Regression": "逻辑回归", "XGBOOST": "XGBoost", "RandomForest": "随机森林"}
COLORS = {"Regression": "#377eb8", "XGBOOST": "#e45756", "RandomForest": "#59a14f"}
SPLIT_NAMES = {"train": "训练集", "val": "验证集", "test": "测试集"}


def score_psi(reference, current, edges):
    reference_share = np.maximum(np.histogram(reference, bins=edges)[0] / len(reference), 1e-6)
    current_share = np.maximum(np.histogram(current, bins=edges)[0] / len(current), 1e-6)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def train_and_predict(features, labels):
    predictions = {}
    estimators = {
        "Regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20,
                                   random_state=42, n_jobs=-1),
        ),
    }
    for model_key, estimator in estimators.items():
        print(f"Training {model_key}", flush=True)
        estimator.fit(features["train"], labels["train"])
        predictions[model_key] = {
            split: estimator.predict_proba(frame)[:, 1] for split, frame in features.items()
        }
    print("Training XGBOOST", flush=True)
    matrices = {split: xgb.DMatrix(frame, label=labels[split]) for split, frame in features.items()}
    parameters = {
        "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
        "max_depth": 5, "min_child_weight": 5, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42,
    }
    booster = xgb.train(parameters, matrices["train"], num_boost_round=1000,
                        evals=[(matrices["train"], "train"), (matrices["val"], "val")],
                        early_stopping_rounds=50, verbose_eval=100)
    best_rounds = booster.best_iteration + 1
    booster = booster[:best_rounds]
    predictions["XGBOOST"] = {split: booster.predict(matrix) for split, matrix in matrices.items()}
    return predictions, best_rounds


def plot_results(predictions, labels, metrics, risk_groups):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_key, model_name in MODEL_NAMES.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(labels["test"], predictions[model_key]["test"])
        auc = metrics.loc[(metrics["模型"] == model_name) & (metrics["样本"] == "测试集"), "AUC"].iloc[0]
        axis.plot(false_positive_rate, true_positive_rate, color=COLORS[model_key], label=f"{model_name} (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    axis.set(xlabel="正常样本误报率（FPR）", ylabel="违约样本召回率（TPR）",
             title="测试集 ROC 曲线", xlim=(0, 1), ylim=(0, 1))
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "roc.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    bad_count = int(labels["test"].sum())
    good_count = len(labels["test"]) - bad_count
    for axis, (model_key, model_name) in zip(axes, MODEL_NAMES.items()):
        false_positive_rate, true_positive_rate, _ = roc_curve(labels["test"], predictions[model_key]["test"])
        population_share = (true_positive_rate * bad_count + false_positive_rate * good_count) / len(labels["test"])
        gap = true_positive_rate - false_positive_rate
        best_position = int(np.argmax(gap))
        axis.plot(population_share, true_positive_rate, color="#e45756", label="累计违约占比")
        axis.plot(population_share, false_positive_rate, color="#377eb8", label="累计正常占比")
        axis.plot(population_share, gap, color="#59a14f", linestyle="--", label="两者差值")
        axis.vlines(population_share[best_position], false_positive_rate[best_position],
                    true_positive_rate[best_position], color="gray", linestyle=":")
        axis.set(title=f"{model_name}：KS={gap[best_position]:.4f}",
                 xlabel="累计样本占比（风险从高到低）", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("累计占比 / 差值")
    axes[-1].legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "ks.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for model_key, model_name in MODEL_NAMES.items():
        groups = risk_groups.loc[risk_groups["模型"] == model_name]
        axis.plot(groups["风险档"], groups["坏账率"] * 100, marker="o",
                   color=COLORS[model_key], label=model_name)
    axis.axhline(labels["test"].mean() * 100, color="gray", linestyle="--",
                 label=f"测试集整体：{labels['test'].mean():.2%}")
    axis.set(xlabel="风险档（1最低，10最高；使用各模型训练集分档边界）",
             ylabel="实际坏账率（%）", title="测试集分档坏账率", xticks=range(1, 11))
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "bad_rate.png", dpi=160)
    plt.close(figure)


def main():
    feature_dir = PROJECT_DIR / "data" / "xgboost_features"
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    features, labels = {}, {}
    for split in SPLIT_NAMES:
        frame = pd.read_parquet(feature_dir / f"{split}.parquet")
        features[split] = frame.loc[:, metadata["feature_columns"]]
        labels[split] = frame[metadata["target"]]
    predictions, best_rounds = train_and_predict(features, labels)
    metric_rows, group_rows = [], []
    edges_by_model = {}
    for model_key, model_name in MODEL_NAMES.items():
        saved = pd.read_csv(PROJECT_DIR / "results" / model_key / "metrics.csv").set_index("样本")
        edges = np.unique(np.quantile(predictions[model_key]["train"], np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        edges_by_model[model_key] = edges
        for split, split_name in SPLIT_NAMES.items():
            probability = predictions[model_key][split]
            false_positive_rate, true_positive_rate, _ = roc_curve(labels[split], probability)
            auc = roc_auc_score(labels[split], probability)
            ks = float((true_positive_rate - false_positive_rate).max())
            if not np.allclose([auc, ks], saved.loc[split_name, ["AUC", "KS"]].astype(float), atol=1e-9, rtol=0):
                raise ValueError(f"{model_key}/{split} 与已有指标不一致，请核对环境和输入")
            metric_rows.append({"模型": model_name, "样本": split_name, "AUC": auc, "KS": ks,
                                "PSI_相对训练集": score_psi(predictions[model_key]["train"], probability, edges)})
        groups = pd.DataFrame({
            "风险档": pd.cut(predictions[model_key]["test"], bins=edges, labels=False, right=False) + 1,
            "实际违约": labels["test"].to_numpy(), "预测概率": predictions[model_key]["test"],
        }).groupby("风险档").agg(样本数=("实际违约", "size"), 违约数=("实际违约", "sum"),
                                 坏账率=("实际违约", "mean"), 平均预测概率=("预测概率", "mean")).reset_index()
        groups.insert(0, "模型", model_name)
        group_rows.append(groups)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    risk_groups = pd.concat(group_rows, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    risk_groups.to_csv(OUTPUT_DIR / "risk_groups.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"样本": SPLIT_NAMES[split], "样本数": len(target), "违约数": int(target.sum()), "违约率": target.mean()}
        for split, target in labels.items()
    ]).to_csv(OUTPUT_DIR / "split_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "calculation_settings.json").write_text(json.dumps({
        "xgboost_best_rounds": best_rounds, "psi_reference": "train",
        "psi_bins": "training score deciles", "psi_epsilon": 1e-6, "metrics_match_saved_csv": True,
        "bin_edges": {key: [float(value) if np.isfinite(value) else str(value) for value in edges]
                      for key, edges in edges_by_model.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(predictions, labels, metrics, risk_groups)
    print(metrics.to_string(index=False), flush=True)
    print(risk_groups.to_string(index=False), flush=True)
    print(f"XGBoost best rounds: {best_rounds}; output: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
