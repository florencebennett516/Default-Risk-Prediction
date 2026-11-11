# 本地数据与重放输入

## 当前文件

以下文件在本次整理中保留原位，内容不变，不随 GitHub 导出包公开：

| 路径 | 内容 |
| --- | --- |
| `data/credit_data(1).csv` | 原始贷款记录 |
| `data/data_processing.csv` | 已保存的清洗数据快照 |
| `data/xgboost_features/train.parquet` | 早期实验的训练特征 |
| `data/xgboost_features/val.parquet` | 早期实验的验证特征 |
| `data/xgboost_features/test.parquet` | 早期实验的测试特征 |
| `data/xgboost_features/metadata.json` | 早期特征矩阵元信息 |

保留现有文件层级，是为了不改变主 notebook、历史实验及模型的路径约定。此目录除本说明外全部默认忽略；不会把数据搬动后留下失效的运行入口。

## 数据来源与使用条件

项目中保留了实习任务材料和数据字典，参见 `docs/计划/`。本次没有核实原始数据下载渠道、完整授权条款或再分发许可，不提供猜测下载地址。请通过原项目合法渠道获取相应数据，确认使用条件后按上表放置。

贷款记录含收入、信用及贷款信息，不应仅凭字段已编码就认定可以公开。导出包不包含原始贷款记录、处理数据、逐笔预测和样本分配。

## 默认模型重放所需输入

`code/main_pipelin.ipynb` 的 `RUN_TRAINING=False` 除上述两个 CSV 外，还依赖本地 `results/xgboost_optimization/` 中的已有模型与结果，例如：

- `final_model.joblib`：已训练模型及自定义预处理对象。
- `validation_predictions.csv`、`test_predictions.csv`：保存的逐笔预测，用于重放核对。
- `sample_summary.csv`、`fold_summary.csv`、`best_parameters.csv`、`cv_summary.csv`、`validation_metrics.csv`、`final_features.csv` 等：与该轮模型对应的汇总证据。

这不是独立数据包的完整授权交付清单。模型、数据、特征模块和预测必须来自同一版本，不要混用不同实验轮次。只有汇总结果而没有原始数据及模型时，可以阅读报告，但不能执行默认重放。

改为重新训练仍然需要合法获取数据和完整依赖，并会消耗较多计算资源。仅加载可信本地模型；原始输入和已训练模型的摘要可参照已有 `results/final_project/run_manifest.json`，其中的绝对路径和生成时间是原运行记录，不代表新机器路径。
