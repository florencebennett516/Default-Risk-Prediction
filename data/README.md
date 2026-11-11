# 本地数据与运行输入

## GitHub 发布边界

**GitHub 版本仅保留本说明，不提供原始贷款数据、清洗快照、特征矩阵、已训练模型和逐笔预测。** 本地旧分支或工作目录可能仍保留历史数据文件，本次只发布文档更新，不将这些文件或其旧提交重新推送到远端。

以下是代码的本地路径约定，不是仓库交付清单：

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `data/credit_data(1).csv` | 原始贷款记录 | 最终流程清洗、训练及重放所需 |
| `data/data_processing.csv` | 清洗数据快照 | 最终流程一致性核对所需 |
| `data/xgboost_features/train.parquet` | 历史实验训练特征 | 非最终模型的完整运行输入 |
| `data/xgboost_features/val.parquet` | 历史实验验证特征 | 非最终模型的完整运行输入 |
| `data/xgboost_features/test.parquet` | 历史实验测试特征 | 非最终模型的完整运行输入 |
| `data/xgboost_features/metadata.json` | 历史特征元信息 | 需与相应实验版本配套 |

保留本地路径约定，不移动用户数据。`.gitignore` 不会自动取消对旧分支已跟踪文件的跟踪；发布应以远端已清理数据的版本为基底，而不是推送含数据的本地旧历史。

## 数据来源与使用条件

项目任务材料和数据字典见 `docs/计划/`。本次未核实完整授权条款或再分发许可，输入需通过原项目合法渠道取得。贷款记录及其衍生特征含收入、信用和贷款信息，编码或特征化不等于获得再分发许可。

## 默认模型重放所需输入

`code/main_pipelin.ipynb` 的 `RUN_TRAINING=False` 除上述两个 CSV 外，还依赖本地 `results/xgboost_optimization/` 中的同版本模型与结果。该路径是代码输入约定，不表示仓库提供了相应文件：

- `final_model.joblib`：已训练模型及自定义预处理对象。
- `validation_predictions.csv`、`test_predictions.csv`：保存的逐笔预测。
- `sample_summary.csv`、`fold_summary.csv`、`best_parameters.csv`、`cv_summary.csv`、`validation_metrics.csv`、`final_features.csv` 等：该轮模型对应的汇总文件。

模型、数据、特征模块和预测应来自同一轮实验。改为 `RUN_TRAINING=True` 仍需原始 CSV、清洗快照和完整依赖。运行清单由代码生成，本仓库不是完整授权数据包；仅加载可信本地模型。

## 策略监控的数据补充

现有标签为 `isDefault`，尚未计算 vintage 30+ / 60+ DPD。按放款月、同账龄监控逾期，需要补齐还款计划、实际还款、到期日、观察日及逾期天数，并固定成熟窗口和分母；详见[策略建议 memo](../docs/最终报告/策略建议memo.md)。
