# 信贷违约预测

面向贷款违约识别的机器学习实践项目，包含数据清洗、特征工程、逻辑回归 / 随机森林 / XGBoost 对比、时间切分调优、阈值分析及项目报告。

**主入口：[code/main_pipelin.ipynb](code/main_pipelin.ipynb)**。

> 项目已有离线实验与报告，但不是已上线的信贷审批系统。整理日期为 2026-11-11；本次仅整理文件、补充环境与上传说明，没有重训模型或重新计算性能指标。

## 快速阅读

- [最终项目报告](docs/最终报告/最终项目报告.md)：完整评估、阈值分析与局限。
- [管理层汇报](docs/最终报告/项目汇报_管理层版.md)：业务侧摘要。
- [最终交付使用说明](docs/最终报告/使用说明.md)：既有流程的操作说明。

## 项目流程

1. 从原始贷款记录执行确定性清洗，并核对已保存的处理数据。
2. 按放款时间划分训练、验证和测试集，避免把早期探索实验与最终评估口径混为一谈。
3. 在训练窗口内处理缺失、类别编码、分组聚合、交叉特征和特征筛选。
4. 对比基线与调优模型，通过时间交叉验证和验证集选择参数及模型。
5. 在验证集研究分类阈值与复核容量，再评估测试期表现。
6. 输出审计表、特征解释、风险分层、图表和最终报告。

具体行为以主 notebook 和 `code/model/` 中的实现为准。早期 notebook 与周报用于展示项目迭代，不代表同一口径下的最终结果。

## 目录结构

```text
credit/
├── README.md
├── requirements.txt             # 直接依赖：来自现有环境的实际版本
├── requirements-lock.txt        # 现有 venv 的完整 pip freeze 快照
├── .gitignore
├── .gitattributes
├── code/
│   ├── main_pipelin.ipynb       # 最终项目流程
│   ├── preprocessing*.ipynb    # 早期清洗与改进实验
│   ├── feature_XGBOOST.ipynb    # 特征工程实验
│   ├── XGBOOST.ipynb            # 早期 XGBoost 实验
│   └── model/                  # 模型对比、调优、特征模块与报告模板
├── data/                       # 原始/处理数据、特征矩阵，仅保留本地
├── docs/
│   ├── 计划/                   # 项目任务、数据字典与业务理解
│   ├── 周报/
│   ├── 最终报告/
│   ├── model/                  # 已有三模型对比报告与图表
├── results/                    # 保留目录名，避免破坏代码和报告引用
│   ├── final_project/          # 最终评估、阈值分析和审计汇总
│   ├── xgboost_optimization/   # 第一轮结果，主入口默认重放依赖
│   └── ...                     # 其他模型及历史实验结果
├── scripts/                    # 上传包导出及 notebook 维护工具
└── dist/                       # 生成的干净上传包，不进入 Git
```

本地 `venv/` 和 `.vscode/` 保留，但不进入上传包。结果目录保留汇总指标、报告与图表；模型二进制、逐笔预测、样本分配及大型搜索表只保留本地。

## 环境准备

现有运行记录和本机环境使用 **Python 3.14.6**。`requirements.txt` 锁定项目直接依赖，`requirements-lock.txt` 记录整理时现有 `venv` 的完整包版本；本次没有升级环境，也没有在全新环境重新安装验证。

已有本地项目可继续使用 `venv/Scripts/python.exe`。在新位置准备环境时，可参考：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name credit --display-name "Python (credit)"
```

在 VS Code/Jupyter 中选择对应内核。不要在未核对兼容性的情况下升级依赖后直接加载旧 `joblib`；也不要加载不可信来源的模型文件。

## 运行方式与边界

**克隆或解压后的公开版本不包含贷款原始数据、处理数据和已训练模型，因此不能不补输入就直接运行完整流程。** 先按[数据说明](data/README.md)恢复本地输入。

1. 打开 `code/main_pipelin.ipynb`，从第一个单元格依次执行。
2. 默认 `RUN_TRAINING=False`：读取原始数据、核对清洗快照，加载 `results/xgboost_optimization/` 的模型与预测，重新生成评估和报告。此模式是**模型重放，不是重新训练**。
3. 设为 `RUN_TRAINING=True`：执行重新训练与调优，输出至 `results/final_pipeline_runs/<时间戳>/`。该流程耗时更长，本次整理未执行。

主 notebook 会向上查找项目数据路径。部分早期实验仍含 `D:/credit` 绝对路径或相对于 `code/` 的输入路径；它们是历史实验，不是推荐的跨机器运行入口。此轮未为整理文件而修改其模型逻辑。

`code/model/` 中的自定义变换模块和报告模板必须随项目保留；不能只复制单个 notebook 作为完整项目。

## 已保存结果摘要

下表来自已有 `results/final_project/recomputed_metrics.csv`，为阈值 0.5 的测试集结果，本次未重新估计：

| 模型         |    AUC |     KS | PR-AUC | Precision | Recall |     F1 |
| ------------ | -----: | -----: | -----: | --------: | -----: | -----: |
| 基线 XGBoost | 0.7220 | 0.3270 | 0.3793 |    0.5093 | 0.0966 | 0.1624 |
| 调优 XGBoost | 0.7198 | 0.3263 | 0.3756 |    0.5304 | 0.0669 | 0.1188 |

调优模型的测试 AUC 没有超过同口径基线，0.5 阈值下召回率也较低。不能把项目概括为“调优全面提升效果”，也不能把回顾性阈值分析当成上线经营收益验证。时间稳定性、标签观察窗口和业务容量约束见最终报告。
