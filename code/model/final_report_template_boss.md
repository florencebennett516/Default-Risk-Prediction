# 信贷风险模型：管理层决策汇报

汇报日期：{date}｜模型：{selected_model}｜特征：{feature_count} 个｜测试：{test_size:,} 笔

## 一、建议决策

**推进以人工复核为入口的容量约束试点，将工作重点转向客户筛选与资源配置。**

- cutoff 参考线为 **{applied_threshold:.4f}**；测试拒绝比例 **{applied_rejection_rate:.2%}**，通过组坏率 **{applied_pass_bad_rate:.2%}**，总体坏率 **{test_bad_rate:.2%}**。
- 优先复核最高风险 10%，该组坏率 **{top_bad_rate:.2%}**、Lift **{top_lift:.2f}**，捕获 **{top_capture:.2%}** 的违约。
- 以 **Top-K 分位队列**控制跨期处理容量，最高风险 10% 包含在高风险队列内；入队和实际拒绝分别统计。

以上为历史回测。Top-K 效果、vintage 逾期指标和成本收益进入下一阶段验证。

## 二、模型效果与迭代判断

项目样本共 {total_size:,} 笔，按完整放款月份划分，采用 {cv_folds} 折时间验证和 {n_trials} 次参数搜索，按验证期表现选定模型。早期随机划分实验单独归档。

{sample_table}

### 2.1 测试表现

{test_metrics_table}

分类指标采用阈值 0.5。调优模型验证 AUC 为 {validation_auc:.4f}，基线为 {validation_baseline_auc:.4f}；测试 AUC 为 {auc:.4f}，基线为 {baseline_auc:.4f}。下一轮在相同入队比例下比较客户质量、违约捕获和处理成本，再决定模型复杂度。

### 2.2 Bootstrap 区间

固定模型实施 {bootstrap_repetitions} 次配对 bootstrap，报告百分位 95% 区间。逐笔方法共享样本权重，整月方法按月分块；重训与调参波动另行评价。

{interval_table}

## 三、阈值与人工复核

阈值在验证期以拒绝率不超过 {max_rejection_rate:.0%}、通过组坏率最低为目标选定，再锁定应用于测试期。

| 指标 | 测试表现 |
| --- | ---: |
| cutoff | {applied_threshold:.4f} |
| 拒绝比例 | {applied_rejection_rate:.2%} |
| 违约捕获率 | {applied_recall:.2%} |
| 通过组坏率 | {applied_pass_bad_rate:.2%} |

固定概率阈值对应的客户占比随分数分布变化，须逐批核对容量。建议按申请池排序，将入队量限制为申请数的 {max_rejection_rate:.0%} 与实际处理预算中的较小者，同分采用预定顺序。最高风险 10% 优先人工核查收入、负债与申请材料，其余客户沿用既有审批政策。Top-K 需单独回测，上表是固定 cutoff 的结果。

## 四、漂移与组合质量监控

### 4.1 PSI 与缺失率

{drift_table}

- n14 PSI 为 **{n14_psi:.4f}**，缺失率 **{n14_development_missing:.2%} → {n14_test_missing:.2%}**；先查采集、填补与客群变化。
- revolUtil PSI 为 **{revol_psi:.4f}**；检查利用率分布、缺失、极值和变化批次，关联同账龄逾期表现。
- 沿用开发集固定分箱，联合分数、通过比例及成熟样本坏率排查异常。

### 4.2 监控看板

| 频率 | 指标 | 动作 |
| --- | --- | --- |
| 每批 / 每日 | 入队率、通过率、实际拒绝率、复核量和处理时长 | 按预算控制 K，记录人工结果 |
| 每月 | 通过组坏率、复核命中率、客户流失与成本 | 固定观察窗口，比较等容量方案 |
| 按放款月 vintage | 同账龄 30+ / 60+ DPD | 接入还款与逾期数据，以成熟贷款为分母 |
| 每月 / 异常批次 | 分数与字段 PSI、缺失率、极值 | 优先排查 n14 与 revolUtil |

现有 isDefault 标签尚未形成 vintage 30+ / 60+ DPD，需补齐还款与逾期明细。

## 五、试点推进与验收

1. 风控确认标签窗口、字段申请时点与入队上限；运营确认 Top 10% 复核预算，数据团队补齐 vintage 标签。
2. 先做基线与调优模型的 Top-K 等容量回测，检查季度稳定性。最新季度 {latest_quarter} 有 {latest_quarter_size:,} 笔、{latest_quarter_defaults:,} 笔违约，AUC {latest_quarter_auc:.4f}，列为优先复验批次。
3. 小范围辅助复核保留既有审批政策，记录模型建议、人工结论与实际通过，跟踪成熟样本质量和成本后决定扩量。

本报告随对应运行结果生成；GitHub 不提供原始数据、特征矩阵和最终模型。运行条件见仓库中的 `docs/最终报告/使用说明.md`。
