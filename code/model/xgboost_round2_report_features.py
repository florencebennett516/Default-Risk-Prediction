import pandas as pd

import xgboost_optimization as legacy


FIELD_NAMES = {
    "loanAmnt": "贷款金额", "interestRate": "贷款利率", "installment": "月供金额",
    "annualIncome": "年收入", "dti": "原有债务收入比", "ficoRange_mean": "FICO信用分区间中点",
    "revolBal": "循环信贷余额", "revolUtil": "循环额度利用率", "term": "贷款期限",
}
CATEGORY_NAMES = {
    "grade": "贷款等级", "subGrade": "贷款子等级", "employmentTitle": "就业岗位编码",
    "homeOwnership": "住房持有状态", "verificationStatus": "收入核验状态", "purpose": "贷款用途编码",
    "postCode": "邮编编码", "regionCode": "地区编码", "initialListStatus": "初始列表状态",
    "applicationType": "申请类型编码", "title": "贷款标题编码",
}
STATISTIC_NAMES = {"max": "最大值", "min": "最小值", "mean": "均值", "median": "中位数"}
STATISTIC_MEANINGS = {
    "max": "已观察上界", "min": "已观察下界", "mean": "平均水平", "median": "中位水平",
}
FIELD_MEANINGS = {
    "loanAmnt": "借款规模", "interestRate": "贷款定价", "installment": "月供负担",
    "annualIncome": "收入能力", "dti": "既有偿债负担", "ficoRange_mean": "信用分水平",
    "revolBal": "循环负债规模", "revolUtil": "循环额度占用程度",
}
DIRECT_FEATURES = {
    "term": ("基础数值特征", "贷款期限", "直接使用term，当前为3或5年", "描述合同期限和风险暴露时长，不等于实际存续时长"),
    "loanAmnt": ("基础数值特征", "贷款金额", "直接使用loanAmnt", "衡量本次借款规模，需结合收入与偿债压力解释"),
    "annualIncome": ("基础数值特征", "年收入", "直接使用annualIncome", "衡量申报收入规模；非正值本身不表示具有正常偿付能力"),
    "revolUtil": ("基础数值特征", "循环额度利用率", "直接使用revolUtil，百分数口径", "反映循环授信额度已占用比例，不等于全部债务收入比"),
    "ficoRange_mean": ("业务衍生特征", "FICO信用分区间中点", "(ficoRangeLow + ficoRangeHigh) / 2", "以分数区间中点代理客户信用分水平"),
    "total_dti": ("业务衍生特征", "合计偿债收入比", "dti / 100 + installment * 12 / annualIncome", "统一小数单位后叠加原有债务与新增贷款年供压力，作为合计偿债压力代理"),
    "total_payment_income": ("业务衍生特征", "全周期分期还款收入比", "installment * term * 12 / annualIncome", "衡量合同全部分期还款总额相对一年收入的规模，不是年度DTI"),
    "n_valid_count": ("匿名字段聚合特征", "匿名字段正值数量", "在填补前统计n0至n14中大于0的字段个数", "反映匿名字段正值覆盖程度；缺少数据字典，不能等同有效信贷账户数"),
    "anonymous_nonzero_count": ("匿名字段聚合特征", "匿名字段非零数量", "在填补前统计n0至n14中非缺失且不等于0的字段个数", "反映匿名字段非零覆盖程度，不能推断具体征信业务含义"),
}


def describe_feature(name):
    if name in DIRECT_FEATURES:
        return DIRECT_FEATURES[name]
    if name in {"employmentTitle", "title", "postCode", "regionCode"}:
        chinese = CATEGORY_NAMES[name]
        return (
            "类别目标编码特征", f"{chinese}目标编码",
            f"对{chinese}使用平滑目标编码；训练阶段内部五折交叉拟合，预测阶段使用拟合映射",
            f"表示同类{chinese}在拟合样本中的违约率相关统计信息；编码值不是原始类别编号，也不是个体真实违约概率",
        )
    if name in {f"n{index}" for index in range(15)}:
        return (
            "原始匿名特征", f"匿名字段{name}", f"保留原始{name}，经拟合样本的缺失处理后入模",
            "仅能确认其为数据源提供的匿名数值信息；缺少公开字段释义，不赋予逾期次数、账户数量等具体业务标签",
        )
    groups = [tuple(pair) for pair in legacy.CATEGORY_PAIRS] + [(category,) for category in legacy.CATEGORICAL_VARIABLES]
    for categories in groups:
        prefix = "_".join(categories) + "_"
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix):]
        for field in legacy.CONTINUOUS_VARIABLES:
            for statistic in STATISTIC_NAMES:
                if remainder != f"{field}_{statistic}":
                    continue
                grouping = "、".join(CATEGORY_NAMES[category] for category in categories)
                field_name = FIELD_NAMES[field]
                statistic_name = STATISTIC_NAMES[statistic]
                kind = "双类别交叉分组统计特征" if len(categories) == 2 else "单类别分组统计特征"
                return (
                    kind, f"同{grouping}组的{field_name}{statistic_name}",
                    f"拟合样本按{grouping}分组，计算{field}的{statistic}，再映射至该组客户",
                    f"以同组{FIELD_MEANINGS[field]}的{STATISTIC_MEANINGS[statistic]}刻画客群背景；不是该客户自身的{field_name}",
                )
    raise ValueError(f"最终特征尚无与实际流程匹配的业务解释：{name}")


def build_final_feature_dictionary(feature_names):
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("最终特征清单包含重复项")
    rows = []
    for number, name in enumerate(feature_names, 1):
        kind, chinese, calculation, meaning = describe_feature(name)
        rows.append({
            "模型列序": number, "特征名称": name, "特征类型": kind,
            "中文名称": chinese, "计算方式": calculation, "业务含义": meaning,
        })
    return pd.DataFrame(rows)
