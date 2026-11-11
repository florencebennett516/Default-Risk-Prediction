import numpy as np
import pandas as pd

from model_optimization_features import AnonymousFeatureAggregator, build_features


FEATURE_VERSION = "business_v2"
ANONYMOUS_COLUMNS = [f"n{index}" for index in range(15)]
FEATURE_SPECS = [
    ("new_installment_ratio", "新增", "new_installment_ratio", "installment * 12 / annualIncome", "新增贷款年供收入比"),
    ("total_dti", "新增", "total_dti", "dti / 100 + new_installment_ratio", "小数单位的合计偿债压力"),
    ("loan_to_income", "新增", "loan_to_income", "loanAmnt / annualIncome", "本金收入比；不同于原按期限年化的比率"),
    ("total_payment_income", "新增", "total_payment_income", "installment * term * 12 / annualIncome", "term为年，换算月数后的全周期还款收入比"),
    ("revol_income", "复用", "annual_revolBal_income_ratio", "revolBal / annualIncome", "循环余额收入比"),
    ("util_high_flag", "新增", "util_high_flag", "revolUtil > 70", "循环额度高占用"),
    ("util_mid_flag", "新增", "util_mid_flag", "30 < revolUtil <= 70", "循环额度中等占用"),
    ("open_acc_ratio", "复用", "openAcc_ratio", "openAcc / totalAcc", "在用账户占比"),
    ("closed_acc", "复用", "closeAcc", "totalAcc - openAcc", "关闭账户数，负数按原流程设为缺失"),
    ("delinq_per_acc", "新增", "delinq_per_acc", "delinquency_2years / (totalAcc + 1)", "近两年逾期密度"),
    ("has_pub_rec", "新增", "has_pub_rec", "pubRec > 0", "公共负面记录标记"),
    ("uncleared_negative", "新增", "uncleared_negative", "pubRec - pubRecBankruptcies", "仅为记录数量差，不能识别污点是否已清除"),
    ("fico_score", "复用", "ficoRange_mean", "(ficoRangeLow + ficoRangeHigh) / 2", "FICO区间中点"),
    ("fico_low_flag", "新增", "fico_low_flag", "ficoRange_mean < 660", "低FICO标记"),
    ("credit_age_month", "新增", "credit_age_month", "12 * (issue_year - first_credit_year) + issue_month - first_credit_month", "精确信用账龄月数，不等同原年份差"),
    ("new_credit_flag", "新增", "new_credit_flag", "credit_age_month < 24", "信用账龄不足24个月"),
    ("short_emp_flag", "新增", "short_emp_flag", "employmentLength < 2", "使用清洗后就业年限，不使用岗位编码"),
    ("term_int_burden", "新增", "term_int_burden", "term * interestRate", "年数乘百分数利率的交互项，不是实际总利息"),
    ("high_install_flag", "新增", "high_install_flag", "installment / annualIncome > 0.05", "单月月供与年收入之比超阈值"),
    ("joint_app_flag", "暂缓", None, "待确认applicationType编码", "不生成标记，保留原申请类型"),
    ("debt_consolidation_flag", "暂缓", None, "待确认purpose编码", "不生成标记，保留原用途"),
    ("n_valid_count", "复用", "n_valid_count", "count(n0..n14 > 0)", "替换anonymous_nonzero_count；当前数据无负数"),
    ("n_missing_count", "复用", "n_missing_count", "count(isna(n0..n14))", "替换anonymous_missing_count，填补前计数"),
    ("n_missing_flag", "新增", "n_missing_flag", "n_missing_count >= 3", "匿名字段多项缺失，填补前生成"),
]
NEW_FEATURES = [name for name, status, _, _, _ in FEATURE_SPECS if status == "新增"]
LEGACY_NAMES = {"n_valid_count": "anonymous_nonzero_count", "n_missing_count": "anonymous_missing_count"}


def nullable_flag(values, condition):
    return condition.astype("float32").where(values.notna())


def build_features_v2(data):
    features = build_features(data)
    income = data["annualIncome"].where(data["annualIncome"] > 0)
    term = data["term"].where(data["term"] > 0)
    features["new_installment_ratio"] = data["installment"] * 12 / income
    features["total_dti"] = data["dti"] / 100 + features["new_installment_ratio"]
    features["loan_to_income"] = data["loanAmnt"] / income
    features["total_payment_income"] = data["installment"] * term * 12 / income
    features["util_high_flag"] = nullable_flag(data["revolUtil"], data["revolUtil"] > 70)
    features["util_mid_flag"] = nullable_flag(data["revolUtil"], data["revolUtil"].gt(30) & data["revolUtil"].le(70))
    accounts = data["totalAcc"].where(data["totalAcc"] >= 0)
    features["delinq_per_acc"] = data["delinquency_2years"] / (accounts + 1)
    features["has_pub_rec"] = nullable_flag(data["pubRec"], data["pubRec"] > 0)
    features["uncleared_negative"] = data["pubRec"] - data["pubRecBankruptcies"]
    features["fico_low_flag"] = nullable_flag(features["ficoRange_mean"], features["ficoRange_mean"] < 660)
    issue_date = pd.to_datetime(data["issueDate"], errors="coerce")
    first_credit = pd.to_datetime(data["earliesCreditLine"], errors="coerce")
    months = (issue_date.dt.year - first_credit.dt.year) * 12 + issue_date.dt.month - first_credit.dt.month
    features["credit_age_month"] = months.where(months >= 0)
    features["new_credit_flag"] = nullable_flag(features["credit_age_month"], features["credit_age_month"] < 24)
    features["short_emp_flag"] = nullable_flag(data["employmentLength"], data["employmentLength"] < 2)
    features["term_int_burden"] = term * data["interestRate"]
    installment_ratio = data["installment"] / income
    features["high_install_flag"] = nullable_flag(installment_ratio, installment_ratio > 0.05)
    return features.replace([np.inf, -np.inf], np.nan)


class AnonymousFeatureAggregatorV2(AnonymousFeatureAggregator):
    def transform(self, features):
        result = super().transform(features)
        result = result.rename(columns={value: key for key, value in LEGACY_NAMES.items()})
        result["n_valid_count"] = features.loc[:, self.anonymous_variables].gt(0).sum(axis=1).astype("int8")
        result["n_missing_flag"] = result["n_missing_count"].ge(3).astype("int8")
        return result


def feature_checklist(data):
    candidates = AnonymousFeatureAggregatorV2(ANONYMOUS_COLUMNS).transform(build_features_v2(data))
    if candidates.columns.has_duplicates or set(LEGACY_NAMES.values()).intersection(candidates.columns):
        raise ValueError("重复列名或新旧等价匿名计数同时存在")
    if np.isinf(candidates.select_dtypes(include="number").to_numpy(dtype="float64")).any():
        raise ValueError("候选特征包含无穷值")
    return pd.DataFrame([{
        "表格变量": name, "处理方式": status, "实际特征": actual,
        "历史对应名称": LEGACY_NAMES.get(actual, actual) if status == "复用" else None,
        "实际公式": formula, "业务解释与限制": meaning, "构造前填补": False,
        "训练集缺失率": float(candidates[actual].isna().mean()) if actual else np.nan,
    } for name, status, actual, formula, meaning in FEATURE_SPECS])


def check_feature_boundaries(sample):
    fixture = pd.concat([sample.iloc[[0]].copy() for _ in range(8)], ignore_index=True)
    fixture["annualIncome"] = 12000.0
    fixture["installment"] = 600.0
    fixture["loanAmnt"] = 12000.0
    fixture["term"] = 3
    fixture["dti"] = 20.0
    fixture["interestRate"] = 10.0
    fixture["revolUtil"] = [30.0, 70.0, 70.1, 30.1, np.nan, 0.0, 100.0, 40.0]
    fixture["ficoRangeLow"] = [658.0, 657.0, 660.0, np.nan, 658.0, 658.0, 658.0, 658.0]
    fixture["ficoRangeHigh"] = fixture["ficoRangeLow"] + 4
    fixture["employmentLength"] = [2.0, 1.0, np.nan, 0.0, 10.0, 2.0, 2.0, 2.0]
    fixture["issueDate"] = pd.Timestamp("2018-01-01")
    fixture["earliesCreditLine"] = pd.to_datetime([
        "2016-01-01", "2016-02-01", "2015-12-01", "2018-02-01", None,
        "2016-01-01", "2016-01-01", "2016-01-01",
    ])
    fixture.loc[5, "annualIncome"] = 0.0
    fixture.loc[6, "installment"] = 601.0
    fixture.loc[7, "totalAcc"] = 0.0
    fixture.loc[7, "openAcc"] = 0.0
    fixture.loc[7, "delinquency_2years"] = 2.0
    fixture.loc[4, "pubRec"] = np.nan
    fixture[ANONYMOUS_COLUMNS] = 0.0
    fixture.loc[0, ANONYMOUS_COLUMNS] = np.nan
    fixture.loc[1, ANONYMOUS_COLUMNS[:2]] = np.nan
    fixture.loc[2, ANONYMOUS_COLUMNS[:3]] = np.nan
    fixture.loc[3, ["n0", "n1", "n2"]] = [1.0, 0.0, -1.0]
    features = build_features_v2(fixture)
    aggregated = AnonymousFeatureAggregatorV2(ANONYMOUS_COLUMNS).transform(features)
    assert np.isclose(features.loc[0, "new_installment_ratio"], 0.6)
    assert np.isclose(features.loc[0, "total_dti"], 0.8)
    assert np.isclose(features.loc[0, "total_payment_income"], 1.8)
    assert features.loc[0, "loan_to_income"] == 1.0
    assert features.loc[0, "term_int_burden"] == 30.0
    assert features["util_mid_flag"].iloc[:4].tolist() == [0, 1, 0, 1]
    assert features["util_high_flag"].iloc[:4].tolist() == [0, 0, 1, 0]
    assert np.isnan(features.loc[4, "util_mid_flag"])
    assert features["fico_low_flag"].iloc[:3].tolist() == [0, 1, 0]
    assert np.isnan(features.loc[3, "fico_low_flag"])
    assert features["credit_age_month"].iloc[:3].tolist() == [24, 23, 25]
    assert features["new_credit_flag"].iloc[:3].tolist() == [0, 1, 0]
    assert features.loc[[3, 4], "new_credit_flag"].isna().all()
    assert features["short_emp_flag"].iloc[:2].tolist() == [0, 1]
    assert np.isnan(features.loc[2, "short_emp_flag"])
    assert features.loc[[0, 6], "high_install_flag"].tolist() == [0, 1]
    ratio_columns = ["new_installment_ratio", "total_dti", "loan_to_income", "total_payment_income", "high_install_flag"]
    assert features.loc[5, ratio_columns].isna().all()
    assert np.isnan(features.loc[4, "has_pub_rec"])
    assert np.isnan(features.loc[7, "openAcc_ratio"])
    assert features.loc[7, "delinq_per_acc"] == 2.0
    assert aggregated.loc[0, "n_missing_count"] == 15
    assert aggregated["n_missing_flag"].iloc[:3].tolist() == [1, 0, 1]
    assert aggregated.loc[3, "n_valid_count"] == 1
    assert aggregated.loc[0, ["anonymous_sum", "anonymous_mean", "anonymous_std"]].isna().all()
    changed = fixture.assign(isDefault=1 - fixture["isDefault"])
    pd.testing.assert_frame_equal(features, build_features_v2(changed))
    added = [name for name in NEW_FEATURES if name != "n_missing_flag"]
    pd.testing.assert_frame_equal(features.drop(columns=added), build_features(fixture))
    assert len(FEATURE_SPECS) == 24 and len(NEW_FEATURES) == 16
    return {"边界断言": "通过", "业务特征标签无关": True, "旧特征构造值不变": True, "表格特征数": 24, "新增候选数": 16}
