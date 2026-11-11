import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def build_features(data):
    features = data.drop(columns="isDefault", errors="ignore").copy()
    income = features["annualIncome"].where(features["annualIncome"] > 0)
    term = features["term"].where(features["term"] > 0)
    total_accounts = features["totalAcc"].where(features["totalAcc"] > 0)
    features["income_nonpositive"] = (features["annualIncome"] <= 0).astype("int8")
    features["annual_loan_income_ratio"] = features["loanAmnt"] / term / income
    features["annual_revolBal_income_ratio"] = features["revolBal"] / income
    features["openAcc_ratio"] = features["openAcc"] / total_accounts
    closed_accounts = features["totalAcc"] - features["openAcc"]
    features["closeAcc"] = closed_accounts.where(closed_accounts >= 0)
    credit_years = features["issueYear"] - features["earliesCreditYear"]
    features["dis_Year"] = credit_years.where(credit_years >= 0)
    features["ficoRange_mean"] = (features["ficoRangeHigh"] + features["ficoRangeLow"]) / 2
    features = features.drop(columns=[
        "id", "issueDate", "issueYear", "earliesCreditLine",
        "ficoRangeHigh", "ficoRangeLow", "policyCode",
    ])
    return features.replace([np.inf, -np.inf], np.nan)

## 优化1：新增匿名变量的最大值、最小值、平均值、中位数、标准差、非零计数、缺失计数等特征
class AnonymousFeatureAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, anonymous_variables):
        self.anonymous_variables = anonymous_variables

    def fit(self, features, target=None):
        return self

    def transform(self, features):
        result = features.copy()
        anonymous = features.loc[:, self.anonymous_variables]
        result["anonymous_sum"] = anonymous.sum(axis=1, min_count=1)
        result["anonymous_mean"] = anonymous.mean(axis=1)
        result["anonymous_median"] = anonymous.median(axis=1)
        result["anonymous_min"] = anonymous.min(axis=1)
        result["anonymous_max"] = anonymous.max(axis=1)
        result["anonymous_std"] = anonymous.std(axis=1, ddof=0)
        result["anonymous_nonzero_count"] = (
            anonymous.notna() & anonymous.ne(0)
        ).sum(axis=1).astype("int8")
        result["anonymous_missing_count"] = anonymous.isna().sum(axis=1).astype("int8")
        return result


class MissingFlagImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_rate_threshold=0.05):
        self.missing_rate_threshold = missing_rate_threshold

    def fit(self, features, target=None):
        self.fill_mode_ = features.mode().iloc[0]
        self.flag_cols_ = features.columns[features.isna().mean() > self.missing_rate_threshold]
        return self

    def transform(self, features):
        result = features.copy()
        for column in self.flag_cols_:
            result[f"{column}_missing_flag"] = result[column].isna().astype("int8")
        return result.fillna(self.fill_mode_)


class CrossFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, continuous_variable, categorical_variable):
        self.continuous_variable = continuous_variable
        self.categorical_variable = categorical_variable

    def fit(self, features, target=None):
        self.group_maps_ = {}
        for category in self.categorical_variable:
            statistics = features.groupby(category, observed=True)[self.continuous_variable].agg(
                ["max", "min", "median", "mean"]
            )
            statistics.columns = [
                f"{category}_{feature}_{statistic}" for feature, statistic in statistics.columns
            ]
            self.group_maps_[category] = statistics
        return self

    def transform(self, features):
        result = features.copy()
        for category, statistics in self.group_maps_.items():
            result = result.join(statistics, on=category, how="left")
        return result


class CrossEntropy2DTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_pairs, cont_vars, use_y_entropy=True):
        self.cat_pairs = cat_pairs
        self.cont_vars = cont_vars
        self.use_y_entropy = use_y_entropy

    def fit(self, features, target=None):
        frame = features.copy()
        if self.use_y_entropy:
            frame["__target__"] = np.asarray(target)
        self.group_maps_ = {}
        for first, second in self.cat_pairs:
            groups = frame.groupby([first, second], observed=True)
            statistics = groups[self.cont_vars].agg(["max", "min", "median", "mean"])
            statistics.columns = [
                f"{first}_{second}_{feature}_{statistic}"
                for feature, statistic in statistics.columns
            ]
            if self.use_y_entropy:
                bad_rate = groups["__target__"].mean()
                proportions = pd.concat([bad_rate, 1 - bad_rate], axis=1)
                entropy = -(proportions * np.log2(proportions.where(proportions > 0))).sum(axis=1)
                statistics[f"{first}_{second}_y_entropy"] = entropy
            self.group_maps_[(first, second)] = statistics
        return self

    def transform(self, features):
        result = features.copy()
        for categories, statistics in self.group_maps_.items():
            result = result.join(statistics, on=list(categories), how="left")
        return result


class IVFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, iv_threshold=0.02, n_bins=10):
        self.iv_threshold = iv_threshold
        self.n_bins = n_bins

    @staticmethod
    def calculate_iv(values, target, n_bins):
        frame = pd.DataFrame({"value": values, "target": target}).dropna()
        if frame["value"].nunique() <= 1:
            return 0.0
        frame["bin"] = frame["value"]
        if frame["value"].nunique() > n_bins:
            frame["bin"] = pd.qcut(frame["value"], q=n_bins, duplicates="drop")
        groups = frame.groupby("bin", observed=True)["target"].agg(["count", "sum"])
        good = (groups["count"] - groups["sum"]).replace(0, 0.5)
        bad = groups["sum"].replace(0, 0.5)
        good_share, bad_share = good / good.sum(), bad / bad.sum()
        return ((good_share - bad_share) * np.log(good_share / bad_share)).sum()

    def fit(self, features, target):
        scores = {
            column: self.calculate_iv(features[column], target, self.n_bins)
            for column in features.columns
        }
        self.iv_result_ = pd.DataFrame(scores.items(), columns=["feature", "IV"])
        self.iv_result_ = self.iv_result_.sort_values("IV", ascending=False)
        self.selected_cols_ = self.iv_result_.loc[
            self.iv_result_["IV"] >= self.iv_threshold, "feature"
        ].tolist()
        return self

    def transform(self, features):
        return features.loc[:, self.selected_cols_]


def calculate_vif(features):
    values = features.astype("float64").fillna(features.median())
    standard_deviation = values.std(ddof=0)
    varying = (standard_deviation > 1e-12) & standard_deviation.notna()
    standardized = (values.loc[:, varying] - values.loc[:, varying].mean()) / standard_deviation[varying]
    correlation = standardized.T @ standardized / len(standardized)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    positive = eigenvalues > 1e-10
    vif = pd.Series(np.nan, index=features.columns, name="VIF")
    vif.loc[varying] = (eigenvectors[:, positive] ** 2 / eigenvalues[positive]).sum(axis=1)
    dependent = (eigenvectors[:, ~positive] ** 2).sum(axis=1) > 1e-8
    vif.loc[vif.index[varying][dependent]] = np.inf
    return vif


def select_features(features, iv_scores, sample_size, seed, correlation_threshold, vif_threshold):
    rank = pd.DataFrame({
        "IV": iv_scores.reindex(features.columns),
        "missing_rate": features.isna().mean(),
        "unique_count": features.nunique(dropna=True),
    }).sort_values(["IV", "missing_rate"], ascending=[False, True], kind="stable")
    sample = features.sample(n=min(sample_size, len(features)), random_state=seed)
    correlation = sample.corr(method="spearman", min_periods=min(100, len(sample)))
    selected = []
    for column in rank.index:
        if rank.loc[column, "unique_count"] > 1 and not correlation.loc[column, selected].abs().ge(correlation_threshold).any():
            selected.append(column)
    while len(selected) > 1:
        vif = calculate_vif(sample[selected])
        high_vif = vif[vif > vif_threshold]
        if high_vif.empty:
            break
        drop_rank = pd.DataFrame({"VIF": high_vif, "IV": iv_scores.reindex(high_vif.index)})
        drop_column = drop_rank.sort_values(
            ["VIF", "IV"], ascending=[False, True], kind="stable"
        ).index[0]
        selected.remove(drop_column)
    return selected
