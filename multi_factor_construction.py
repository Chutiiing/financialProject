import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from database_reader import DatabaseReader
from factors import FactorsLibrary
from pyswarm import pso


# 调用因子库
factor_tool = FactorsLibrary()
# 数据库读写工具
database_reader = DatabaseReader()
# 创建示例数据框，进行数据预处理
data = database_reader.read_from_db(
    dbname="mikuang_new",
    sql=f'''
        SELECT 
            order_book_id AS Stock_Code,
            datetime AS Trade_Date,
            close AS Close
        FROM
            mikuang_202303
    '''
)
df_origin = pd.DataFrame(data)
# 将 'Trade_Date' 列转换为日期时间格式，同时转化为年月日格式
df_origin['Trade_Date'] = pd.to_datetime(df_origin['Trade_Date']).dt.strftime('%Y-%m-%d')
# 先按股票代码排序，然后在每个股票内部按照时间排序
df_origin = df_origin.sort_values(by=['Stock_Code', 'Trade_Date'])
# 添加变化率行
df_origin['Ret'] = df_origin.groupby(['Stock_Code'])['Close'].pct_change(1)

# 因子值计算
df_var = factor_tool.real_var(df_origin)
df_skew = factor_tool.real_skew(df_origin)
df_kurt = factor_tool.real_kurtosis(df_origin)
df_upvar = factor_tool.real_upvar(df_origin)
df_maxdrawdown = factor_tool.intraday_maxdrawdown(df_origin)


# 计算每天的收益率
df_daily = df_origin.groupby(['Stock_Code', 'Trade_Date']).last().reset_index()
df_daily = df_daily.sort_values(by=['Stock_Code', 'Trade_Date'])
df_daily['Daily_Ret'] = df_daily.groupby('Stock_Code')['Close'].pct_change(1)

# 因子列表
factor_dfs = [df_var, df_skew, df_kurt, df_upvar, df_maxdrawdown]
factors = ['var', 'skew', 'kurt', 'upvar', 'maxdrawdown']

# 初始化合并后的 DataFrame 为第一个因子 DataFrame
merged_df = factor_dfs[0]
# 逐个将其他因子 DataFrame 合并进来
for df in factor_dfs[1:]:
    merged_df = pd.merge(merged_df, df, on=['Trade_Date', 'Stock_Code'])
# 合并因子数据和未来收益
df = pd.merge(merged_df, df_daily, on=['Trade_Date', 'Stock_Code'])
# 清除nan和inf数据
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=factors + ['Daily_Ret'])

# 计算每个因子的 Rank IC以做评估
factor_rank_ics = {}
for factor in factors:
    grouped = df.groupby('Trade_Date').apply(
        lambda x: spearmanr(x[factor], x['Daily_Ret'])[0] if x[factor].nunique() > 1 and x['Daily_Ret'].nunique() > 1 else np.nan
    )
    rank_ic = grouped.mean()
    factor_rank_ics[factor] = rank_ic

print("Rank ICs for each factor:", factor_rank_ics)

"""


'''
    使用pca+pso算法实现权重的优化
'''

# 定义计算 Rank IC 的函数
def calculate_rank_ic(weights, df, factors):
    def rank_ic_for_group(x):
        if x['combined_factor'].nunique() > 1 and x['Daily_Ret'].nunique() > 1:
            return spearmanr(x['combined_factor'], x['Daily_Ret'])[0]
        else:
            return np.nan

    df['combined_factor'] = df[factors].mul(weights).sum(axis=1)
    rank_ic = df.groupby('Trade_Date').apply(rank_ic_for_group).mean()
    return abs(rank_ic)

# 标准化因子数据
scaler = StandardScaler()
factors_scaled = scaler.fit_transform(df[factors])

# 应用 PCA
pca = PCA(n_components=0.95)  # 选择足够的成分以解释 95% 的方差
factors_pca = pca.fit_transform(factors_scaled)

# 定义优化目标函数
def objective(weights):
    return -calculate_rank_ic(weights, df, factors)  # 最小化负 Rank IC

# 权重界限
lb = np.array([0] * pca.n_components_)  # 下限
ub = np.array([1] * pca.n_components_)  # 上限

# 使用粒子群优化
optimal_weights, _ = pso(objective, lb, ub, swarmsize=30, maxiter=100)

# 应用最优权重并计算 Rank IC
df['optimal_combined_factor'] = np.dot(factors_pca, optimal_weights)
optimal_rank_ic = calculate_rank_ic(optimal_weights, df, factors)

print('Optimal Weights:', optimal_weights)
print('Optimal Combined Factor Rank IC:', optimal_rank_ic)
print(df[['Stock_Code', 'Trade_Date','optimal_combined_factor']])

"""

'''
    使用Adaboost算法
'''

# 标准化因子数据
scaler = StandardScaler()
factors_scaled = scaler.fit_transform(df[factors])

# 定义函数来将因子值转换为排名并归一化
def rank_and_normalize_factors(factors):
    ranked_factors = np.argsort(factors, axis=0).argsort() + 1
    normalized_factors = ranked_factors / (len(factors) + 1)
    return normalized_factors

# 构建特征矩阵
normalized_factors = np.apply_along_axis(rank_and_normalize_factors, axis=0, arr=factors_scaled)

# 改进版Adaboost算法
def custom_adaboost_improved(df, normalized_factors, factors, Q, L=10):
    N = len(df)
    D = np.ones(N) / N  # 初始化样本权重
    alpha = []  # 存储每轮的信心度
    selected_factors = []  # 存储选定的因子和分组
    factor_weights = {factor: 0 for factor in factors}  # 初始化因子权重

    # 假设正收益标记为1，非正收益（包括负收益和零收益）标记为-1
    df['y'] = np.where(df['Daily_Ret'] > 0, 1, -1)

    for l in tqdm(range(L), desc='Adaboost Progress'):
        best_epsilon = float('inf')
        best_factor_index = None
        best_group = None

        for i, factor in enumerate(factors):
            # 对归一化的因子值进行分组
            thresholds = np.quantile(normalized_factors[:, i], np.linspace(0, 1, Q + 1)[1:-1])
            groups = np.digitize(normalized_factors[:, i], thresholds)

            for q in range(Q):
                group_mask = groups == q
                W_plus = np.sum(D[group_mask] * (df['y'][group_mask] == 1))
                W_minus = np.sum(D[group_mask] * (df['y'][group_mask] == -1))
                epsilon = min(W_plus, W_minus)

                if epsilon < best_epsilon:
                    best_epsilon = epsilon
                    best_factor_index = i
                    best_group = q

        # 计算信心度
        alpha_l = 0.5 * np.log((W_plus + 1e-10) / (W_minus + 1e-10))
        alpha.append(alpha_l)
        selected_factors.append((factors[best_factor_index], best_group))
        factor_weights[factors[best_factor_index]] += alpha_l  # 累计因子权重

        # 更新样本权重
        for i in range(N):
            group_mask = groups[i] == best_group
            D[i] *= np.exp(-alpha_l * df['y'].iloc[i] * group_mask)
        D /= np.sum(D)  # 归一化

    # 构建强分类器
    def strong_classifier(x):
        final_prediction = 0
        for factor_name, group, alpha_l in zip(*zip(*selected_factors), alpha):
            factor_index = factors.index(factor_name)
            group_mask = np.digitize(x[factor_index], thresholds) == group
            final_prediction += alpha_l * group_mask
        return np.sign(final_prediction)

    # 应用强分类器并计算rankIC
    df['Predicted_Signal'] = np.apply_along_axis(strong_classifier, axis=1, arr=normalized_factors)
    rank_ic = df.groupby('Trade_Date').apply(lambda x: x[['Predicted_Signal', 'y']].corr().iloc[0, 1]).mean()

    return selected_factors, alpha, factor_weights, rank_ic

# 使用示例
selected_factors, alpha, factor_weights, rank_ic = custom_adaboost_improved(df, normalized_factors, factors, Q=5, L=10)

# 输出因子权重和优化后的rankic值
print("Factor Weights:", factor_weights)
print("rank_ic:", rank_ic)
