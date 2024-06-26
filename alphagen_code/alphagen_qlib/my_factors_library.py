from scipy import stats
import pandas as pd
import numpy as np
# from database_reader import DatabaseReader
from qlib.data import D
from qlib.config import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from datetime import datetime
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
# from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import *
# from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
# from alphagen.rl.env.wrapper import AlphaEnv
# from alphagen.rl.policy import LSTMSharedNet
# from alphagen.utils.random import reseed_everything
# from alphagen.rl.env.core import AlphaEnvCore


class FactorsLibrary:

    ########################## 日内价格相关因子 ##########################
    # 方差
    def real_var(self, df_origin):
        df = df_origin.copy(deep=False)
        # 计算方差
        df = df.groupby(['instrument', 'datetime'])['ret'].var().reset_index()
        df = df.rename(columns={'ret': '$real_var'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$real_var']]
        return df

    # 峰度
    def real_kurtosis(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算峰度并直接赋值,计算超额峰度
        df = df.groupby(['datetime', 'instrument'])['ret'].apply(lambda x: x.kurt() + 3).reset_index()
        df = df.rename(columns={'ret': '$real_kurtosis'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$real_kurtosis']]
        return df

    # 偏度
    def real_skew(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算偏度值
        df = df.groupby(['datetime', 'instrument'])['ret'].skew().reset_index()
        df = df.rename(columns={'ret': '$real_skew'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$real_skew']]
        return df

    # 日内最大回撤
    def intraday_maxdrawdown(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算日内最大回撤函数
        def max_drawdown(group):
            # 目前累计最大值
            group['cummax'] = group['$close'].cummax()
            group['drawdown'] = 1 - group['$close'].div(group['cummax'])
            max_drawdown = group['drawdown'].max()
            return pd.Series({'$intraday_maxdrawdown': max_drawdown})
        # 计算最大回撤
        df = df.groupby(['instrument', 'datetime']).apply(max_drawdown).reset_index()
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$intraday_maxdrawdown']]
        return df

    # volume-weight average price 成交量加权平均价格
    def vwap(self, df_origin):
        df = df_origin.copy(deep=True)
        # tp为经典价格 即权重 tp=(high+low+close)/3
        df['tp'] = (df['$high'] + df['$low'] + df['$close']) / 3
        # 计算每个分组的vwap
        def group_vwap(group):
            return (group['tp'] * group['$volume']).mean()
        grouped = df.groupby(['datetime', 'instrument']).apply(group_vwap).reset_index()
        df = grouped.rename(columns={0: '$vwap'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$vwap']]
        return df

    # time-weighted average price 时间加权平均价格
    def twap(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算每个分组的twap
        def group_twap(group):
            num_rows = len(group)
            weights = np.arange(1, num_rows + 1) / num_rows
            return (weights * group['$close']).mean()
        grouped = df.groupby(['datetime', 'instrument']).apply(group_twap).reset_index()
        df = grouped.rename(columns={0: '$twap'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$twap']]
        return df

    # 上行收益率方差
    def real_upvar(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算方差
        df_up = df[df['ret']>0]
        df_up = df_up.groupby(['instrument', 'datetime'])['ret'].var().reset_index()
        df_up = df_up.rename(columns={'ret': '$real_upvar'})
        df_up = df_up.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df_up = df_up[['instrument', 'datetime', '$real_upvar']]
        return df_up

    # 股价当天的真实波幅
    def true_range(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取前一天的收盘价
        df['lClose'] = df.groupby('instrument')['$close'].shift(1)
        # 取各个分组的真是波幅
        def group_range(group):
            # 最高价
            high = group['$high'].max()
            # 最低价
            low = group['$low'].min()
            # 前一天的收盘价(取前一天分钟线数据的最后一个非nan值)
            group.dropna()
            lcose = group['lClose'].iloc[-1]
            return max(high - low, abs(high - lcose), abs(low - lcose))
        grouped = df.groupby(['datetime', 'instrument']).apply(group_range).reset_index()
        df = grouped.rename(columns={0: '$true_range'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$true_range']]
        return df

    # 买气
    def dtm(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取每支股票每天最早非NaN的开盘价
        open = df.groupby(['instrument', 'datetime'])['$open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        # 日内最高价
        high = df.groupby(['instrument', 'datetime'])['$high'].apply(
            lambda x: x.dropna().max() if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(open, high, on=['instrument', 'datetime'], how='inner')
        df['open_diff'] = df['$open'] - df.groupby('instrument')['$open'].shift(1)
        df['intra_diff'] = df['$high'] - df['$open']
        df['$dtm'] = df.apply(lambda x: np.maximum(x['intra_diff'], x['open_diff'])
                                if pd.notnull(x['open_diff']) and x['open_diff'] > 0 else 0, axis=1)
        df = df[['instrument', 'datetime', '$dtm']]
        return df

    # 卖气
    def dbm(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取每支股票每天最早非NaN的开盘价
        open = df.groupby(['instrument', 'datetime'])['$open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        # 日内最低价
        low = df.groupby(['instrument', 'datetime'])['$low'].apply(
            lambda x: x.dropna().min() if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(open, low, on=['instrument', 'datetime'], how='inner')
        df['open_diff'] = df['$open'] - df.groupby('instrument')['$open'].shift(1)
        df['intra_diff'] = df['$open'] - df['$low']
        df['$dbm'] = df.apply(lambda x: np.maximum(x['intra_diff'], x['open_diff'])
                                if pd.notnull(x['open_diff']) and x['open_diff'] < 0 else 0, axis=1)
        df = df[['instrument', 'datetime', '$dbm']]
        return df

    # 最高价差值
    def hd(self, df_origin):
        df = df_origin.copy(deep=True)
        # 日内最高价
        df = df.groupby(['instrument', 'datetime'])['$high'].apply(
            lambda x: x.dropna().max() if not x.dropna().empty else np.nan).reset_index()
        df['$hd'] = df['$high'] - df.groupby('instrument')['$high'].shift(1)
        df = df[['instrument', 'datetime', '$hd']]
        return df

    # 最低价差值
    def ld(self, df_origin):
        df = df_origin.copy(deep=True)
        # 日内最高价
        df = df.groupby(['instrument', 'datetime'])['$low'].apply(
            lambda x: x.dropna().min() if not x.dropna().empty else np.nan).reset_index()
        df['$ld'] = df['$low'] - df.groupby('instrument')['$low'].shift(1)
        df = df[['instrument', 'datetime', '$ld']]
        return df

    # 日内收益率
    def ret_intraday(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取每支股票每天的最后一个非NaN的收盘价
        numer = df.groupby(['instrument', 'datetime'])['$close'].apply(
            lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan).reset_index()
        # 获取每支股票每天的第一个非NaN的开盘价
        denom = df.groupby(['instrument', 'datetime'])['$open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(denom, numer, on=['instrument', 'datetime'], how='inner')
        df['$ret_intraday'] = df['$close'].div(df['$open']) - 1
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$ret_intraday']]
        return df

    # 尾盘成交量占比
    def ratio_volumeh8(self, df_origin):
        df = df_origin.copy(deep=True)
        # 每支股票每一天的成交量总数
        total_volume = df.groupby(['instrument', 'datetime'])['$volume'].sum().reset_index()
        # 计算尾盘的成交量
        hours_volume = df.groupby(['datetime', 'instrument']).apply(lambda x: x.tail(30)).reset_index(drop=True)
        hours_volume = hours_volume.groupby(['datetime', 'instrument'])['$volume'].sum().reset_index()
        hours_volume = hours_volume.rename(columns={'$volume': 'hour_volume'})
        # 合并
        df = pd.merge(total_volume, hours_volume, on=['instrument', 'datetime'], how='inner')
        df['$ratio_volumeh8'] = df['hour_volume'].div(df['$volume'])
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$ratio_volumeh8']]
        return df


    ##########################日内价量相关因子#########################################
    # 价量相关性:分钟成交量与价格相关性
    def corr_vp(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算相关性（分钟成交量与价格相关性）
        def cal_corr(data):
            # 默认使用皮尔逊相关系数
            return data['$close'].corr(data['$volume'])
        df = df.groupby(['datetime', 'instrument']).apply(cal_corr).reset_index().rename(columns={0: '$corr_vp'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$corr_vp']]
        return df

    # 量与收益率相关性:分钟成交量与收益率相关性
    def corr_vr(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算相关性（分钟成交量与价格相关性）
        def cal_corr(data):
            # 默认使用皮尔逊相关系数
            return data['ret'].corr(data['$volume'])
        df = df.groupby(['datetime', 'instrument']).apply(cal_corr).reset_index().rename(columns={0: '$corr_vr'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$corr_vr']]
        return df

    # 量与超前收益率相关性: 分钟成交量与下一时刻收益率相关性
    def corr_vrlead(self, df_origin):
        df = df_origin.copy(deep=True)
        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['retlead'].corr(data['$volume'], method='kendall')
        # 获取后一分钟的收益率
        df['retlead'] = df.groupby(['instrument', 'datetime'])['ret'].shift(-1)
        grouped = df.groupby(['datetime', 'instrument']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: '$corr_vrlead'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$corr_vrlead']]
        return df

    # 量与滞后收益率相关性:分钟成交量与上一时刻收益率相关性
    def corr_vrlag(self, df_origin):
        df = df_origin.copy(deep=True)
        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['retlag'].corr(data['$volume'], method='kendall')
        # 获取前一分钟的收益率
        df['retlag'] = df.groupby(['instrument', 'datetime'])['ret'].shift(1)
        grouped = df.groupby(['datetime', 'instrument']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: '$corr_vrlag'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$corr_vrlag']]
        return df

    # Amihud 非流动性因子
    def amihud_illiq(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算分钟的流动性
        df['$amihud_illiq'] = (abs(df['ret']) / (df['$close'] * df['$volume'])) * pow(10, 10)
        df = df.groupby(['datetime', 'instrument'])['$amihud_illiq'].mean().reset_index()
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$amihud_illiq']]
        return df

    # 开盘价相对第一阶段集合竞价最高价的收益率（假设开盘前30分钟为第一阶段）
    def ret_open2ah1(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算第一阶段集合竞价时间范围内的数据并计算最高价
        first_auction_max_high = df.groupby(['instrument', 'datetime']).apply(
            lambda x: x.head(30)['$high'].max()).reset_index(name='$high')
        # 获取每支股票每天的第一个非NaN的开盘价
        denom = df.groupby(['instrument', 'datetime'])['$open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(denom, first_auction_max_high, on=['instrument', 'datetime'], how='inner')
        df['$ret_open2ah1'] = (df['$open'] - df['$high']) / df['$high']
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$ret_open2ah1']]
        return df

    # 开盘价相对第一阶段集合竞价最低价的收益率（假设开盘前30分钟为第一阶段）
    def ret_open2al1(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算第一阶段集合竞价时间范围内的数据并计算最低价
        first_auction_max_Low = df.groupby(['instrument', 'datetime']).apply(
            lambda x: x.head(30)['$low'].min()).reset_index(name='$low')
        # 获取每支股票每天的第一个非NaN的开盘价
        denom = df.groupby(['instrument', 'datetime'])['$open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(denom, first_auction_max_Low, on=['instrument', 'datetime'], how='inner')
        df['$ret_open2al1'] = (df['$open'] - df['$low']) / df['$low']
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$ret_open2al1']]
        return df

    # 收盘前半小时的收益率
    def ret_h8(self, df_origin):
        df = df_origin.copy(deep=True)
        # 取收盘前30分钟的数据
        df = df.groupby(['datetime', 'instrument']).apply(lambda x: x.tail(30)).reset_index(drop=True)
        df = df.groupby(['instrument', 'datetime'])['$close'].agg(['first', 'last']).reset_index()
        df['$ret_h8'] = df['last'].div(df['first']) - 1
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$ret_h8']]
        return df

    # 大成交量对应的收益率偏度（分钟成交量排名前1/3的成交量定义为“大成交量”）
    def real_skewlarge(self, df_origin):
        df = df_origin.copy(deep=True)
        # 筛选分钟成交量排名前1/3的成交量
        df = df.groupby(['instrument', 'datetime']).apply(
            lambda x: x.nlargest(int(len(x) / 3), '$volume')).reset_index(drop=True)
        df['$real_skewlarge'] = df.groupby(['instrument', 'datetime'])['ret'].skew().reset_index(drop=True)
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$real_skewlarge']]
        return df

    # 大成交量对应的 corr_VP
    def corr_vplarge(self, df_origin):
        df = df_origin.copy(deep=True)
        # 筛选分钟成交量排名前1/3的成交量
        df = df.groupby(['instrument', 'datetime']).apply(
            lambda x: x.nlargest(int(len(x) / 3), '$volume')).reset_index(drop=True)
        def cal_corr(data):
            # 默认使用皮尔逊相关系数
            return data['$close'].corr(data['$volume'])
        df = df.groupby(['datetime', 'instrument']).apply(cal_corr).reset_index().rename(columns={0: '$corr_vplarge'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$corr_vplarge']]
        return df

    def corr_vrlaglarge(self, df_origin):
        df = df_origin.copy(deep=True)
        # 筛选分钟成交量排名前1/3的成交量
        df = df.groupby(['instrument', 'datetime']).apply(
            lambda x: x.nlargest(int(len(x) / 3), '$volume')).reset_index(drop=True)
        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['retlag'].corr(data['$volume'], method='kendall')
        # 获取前一分钟的收益率
        df['retlag'] = df.groupby(['instrument', 'datetime'])['ret'].shift(1)
        grouped = df.groupby(['datetime', 'instrument']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: '$corr_vrlaglarge'})
        df = df.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
        df = df[['instrument', 'datetime', '$corr_vrlaglarge']]
        return df


    ########################## 日间价格相关因子 ##########################

    # 向量升序排序
    def rank(self, series_origin):
        factor_series = series_origin.copy(deep=True)
        factor_series = factor_series.rank()
        return factor_series

    # 序列 A 过去 n 天标准差
    def std_n(self, series_origin, n_days):
        factor_series = series_origin.copy(deep=True)
        factor_series = factor_series.rolling(n_days).std()
        return factor_series

    # 序列 A/B 过去 n 天相关系数
    def corr_n(self, series_A_origin, series_B_origin, n_days):
        factor_A_series, factor_B_series = \
            series_A_origin.copy(deep=True), series_B_origin.copy(deep=True)
        res = factor_A_series.rolling(n_days).corr(factor_B_series)
        return res

    # 序列 A n 天的差值
    def delta_n(self, series_origin, n_days):
        factor_series = series_origin.copy(deep=True)
        factor_series = factor_series.diff(n_days)
        return factor_series

    # 序列 A 过去 n 天的最小值
    def tsmin_n(self, series_origin, n_days):
        factor_series = series_origin.copy(deep=True).tail(n_days)
        factor_series = factor_series.rolling(n_days).min()
        return factor_series

    # 序列 A 过去 n 天的最大值
    def tsmax_n(self, series_origin, n_days):
        factor_series = series_origin.copy(deep=True)
        factor_series = factor_series.rolling(n_days).max()
        return factor_series

    # 前 n 期回归系数
    def reg_beta(self, series_A_origin, series_B_origin, n_list):
        factor_A_series, factor_B_series = \
            series_A_origin.copy(deep=True), series_B_origin.copy(deep=True)
        res = factor_A_series.expanding().cov(factor_B_series)/factor_B_series.expanding().var()
        start = np.min(n_list)-1 if np.min(n_list)-1>0 else 0
        end = np.max(n_list)
        res = res[start:end]
        return res

    # 前 n 期回归残差
    def reg_alpha(self, series_A_origin, series_B_origin, n_list):
        factor_A_series, factor_B_series = \
            series_A_origin.copy(deep=True), series_B_origin.copy(deep=True)
        factor_A_mean, factor_B_mean = factor_A_series.expanding().mean(), \
                                       factor_B_series.expanding().mean()
        res = factor_A_mean - \
              self.reg_beta(factor_A_series, factor_B_series, n_list) * factor_B_mean

        start = np.min(n_list)-1 if np.min(n_list)-1>0 else 0
        end = np.max(n_list)
        res = res[start:end]
        return res

    # 序列 A 过去 n 天的累乘
    def prod_n(self, series_origin, n_days):
        factor_series = series_origin.copy(deep=True)
        factor_series = factor_series.rolling(n_days).cumprod(skipna=True)
        return factor_series

    # 一次指数平滑法
    # 通过当前时间的真实值和预测值，来预测下一个时期
    def sma(self, series_origin, n, m):
        factor_series = series_origin.copy(deep=True)
        alpha = m/n
        pred_series = SimpleExpSmoothing(factor_series).\
            fit(smoothing_level=alpha, optimized=False).predict(0, len(factor_series))
        return pred_series

    # 指数加权移动平均 (WMA)
    # 通过当前的实际值和前一段时期（由 d 约定平均了多少以前的数据），来平滑修改当前的值
    def decay_linear(self, series_origin, d):
        factor_series = series_origin.copy(deep=True)
        weights = np.arange(1, d + 1)
        sum_weights = np.sum(weights)
        pred_series = factor_series.rolling(d).apply(lambda x
                                                     : np.sum(weights * x) / sum_weights, raw=False)
        return pred_series

    # 量变动速率指标
    # 一般取 n = 6
    def vroc_n(self, df_origin, n=6):
        df = df_origin.copy(deep=True)
        df['$vroc_n'] = df.groupby('instrument', group_keys=False).apply(
            lambda x: x.sort_values(by='datetime', ascending=True))['$volume'].pct_change(n)
        df = df[['instrument', 'datetime', '$vroc_n']]
        return df

    # 人气指标
    def ar(self, df_origin, n_days=26):
        df = df_origin.copy(deep=False)
        df = df.groupby('instrument', group_keys=False).\
            apply(lambda x: x.sort_values(by='datetime', ascending=True))
        df['$ar'] = None
        def calc_diff(a, b):
            return a-b
        # 对每只股票计算 ar
        grouped = df.groupby('instrument', group_keys=False)
        for _, data in grouped:
            denom = data.apply(lambda x: calc_diff(x['$open'], x['$low']), axis=1).\
                rolling(n_days, closed='left').sum()
            numer = data.apply(lambda x: calc_diff(x['$high'], x['$open']), axis=1).\
                rolling(n_days, closed='left').sum()
            result = numer/denom*100
            df.loc[data.index, '$ar'] = result
        df = df[['instrument', 'datetime', '$ar']]
        return df

    # 意愿指标
    def br(self, df_origin, n_days=26):
        df = df_origin.copy(deep=False)
        df = df.groupby('instrument', group_keys=False). \
            apply(lambda x: x.sort_values(by='datetime', ascending=True))
        df['$br'] = None
        def calc_diff(a, b):
            return a - b
        # 对每只股票计算 br
        grouped = df.groupby('instrument', group_keys=False)
        for _, data in grouped:
            denom = data.apply(lambda x: calc_diff(x['$close'], x['$low']), axis=1). \
                rolling(n_days, closed='left').sum()
            numer = data.apply(lambda x: calc_diff(x['$high'], x['$close']), axis=1). \
                rolling(n_days, closed='left').sum()
            result = numer/denom*100
            df.loc[data.index, '$br'] = result
        df = df[['instrument', 'datetime', '$br']]
        return df

    # 意愿指标
    def wvad(self, df_origin, n_days=26):
        df = df_origin.copy(deep=True)
        df = df.groupby('instrument', group_keys=False). \
            apply(lambda x: x.sort_values(by='datetime', ascending=True))
        df['$wvad'] = None
        def calc_wvad(close, open, low, volume):
            return (close-open)/(close-low)*volume if close-low!=0 else np.NaN
        # 对每只股票计算 wvad
        grouped = df.groupby('instrument', group_keys=False)
        for _, data in grouped:
            res = data.apply(lambda x: calc_wvad(
                x['$close'], x['$open'], x['$low'], x['$volume']), axis=1).\
                rolling(n_days, closed='left').sum()
            df.loc[data.index, '$wvad'] = res
        df = df[['instrument', 'datetime', '$wvad']]
        return df