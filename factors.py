import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.config import REG_CN
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA


class FactorsLibrary:

    ##########################日内价格相关因子#########################################
    # 方差
    def real_var(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算方差
        df = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].var().reset_index()
        df = df.rename(columns={'Ret': 'var'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'var']]
        return df

    # 峰度
    def real_kurtosis(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算峰度并直接赋值,计算超额峰度
        df = df.groupby(['Trade_Date', 'Stock_Code'])['Ret'].apply(lambda x: x.kurt() + 3).reset_index()
        df = df.rename(columns={'Ret': 'kurt'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'kurt']]
        return df

    # 偏度
    def real_skew(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算偏度值
        df = df.groupby(['Trade_Date', 'Stock_Code'])['Ret'].skew().reset_index()
        df = df.rename(columns={'Ret': 'skew'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'skew']]
        return df

    # 日内最大回撤
    def intraday_maxdrawdown(self, df_origin):
        df = df_origin.copy(deep=True)

        # 计算日内最大回撤函数
        def max_drawdown(group):
            # 目前累计最大值
            group['cummax'] = group['Close'].cummax()
            group['drawdown'] = 1 - group['Close'].div(group['cummax'])
            max_drawdown = group['drawdown'].max()
            return pd.Series({'maxdrawdown': max_drawdown})

        # 计算最大回撤
        df = df.groupby(['Stock_Code', 'Trade_Date']).apply(max_drawdown).reset_index()
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'maxdrawdown']]
        return df

    # volume-weight average price 成交量加权平均价格
    def vwap(self, df_origin):
        df = df_origin.copy(deep=True)
        # tp为经典价格 即权重 tp=(high+low+close)/3
        df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3

        # 计算每个分组的vwap
        def group_vwap(group):
            return (group['tp'] * group['Volume']).mean()

        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(group_vwap).reset_index()
        df = grouped.rename(columns={0: 'vwap'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'vwap']]
        return df

    # time-weighted average price 时间加权平均价格
    def twap(self, df_origin):
        df = df_origin.copy(deep=True)

        # 计算每个分组的twap
        def group_twap(group):
            num_rows = len(group)
            weights = np.arange(1, num_rows + 1) / num_rows
            return (weights * group['Close']).mean()

        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(group_twap).reset_index()
        df = grouped.rename(columns={0: 'twap'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'twap']]
        return df

    # 上行收益率方差
    def real_upvar(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算方差
        df_up = df[df['Ret'] > 0]
        df_up = df_up.groupby(['Stock_Code', 'Trade_Date'])['Ret'].var().reset_index()
        df_up = df_up.rename(columns={'Ret': 'upvar'})
        df_up = df_up.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df_up = df_up[['Stock_Code', 'Trade_Date', 'upvar']]
        return df_up

    # 上行收益率方差比值
    def ratio_realupvar(self, df_orign):

        df = df_orign.copy(deep=True)

        upvar = self.real_upvar(df)
        upvar = upvar.rename(columns={'Factor_Value': 'upvar'})

        var = self.real_var(df)
        var = var.rename(columns={'Factor_Value': 'var'})

        df = pd.merge(upvar, var, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['Factor_Value'] = df['upvar'].div(df['var'])

        df = df[['Trade_Date', 'Stock_Code', 'Factor_Value']]

        return df

    # 下行收益率方差
    def real_downvar(self, df_orign):
        df = df_origin.copy(deep=True)
        # 计算方差
        df_up = df[df['Ret'] < 0]
        df_up = df_up.groupby(['Stock_Code', 'Trade_Date'])['Ret'].var().reset_index()
        df_up = df_up.rename(columns={'Ret': 'upvar'})
        df_up = df_up.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df_up = df_up[['Stock_Code', 'Trade_Date', 'upvar']]
        return df_up

    # 下行收益率方差比值
    def ratio_realdownvar(self, df_orign):

        df = df_orign.copy(deep=True)

        downvar = self.real_downvar(df)
        downvar = downvar.rename(columns={'Factor_Value': 'downvar'})

        var = self.real_var(df)
        var = var.rename(columns={'Factor_Value': 'var'})

        df = pd.merge(downvar, var, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['Factor_Value'] = df['downvar'].div(df['var'])

        df = df[['Trade_Date', 'Stock_Code', 'Factor_Value']]

        return df

    # 趋势占比，日内价格变化/分钟频价格变化绝对值之和
    def trendratio(self, df_orign):

        df = df_orign.copy(deep=True)

        df = df.sort_values(by=['Stock_Code', 'datetime']).reset_index(drop=True)

        # 分母denominator：close差值的绝对值之和
        df['diff'] = df.groupby(['Stock_Code', 'Trade_Date'])['close'].diff().abs()
        grouped = df.groupby(['Stock_Code', 'Trade_Date'])['diff']
        denom = grouped.apply(np.sum).reset_index()
        denom = denom.rename(columns={'diff': 'diff_sum'})

        # 分子numerator：最后一个close和第一个close的差值
        numer = df.groupby(['Stock_Code', 'Trade_Date'])['close'].apply(lambda x: x.iloc[-1] - x.iloc[0]).reset_index()
        numer = numer.rename(columns={'close': 'close_diff'})

        df = pd.merge(denom, numer, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['Factor_Value'] = df['close_diff'].div(df['diff_sum'])

        df = df[['Trade_Date', 'Stock_Code', 'Factor_Value']]

        df = df.sort_values(by=['Trade_Date', 'Stock_Code'], ascending=[True, True])

        return df

    # 股价当天的真实波幅
    def true_range(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取前一天的收盘价
        df['lClose'] = df.groupby('Stock_Code')['Close'].shift(1)

        # 取各个分组的真是波幅
        def group_range(group):
            # 最高价
            high = group['High'].max()
            # 最低价
            low = group['Low'].min()
            # 前一天的收盘价(取前一天分钟线数据的最后一个非nan值)
            group.dropna()
            lcose = group['lClose'].iloc[-1]
            return max(high - low, abs(high - lcose), abs(low - lcose))

        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(group_range).reset_index()
        df = grouped.rename(columns={0: 'true_range'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'true_range']]
        return df

    # 买气
    def dtm(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取每支股票每天最早非NaN的开盘价
        open = df.groupby(['Stock_Code', 'Trade_Date'])['Open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        # 日内最高价
        high = df.groupby(['Stock_Code', 'Trade_Date'])['High'].apply(
            lambda x: x.dropna().max() if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(open, high, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['open_diff'] = df['Open'] - df.groupby('Stock_Code')['Open'].shift(1)
        df['intra_diff'] = df['High'] - df['Open']
        df['dtm'] = df.apply(lambda x: np.maximum(x['intra_diff'], x['open_diff'])
        if pd.notnull(x['open_diff']) and x['open_diff'] > 0 else 0, axis=1)
        df = df[['Stock_Code', 'Trade_Date', 'dtm']]
        return df

    # 卖气
    def dbm(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取每支股票每天最早非NaN的开盘价
        open = df.groupby(['Stock_Code', 'Trade_Date'])['Open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        # 日内最低价
        low = df.groupby(['Stock_Code', 'Trade_Date'])['Low'].apply(
            lambda x: x.dropna().min() if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(open, low, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['open_diff'] = df['Open'] - df.groupby('Stock_Code')['Open'].shift(1)
        df['intra_diff'] = df['Open'] - df['Low']
        df['dbm'] = df.apply(lambda x: np.maximum(x['intra_diff'], x['open_diff'])
        if pd.notnull(x['open_diff']) and x['open_diff'] < 0 else 0, axis=1)
        df = df[['Stock_Code', 'Trade_Date', 'dbm']]
        return df

    # 最高价差值
    def hd(self, df_origin):
        df = df_origin.copy(deep=True)
        # 日内最高价
        df = df.groupby(['Stock_Code', 'Trade_Date'])['High'].apply(
            lambda x: x.dropna().max() if not x.dropna().empty else np.nan).reset_index()
        df['hd'] = df['High'] - df.groupby('Stock_Code')['High'].shift(1)
        df = df[['Stock_Code', 'Trade_Date', 'hd']]
        return df

    # 最低价差值
    def ld(self, df_origin):
        df = df_origin.copy(deep=True)
        # 日内最高价
        df = df.groupby(['Stock_Code', 'Trade_Date'])['Low'].apply(
            lambda x: x.dropna().min() if not x.dropna().empty else np.nan).reset_index()
        df['ld'] = df['Low'] - df.groupby('Stock_Code')['Low'].shift(1)
        df = df[['Stock_Code', 'Trade_Date', 'ld']]
        return df

    # 日内收益率
    def ret_intraday(self, df_origin):
        df = df_origin.copy(deep=True)
        # 获取每支股票每天的最后一个非NaN的收盘价
        numer = df.groupby(['Stock_Code', 'Trade_Date'])['Close'].apply(
            lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan).reset_index()
        # 获取每支股票每天的第一个非NaN的开盘价
        denom = df.groupby(['Stock_Code', 'Trade_Date'])['Open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(denom, numer, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['ret_intraday'] = df['Close'].div(df['Open']) - 1
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'ret_intraday']]
        return df

    # 尾盘成交量占比
    def ratio_volumeH8(self, df_origin):
        df = df_origin.copy(deep=True)
        # 每支股票每一天的成交量总数
        total_volume = df.groupby(['Stock_Code', 'Trade_Date'])['Volume'].sum().reset_index()
        # 计算尾盘的成交量
        hours_volume = df.groupby(['Trade_Date', 'Stock_Code']).apply(lambda x: x.tail(30)).reset_index(drop=True)
        hours_volume = hours_volume.groupby(['Trade_Date', 'Stock_Code'])['Volume'].sum().reset_index()
        hours_volume = hours_volume.rename(columns={'Volume': 'hour_volume'})
        # 合并
        df = pd.merge(total_volume, hours_volume, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['ratio_volumeH8'] = df['hour_volume'].div(df['Volume'])
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'ratio_volumeH8']]
        return df

    ##########################日内价量相关因子#########################################
    # 价量相关性:分钟成交量与价格相关性
    def corr_VP(self, df_origin):
        df = df_origin.copy(deep=True)

        # 计算相关性（分钟成交量与价格相关性）
        def cal_corr(data):
            # 默认使用皮尔逊相关系数
            return data['Close'].corr(data['Volume'])

        df = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index().rename(columns={0: 'corr_VP'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_VP']]
        return df

    # 量与收益率相关性:分钟成交量与收益率相关性
    def corr_VR(self, df_origin):
        df = df_origin.copy(deep=True)

        # 计算相关性（分钟成交量与价格相关性）
        def cal_corr(data):
            # 默认使用皮尔逊相关系数
            return data['Ret'].corr(data['Volume'])

        df = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index().rename(columns={0: 'corr_VR'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_VR']]
        return df

    # 量与超前收益率相关性: 分钟成交量与下一时刻收益率相关性
    def corr_VRlead(self, df_origin):
        df = df_origin.copy(deep=True)

        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['retlead'].corr(data['Volume'], method='kendall')

        # 获取后一分钟的收益率
        df['retlead'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].shift(-1)
        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: 'corr_VRlead'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_VRlead']]
        return df

    # 量与滞后收益率相关性:分钟成交量与上一时刻收益率相关性
    def corr_VRlag(self, df_origin):
        df = df_origin.copy(deep=True)

        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['retlag'].corr(data['Volume'], method='kendall')

        # 获取前一分钟的收益率
        df['retlag'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].shift(1)
        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: 'corr_VRlag'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_VRlag']]
        return df

    # Amihud 非流动性因子
    def Amihud_illiq(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算分钟的流动性
        df['illiq'] = (abs(df['Ret']) / (df['Close'] * df['Volume'])) * pow(10, 10)
        df = df.groupby(['Trade_Date', 'Stock_Code'])['illiq'].mean().reset_index()
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'illiq']]
        return df

    # 开盘价相对第一阶段集合竞价最高价的收益率（假设开盘前30分钟为第一阶段）
    def ret_open2AH1(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算第一阶段集合竞价时间范围内的数据并计算最高价
        first_auction_max_high = df.groupby(['Stock_Code', 'Trade_Date']).apply(
            lambda x: x.head(30)['High'].max()).reset_index(name='High')
        # 获取每支股票每天的第一个非NaN的开盘价
        denom = df.groupby(['Stock_Code', 'Trade_Date'])['Open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(denom, first_auction_max_high, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['ret_open2AH1'] = (df['Open'] - df['High']) / df['High']
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'ret_open2AH1']]
        return df

    # 开盘价相对第一阶段集合竞价最低价的收益率（假设开盘前30分钟为第一阶段）
    def ret_open2AL1(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算第一阶段集合竞价时间范围内的数据并计算最低价
        first_auction_max_Low = df.groupby(['Stock_Code', 'Trade_Date']).apply(
            lambda x: x.head(30)['Low'].min()).reset_index(name='Low')
        # 获取每支股票每天的第一个非NaN的开盘价
        denom = df.groupby(['Stock_Code', 'Trade_Date'])['Open'].apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()
        df = pd.merge(denom, first_auction_max_Low, on=['Stock_Code', 'Trade_Date'], how='inner')
        df['ret_open2AL1'] = (df['Open'] - df['Low']) / df['Low']
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'ret_open2AL1']]
        return df

    # 收盘前半小时的收益率
    def ret_H8(self, df_origin):
        df = df_origin.copy(deep=True)
        # 取收盘前30分钟的数据
        df = df.groupby(['Trade_Date', 'Stock_Code']).apply(lambda x: x.tail(30)).reset_index(drop=True)
        df = df.groupby(['Stock_Code', 'Trade_Date'])['Close'].agg(['first', 'last']).reset_index()
        df['ret_H8'] = df['last'].div(df['first']) - 1
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'ret_H8']]
        return df

    # 大成交量对应的收益率偏度（分钟成交量排名前1/3的成交量定义为“大成交量”）
    def real_skewlarge(self, df_origin):
        df = df_origin.copy(deep=True)
        # 筛选分钟成交量排名前1/3的成交量
        df = df.groupby(['Stock_Code', 'Trade_Date']).apply(
            lambda x: x.nlargest(int(len(x) / 3), 'Volume')).reset_index(drop=True)
        df['real_skewlarge'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].skew().reset_index(drop=True)
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'real_skewlarge']]
        return df

    # 大成交量对应的 corr_VP
    def corr_VPlarge(self, df_origin):
        df = df_origin.copy(deep=True)
        # 筛选分钟成交量排名前1/3的成交量
        df = df.groupby(['Stock_Code', 'Trade_Date']).apply(
            lambda x: x.nlargest(int(len(x) / 3), 'Volume')).reset_index(drop=True)

        def cal_corr(data):
            # 默认使用皮尔逊相关系数
            return data['Close'].corr(data['Volume'])

        df = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index().rename(columns={0: 'corr_VPlarge'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_VPlarge']]
        return df

    def corr_VRlaglarge(self, df_origin):
        df = df_origin.copy(deep=True)
        # 筛选分钟成交量排名前1/3的成交量
        df = df.groupby(['Stock_Code', 'Trade_Date']).apply(
            lambda x: x.nlargest(int(len(x) / 3), 'Volume')).reset_index(drop=True)

        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['retlag'].corr(data['Volume'], method='kendall')

        # 获取前一分钟的收益率
        df['retlag'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].shift(1)
        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: 'corr_VRlaglarge'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_VRlaglarge']]
        return df

    # 反转因子
    def rev_vol(self, df_origin):
        df = df_origin.copy(deep=True)

        def rev(data):
            data['Close_lead'] = data['Close'].shift(-1)
            data['log'] = np.log(data['Close_lead'] / data['Close'])
            data['log_rev'] = data['log'][::-1]

            return np.sum(data['Volume'] * data['log_rev']) / np.sum(data['Volume'])

        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(rev).reset_index()
        df = grouped.rename(columns={0: 'rev_cov'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'rev_cov']]

        return df

    # 价格与成交笔数的相关性
    def corr_PN(self, df_origin):
        df = df_origin.copy(deep=True)

        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['Close'].corr(data['Num_trade'])

        # 获取前一分钟的收益率
        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: 'corr_PN'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_PN']]
        return df

    # 价格与成交量/成交笔数的相关性
    def corr_PVPN(self, df_origin):
        df = df_origin.copy(deep=True)

        def cal_corr(data):
            # 肯德尔相关系数计算相关性
            return data['Close'].corr(data['div'])

        # 获取前一分钟的收益率
        df['div'] = df['Volume'] / df['Num_trade']
        grouped = df.groupby(['Trade_Date', 'Stock_Code']).apply(cal_corr).reset_index()
        df = grouped.rename(columns={0: 'corr_PVPN'})
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'corr_VPN']]
        return df

    ##########################日间价格相关因子#########################################
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
        res = factor_A_series.expanding().cov(factor_B_series) / factor_B_series.expanding().var()
        start = np.min(n_list) - 1 if np.min(n_list) - 1 > 0 else 0
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

        start = np.min(n_list) - 1 if np.min(n_list) - 1 > 0 else 0
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
    def SMA(self, series_origin, n, m):
        factor_series = series_origin.copy(deep=True)
        alpha = m / n
        pred_series = SimpleExpSmoothing(factor_series). \
            fit(smoothing_level=alpha, optimized=False).predict(0, len(factor_series))
        return pred_series

    # 指数加权移动平均 (WMA)
    # 通过当前的实际值和前一段时期（由 d 约定平均了多少以前的数据），来平滑修改当前的值
    def decay_linear(self, series_origin, d):
        factor_series = series_origin.copy(deep=True)
        weights = np.arange(1, d + 1)
        sum_weights = np.sum(weights)
        pred_series = factor_series.rolling(d).apply(lambda x: np.sum(weights * x) / sum_weights, raw=False)
        return pred_series

    # 量变动速率指标
    # 一般取 n = 6
    def vroc_n(self, df_origin, n=6):
        df = df_origin.copy(deep=True)
        df['vroc_n'] = df.groupby('Stock_Code', group_keys=False).apply(
            lambda x: x.sort_values(by='Trade_Date', ascending=True))['Volume'].pct_change(n)
        df = df[['Stock_Code', 'Trade_Date', 'vroc_n']]
        return df

    # 人气指标
    def ar(self, df_origin, n_days=26):
        df = df_origin.copy(deep=True)
        df = df.groupby('Stock_Code', group_keys=False). \
            apply(lambda x: x.sort_values(by='Trade_Date', ascending=True))
        df['ar'] = None

        def calc_diff(a, b):
            return a - b

        # 对每只股票计算 ar
        grouped = df.groupby('Stock_Code', group_keys=False)
        for _, data in grouped:
            denom = data.apply(lambda x: calc_diff(x['Open'], x['Low']), axis=1). \
                rolling(n_days, closed='left').sum()
            numer = data.apply(lambda x: calc_diff(x['High'], x['Open']), axis=1). \
                rolling(n_days, closed='left').sum()
            df.loc[data.index, 'ar'] = numer / denom * 100
        df = df[['Stock_Code', 'Trade_Date', 'ar']]
        return df

    # 意愿指标
    def br(self, df_origin, n_days=26):
        df = df_origin.copy(deep=True)
        df = df.groupby('Stock_Code', group_keys=False). \
            apply(lambda x: x.sort_values(by='Trade_Date', ascending=True))
        df['br'] = None

        def calc_diff(a, b):
            return a - b

        # 对每只股票计算 br
        grouped = df.groupby('Stock_Code', group_keys=False)
        for _, data in grouped:
            denom = data.apply(lambda x: calc_diff(x['Close'], x['Low']), axis=1). \
                rolling(n_days, closed='left').sum()
            numer = data.apply(lambda x: calc_diff(x['High'], x['Close']), axis=1). \
                rolling(n_days, closed='left').sum()
            df.loc[data.index, 'br'] = numer / denom * 100
        df = df[['Stock_Code', 'Trade_Date', 'br']]
        return df

    # 意愿指标
    def wvad(self, df_origin, n_days=6):
        df = df_origin.copy(deep=True)
        df = df.groupby('Stock_Code', group_keys=False). \
            apply(lambda x: x.sort_values(by='Trade_Date', ascending=True))
        df['wvad'] = None

        def calc_wvad(close, open, low, volume):
            return (close - open) / (close - low) * volume if close - low != 0 else np.NaN

        # 对每只股票计算 wvad
        grouped = df.groupby('Stock_Code', group_keys=False)
        for _, data in grouped:
            res = data.apply(lambda x: calc_wvad(
                x['Close'], x['Open'], x['Low'], x['Volume']), axis=1). \
                rolling(n_days, closed='left').sum()
            df.loc[data.index, 'wvad'] = res
        df = df[['Stock_Code', 'Trade_Date', 'wvad']]
        return df

    # ARIMA因子计算的函数
    def arima(self, df_origin, order=(1, 1, 1)):
        df = df_origin.copy(deep=True)[['Stock_Code', 'Trade_Date', 'Ret']]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(axis=0, how='any')
        def cal_arima_factor(data, order):
            # 确保 Trade_Date 是 datetime 类型并设置为索引
            data = data.set_index(pd.to_datetime(data['Trade_Date']))
            data = data.resample('D').sum(numeric_only=True)
            print(data)
            arima_factors = [pd.NA] * max(order[1], 1)  # 初始化至少有 order[1] 个元素
            for end in range(order[1], len(data)):  # 从能够拟合模型的第一个点开始
                print(data['Ret'].iloc[:end + 1])
                model = ARIMA(data['Ret'].iloc[:end + 1], order=order).fit()  # 使用收益率 Ret 拟合模型
                print(model)


                # 进行预测
                pred = model.predict(exog=data[['Ret']])

                # 打印预测结果的类型
                print(type(pred))

                # 如果是 Series 或 DataFrame，打印形状
                if isinstance(pred, (pd.Series, pd.DataFrame)):
                    print(pred.shape)

                # 打印前几个预测值
                print(pred.head())

                if isinstance(pred, pd.Series):
                    arima_factors.append(pred.iloc[0])
                else:
                    arima_factors.append(pred)
            return pd.Series(arima_factors, index=data.index)
        # 应用ARIMA模型并合并结果
        arima_factors_series = df.groupby('Stock_Code').apply(cal_arima_factor, order=order)
        df['ARIMA_Factor'] = arima_factors_series  # 将ARIMA因子值添加到原始DataFrame
        return df[['Stock_Code', 'Trade_Date', 'ARIMA_Factor']]

if __name__ == '__main__':
    # 初始化 Qlib
    # qlib.init(provider_uri='E:\cn_data_1min', region=REG_CN)
    qlib.init(provider_uri='E:\cn_data_1d', region=REG_CN)
    # 获取市场上所有股票的列表
    stock_list = D.instruments(market='csi500')
    # 定义时间范围
    start_time = "2010-08-25"  # 例如 '2020-01-01'
    end_time = "2020-09-25"  # 例如 '2020-12-31'
    # $开头的是 Qlib 内置的字段，表示股票市场数据
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    # freq = "1min"
    freq = "day"
    # 获取数据
    df_origin = D.features(instruments=stock_list, start_time=start_time, end_time=end_time, freq=freq,
                           fields=fields).reset_index()
    # 修改字段名
    new_column_names = {
        "$open": "Open",
        "$high": "High",
        "$low": "Low",
        "$close": "Close",
        "$volume": "Volume",
        "instrument": "Stock_Code",  # 修改股票代码列的字段名
        "datetime": "Trade_Date"  # 修改时间列的字段名
    }
    df_origin = df_origin.rename(columns=new_column_names)
    df_origin['Trade_Date'] = pd.to_datetime(df_origin['Trade_Date']).dt.strftime('%Y-%m-%d')
    # 先按股票代码排序，然后在每个股票内部按照时间排序
    df_origin = df_origin.sort_values(by=['Stock_Code', 'Trade_Date'])
    # 添加变化率行
    df_origin['Ret'] = df_origin.groupby(['Stock_Code'])['Close'].pct_change(1)

     # 调用因子库
    factor_tool = FactorsLibrary()
    dff = factor_tool.corr_VP(df_origin)
    print(dff)
    # 使用isnull()方法标识空值
    null_values = dff['corr_VP'].isnull()
    # 计算空值的数量
    null_count = null_values.sum()
    # 计算空值的占比
    null_percentage = (null_count / len(dff)) * 100
    print(f"{null_percentage:.2f}%")




