import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import time
from database_reader import DatabaseReader

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.float_format', '{:,.5f}'.format)  # 设置浮点精度


class FactorBacktesting:

    def __init__(self, start_date, end_date, factor_table, group_counts, freq=['M', 'W', 'D '], post_ipo_days=250,
                 remove_bj=True, drop_na_percent=0.3, stock_universe=['全市场', '沪深300', '中证1000', '中证500']):

        self.start_date = start_date
        self.end_date = end_date
        self.factor_table = factor_table
        self.group_counts = group_counts
        self.freq = freq
        self.post_ipo_days = post_ipo_days
        self.remove_bj = remove_bj
        self.drop_na_percent = drop_na_percent
        self.stock_universe = stock_universe

        # 创建 database_reader 实例
        self.database_reader = DatabaseReader()
        # 文件保存地址
        self.hostname = 'C:/Users/chuti/desktop/'

        if self.stock_universe == "沪深300":
            self.stock_universe_table_name = 'monthlyweights000300sh'
        elif self.stock_universe == "中证500":
            self.stock_universe_table_name = 'monthlyweights000905sh'
        elif self.stock_universe == "中证1000":
            self.stock_universe_table_name = 'monthlyweights000852sh'
        else:
            self.stock_universe_table_name = None

        self.valid_stk_group_daily_rtn_df = None

    def get_stock_list(self):
        """根据频率freq，选出每个周期末尾的有效股票，即剔除当期ST、停牌、IPO上市未满 n 天的股票。"""
        # Get a dataframe with a column named First_Valid_Date = the ipo date + n days of each stock
        # 筛掉IPO上市未满 n 天的股票
        n_days_after_ipo_df = self.database_reader.read_from_db(
            dbname='gogoalest3',
            sql=f'''
                SELECT DISTINCT
                  MIN(trade_date) + INTERVAL {self.post_ipo_days} DAY AS First_Valid_Date,
                  CASE
                    WHEN exchange = '001001'
                    THEN CONCAT(stock_code, '.SH')
                    WHEN exchange = '001002'
                    THEN CONCAT(stock_code, '.SZ')
                    ELSE CONCAT(stock_code, '.BJ')
                  END AS Stock_Code
                FROM
                  qt_stk_daily
                GROUP BY Stock_Code, exchange;
            '''
        )

        valid_stock_df = self.database_reader.read_from_db(
            dbname='gogoalest3',
            sql=f'''
                SELECT DISTINCT
                  trade_date AS Trade_Date,
                  CASE
                    WHEN exchange = '001001'
                    THEN CONCAT(stock_code, '.SH')
                    WHEN exchange = '001002'
                    THEN CONCAT(stock_code, '.SZ')
                    ELSE CONCAT(stock_code, '.BJ')
                  END AS Stock_Code,
                  stock_name AS Stock_Name,
                  volume AS Volume
                FROM
                  qt_stk_daily
                WHERE Trade_Date >= '{self.start_date}'
                  AND Trade_Date <= '{self.end_date}'
                ORDER BY Trade_Date ASC,
                  Stock_Code ASC;
            '''
        )

        # 将交易日期按照频率 freq 进行分组，以找到按照特定频率确定的期末交易日期。
        valid_stock_df['Group_By_Date'] = valid_stock_df['Trade_Date']

        # 返回的是 每个分组内的最后一个交易日期
        period_end_dates = valid_stock_df.groupby(pd.Grouper(
            key='Group_By_Date', freq=self.freq))['Trade_Date'].last()

        # Only keep the stocks that still have daily info on the period-end trade dates
        valid_stock_df = valid_stock_df[valid_stock_df.Trade_Date.isin(period_end_dates)] \
            .drop(columns='Group_By_Date')

        # TODO: inner join with the index component stocks

        # 剔除当期ST、停牌的股票
        valid_stock_df = valid_stock_df[~valid_stock_df.Stock_Name.str.contains(
            'ST') & valid_stock_df.Volume != 0]
        valid_stock_df = valid_stock_df.drop(columns={'Stock_Name', 'Volume'})

        # Get the final valid (Stock_Code, Trade_Date) pairs dataframe
        valid_stock_df = valid_stock_df.merge(
            right=n_days_after_ipo_df, on='Stock_Code', how='left')
        valid_stock_df = valid_stock_df.loc[valid_stock_df.Trade_Date >= valid_stock_df.First_Valid_Date] \
            .drop(columns='First_Valid_Date')

        # 是否剔除北京证券交易所的股票数据
        if self.remove_bj:
            valid_stock_df = valid_stock_df[~valid_stock_df.Stock_Code.str.contains(
                'BJ')]

        # Convert valid_stock_df from a long table to a wide table where the value is True/False representing
        # whether the stock should be present as a valid stock on the trade date
        # Shift the cell values to 1 period later so that the stock validity of current period comes from the previous period
        valid_stock_df['dummy'] = 1
        valid_stock_df = valid_stock_df.pivot_table(values='dummy', index='Trade_Date', columns='Stock_Code') \
            .shift(1)
        stock_code = valid_stock_df.columns
        valid_stock_df = valid_stock_df.reset_index() \
            .melt(id_vars='Trade_Date', value_vars=stock_code)
        valid_stock_df = valid_stock_df.loc[~valid_stock_df.value.isna()].drop(columns={
            'value'})

        # 指数公司每个月更新一次成分股权重。每半年调仓一次，一般是在6月份和12月份。
        if self.stock_universe_table_name is not None:
            # 从已经筛选好的全市场股票中，挑选出属于设定指数的成分股的股票。
            pass
        # 如果每半年调仓一次，那我是不是只要每半年更新一次

        return valid_stock_df

    # 返回的是一个仅包含需要调仓的日期的因子值集合，同时包含各个调仓日期的分组信息
    def get_prev_period_group(self):
        """
        todo: 假如一个因子总是出现很多NA的话，还得另外提示我们一下,讨论之后认为放在厦大的get_group函数中写， 这个函数主要考虑运行速度和效率
        todo: 需要能计算n日的分档和收益，现在只能计算1个自然月、1个自然周的分档和收益
        例：1.31有效股票列表 -> 1.31因子值 -> 1.31排序分组 -> 2.31各组的月收益，前脚
        """
        # 从数据库表中获取数据
        if type(self.factor_table) is str:
            # Get factor dataframe with trade dates between start_date and end_date
            factor_df = self.database_reader.read_from_db(
                'super_symmetry',
                f'''
                    SELECT
                    date AS Trade_Date,
                    code AS Stock_Code,
                    sentiment AS Factor_Value
                    FROM
                    {self.factor_table}
                    WHERE date >= '{self.start_date}'
                    AND date <= '{self.end_date}'
                    ORDER BY date ASC,
                    code ASC;
                '''
            )
        else:
            # Note that the factor dataframe should be the same format as the one read from the SQL query above
            # i.e. it should have columns Trade_Date, Stock_Code, Factor_Value
            factor_df = self.factor_table

        factor_df['Trade_Date'] = pd.to_datetime(
            factor_df['Trade_Date'], format='%Y-%m-%d')
        factor_df['Group_By_Date'] = factor_df['Trade_Date']

        factor_df['Stock_Code'] = factor_df['Stock_Code'].astype(str).apply(lambda x: x.zfill(6))

        print("1.从数据库获取因子得数据")
        print(factor_df)

        # factor_df now have columns Stock_Code, Trade_Date, Factor_Value where Trade_Date and Factor_Value are the period-end values
        # 根据频率分组，获取作为分组标准的数据
        # .agg('last')对每个分组应用last聚合函数，这意味着对于每个组，选择该组中的最后一行数据
        factor_df = factor_df.groupby([pd.Grouper(key='Group_By_Date', freq=self.freq), 'Stock_Code']) \
            .agg('last') \
            .reset_index(level=0, drop=True) \
            .reset_index()

        print("2.根据频率分组,获取每个时间窗口的数据")
        print(factor_df)

        # For each stock, sort the trade dates with ascending order and shift the factor values down 1 row
        # so that the factor values now correspond to the factor values from the previous period
        # 将每支股票在每个交易日期上的因子值调整为前一个交易日期的值
        factor_df['Factor_Value'] = factor_df.sort_values('Trade_Date') \
            .groupby('Stock_Code')['Factor_Value'] \
            .shift(1)

        factor_df = factor_df.rename(
            columns={'Factor_Value': 'Prev_Factor_Value'})

        print("3.将每支股票在每个交易日期上的因子值调整为前一个交易日期的值")
        print(factor_df)

        # 为每支在期末交易日期上有效的股票添加前一个交易周期的因子值
        # 获取调仓日期上的有效股票
        valid_stock_df = self.get_stock_list()
        valid_stock_df['Stock_Code'] = valid_stock_df['Stock_Code'].astype(str).str[:6]

        print("4.获取调仓日期上的有效股票")
        print(valid_stock_df)

        factor_df = factor_df.merge(
            valid_stock_df, on=['Stock_Code', 'Trade_Date'])

        print("5.合并df获取调仓日期上的股票数据以及对应的因子值")
        print(factor_df)

        # 对因子的缺失值进行处理，并根据因子值排序进行分组
        def treat_na_then_rank_group(series):
            """On each trade date, remove or fill the na values based on the drop_na_percent argument
            from the outer function. Then sort the factor value of all valid stocks in ascending order
            and split the stocks into groups accourding to group_counts.

            Note: this function is to be used in pair with the SeriesGroupBy.apply function call below."""
            # 计算缺失情况
            na_percent_in_series = series.isna().sum() / series.size

            # 缺失情况小于设置的缺失率
            if na_percent_in_series < self.drop_na_percent:
                series = series.dropna()       # 删除缺失值
            else:
                # 计算除了 NaN 以外的数据的均值
                non_nan_mean = series[series.notna()].mean()
                # 使用均值进行填充
                series = series.fillna(non_nan_mean)

            # 对传入的series进行排序分组
            return pd.qcut(series.rank(method='first'),
                           q=self.group_counts,
                           labels=range(1, self.group_counts + 1))     # 标签是[1,self.group_counts + 1)

        # Treat na values and split the stocks into groups according to their factor values from the previous period
        # 对各个日期的有效股票按照Trade_Date先进行分组，在组内将 Prev_Factor_Value 列传递给 treat_na_then_rank_group 函数进行处理
        group_series = factor_df.groupby('Trade_Date')['Prev_Factor_Value'] \
            .apply(lambda x: treat_na_then_rank_group(x)) \
            .rename('Group') \
            .reset_index(level=0, drop=True)

        print("6.对因子的缺失值进行处理，对交易日期内的数据根据因子值进行分组，返回的是group_series")
        print(group_series)

        factor_df = factor_df.join(
            group_series, how='inner').reset_index(drop=True)

        print("7.为股票加入分组信息，添加group_series")
        print(factor_df)

        return factor_df

    # 返回带有分组信息的股票列表
    def get_valid_stk_group_daily_rtn_df(self, ):
        if self.valid_stk_group_daily_rtn_df is None:
            # Columns: Trade_Date (consecutive),Stock_Code, Daily_Return
            self.valid_stk_group_daily_rtn_df = self.database_reader.read_from_db(
                "gogoalest3",
                f"""
                SELECT DISTINCT
                  CASE
                    WHEN exchange = '001001'
                    THEN CONCAT(stock_code, '.SH')
                    WHEN exchange = '001002'
                    THEN CONCAT(stock_code, '.SZ')
                    ELSE CONCAT(stock_code, '.BJ')
                  END AS Stock_Code,
                  trade_date AS Trade_Date,
                  change_rate / 100 AS Daily_Return
                FROM
                  qt_stk_daily
                WHERE Stock_Code IS NOT NULL
                  AND Trade_Date >= '{self.start_date}'
                  AND Trade_Date <= '{self.end_date}'
                  AND change_rate IS NOT NULL
                ORDER BY Trade_Date ASC,
                  Stock_Code ASC;
                """
            )

            # Add one column called Period, which represents the period that the corresponding row belongs to
            # 增加一列'Period'，用于表示该日期的数据属于哪个时间的分组
            self.valid_stk_group_daily_rtn_df['Trade_Date'] = pd.to_datetime(
                self.valid_stk_group_daily_rtn_df['Trade_Date'], format='%Y-%m-%d')
            self.valid_stk_group_daily_rtn_df['Period'] = self.valid_stk_group_daily_rtn_df['Trade_Date'].dt.to_period(
                freq=self.freq)

            print("一、从qt_stk_daily获取了从开始到结束得时间得数据，并增加了一列Period")
            print(self.valid_stk_group_daily_rtn_df)

            # Column: Period_End_Date (period-end trade dates), Stock_Code, Group
            # 获取分组信息
            stk_group_df = self.get_prev_period_group().rename(
                columns={'Trade_Date': 'Period_End_Date'})

            print("返回的是一个仅包含需要调仓的日期的因子值集合，同时包含各个调仓日期的分组信息")
            print(stk_group_df)

            # Add one column called Period same as above
            stk_group_df['Period'] = stk_group_df['Period_End_Date'].dt.to_period(
                freq=self.freq)

            self.valid_stk_group_daily_rtn_df['Stock_Code'] = self.valid_stk_group_daily_rtn_df['Stock_Code'].astype(str).str[:6]

            # Merge the 2 dataframes on Stock_Code and Period,
            # now each row of stock daily return info has a corresponding period and the group in that period
            self.valid_stk_group_daily_rtn_df = self.valid_stk_group_daily_rtn_df.merge(right=stk_group_df,
                                                                                        how='left',
                                                                                        on=['Stock_Code', 'Period'])

            # Drop the invalid stocks, i.e. the ones without an assigned Group
            # 来过滤掉 Group 列中为空的行
            self.valid_stk_group_daily_rtn_df = self.valid_stk_group_daily_rtn_df.loc[
                ~self.valid_stk_group_daily_rtn_df.Group.isna()]

            self.valid_stk_group_daily_rtn_df = self.valid_stk_group_daily_rtn_df.drop(columns={'Period'}) \
                .sort_values(['Stock_Code', 'Trade_Date']) \
                .reset_index(drop=True)

        return self.valid_stk_group_daily_rtn_df

    def plot(self, year_interested=None, month_interested=None):

        # 设置感兴趣的月份和年份
        start_datetime = pd.to_datetime(self.start_date, format='%Y-%m-%d')
        if year_interested == start_datetime.year and month_interested == start_datetime.month:
            raise ValueError(
                'year_interested和month_interested应该至少比start_date晚一个月')

        # 返回带有分组信息的股票列表完整信息
        if self.valid_stk_group_daily_rtn_df is None:
            self.valid_stk_group_daily_rtn_df = self.get_valid_stk_group_daily_rtn_df()

        group_daily_rtn_df = self.valid_stk_group_daily_rtn_df.copy(deep=True)

        # 筛选股票的市场池子，直接选择按月频进行调仓
        if self.stock_universe_table_name is not None:
            # 筛选出对于股票池的股票列表
            df_stok = self.database_reader.read_from_db(
                'stockmultifactor',
                f'''
                    SELECT Stock_Code,Trade_Date
                    FROM {self.stock_universe_table_name}
                    WHERE Trade_Date >= '{self.start_date}'
                    AND Trade_Date <= '{self.end_date}'
                    ORDER BY Trade_Date ASC,
                    Stock_Code ASC;
                '''
            )
            # 将Trade_Date列转换为日期类型
            group_daily_rtn_df['Trade_Date'] = pd.to_datetime(group_daily_rtn_df['Trade_Date'])
            df_stok['Trade_Date'] = pd.to_datetime(df_stok['Trade_Date'])
            # 提取每个月的日期
            monthly_dates = df_stok['Trade_Date'].dt.to_period('M')
            # 根据股票列表和日期筛选数据
            group_daily_rtn_df = group_daily_rtn_df[group_daily_rtn_df['Stock_Code'].isin(df_stok['Stock_Code']) &
                                                    group_daily_rtn_df['Trade_Date'].dt.to_period('M').isin(monthly_dates)]
            # 按照交易日期分组，并计算每组的行数
            grouped_df = group_daily_rtn_df.groupby('Trade_Date').size().reset_index(name='RowCount')
            # 输出每组的行数
            print(grouped_df)

        # Compute the IC values
        # spearmanr()用于计算秩相关系数
        rank_ic = group_daily_rtn_df.groupby('Trade_Date').apply(lambda df: stats.spearmanr(df['Daily_Return'], df['Prev_Factor_Value'])[0])
        normal_ic = group_daily_rtn_df.groupby('Trade_Date').apply(lambda df: stats.pearsonr(df['Daily_Return'], df['Prev_Factor_Value'])[0])
        # Create DataFrames for the computed IC values
        normal_ic_df = pd.DataFrame(normal_ic, columns=['Normal_IC'])
        rank_ic_df = pd.DataFrame(rank_ic, columns=['Rank_IC'])
        # Add Trade_Date column to the IC DataFrames
        normal_ic_df['Trade_Date'] = normal_ic_df.index
        rank_ic_df['Trade_Date'] = rank_ic_df.index
        # Concatenate the IC DataFrames
        ic_df = pd.concat(
            [
                normal_ic_df,
                rank_ic_df,
                normal_ic_df['Normal_IC'].cumsum().rename('Cum_Normal_IC'),
                rank_ic_df['Rank_IC'].cumsum().rename('Cum_Rank_IC')
            ],
            axis=1
        ).reset_index(drop=True)

        # Check for duplicated columns and drop if necessary
        if ic_df.columns.duplicated().any():
            ic_df = ic_df.loc[:, ~ic_df.columns.duplicated()]

        # 计算累计收益率
        group_daily_rtn_df['Daily_Return'] += 1
        group_daily_rtn_df['Daily_Return'] = group_daily_rtn_df.groupby(
            ['Group', 'Stock_Code', 'Period_End_Date'])['Daily_Return'].cumprod()  # 分组内每只股票的每日收益率的累计值

        print("累计收益率之后是否有空值")
        print(group_daily_rtn_df['Daily_Return'].isna().any())

        print(group_daily_rtn_df.groupby(['Group', 'Period_End_Date', 'Trade_Date'])['Daily_Return'].count())

        group_daily_rtn_df = group_daily_rtn_df.groupby(
            ['Group', 'Period_End_Date', 'Trade_Date'], as_index=False)['Daily_Return'].mean()  # 组内的每日收益的累计值取平均值


        group_period_rtn_df = group_daily_rtn_df[group_daily_rtn_df.Period_End_Date ==
                                                 group_daily_rtn_df.Trade_Date].copy(deep=True)    # 创建新的df存取获取交易期末的数据
        group_period_rtn_df['Daily_Return'] -= 1     # 还原为收益率

        prev_period_rtn = group_daily_rtn_df[group_daily_rtn_df.Period_End_Date ==
                                             group_daily_rtn_df.Trade_Date]
        prev_period_rtn = prev_period_rtn.drop(columns='Trade_Date')
        prev_period_rtn = prev_period_rtn.rename(
            columns={'Daily_Return': 'Prev_Period_Return'})
        prev_period_rtn['Prev_Period_Return'] = prev_period_rtn.groupby('Group', sort=False)[
            'Prev_Period_Return'].cumprod()
        prev_period_rtn['Prev_Period_Return'] = prev_period_rtn.groupby('Group', sort=False)[
            'Prev_Period_Return'].shift(1, fill_value=1)     #  将'Prev_Period_Return'列向上平移一行，用前一周期的累积收益率填充首行

        group_daily_rtn_df = group_daily_rtn_df.merge(prev_period_rtn,
                                                      on=['Group',
                                                          'Period_End_Date'],
                                                      how='left')

        group_daily_rtn_df['Daily_Return'] = group_daily_rtn_df['Daily_Return'] * \
            group_daily_rtn_df['Prev_Period_Return'] - 1
        group_daily_rtn_df = group_daily_rtn_df.drop(
            columns='Prev_Period_Return')

        ############################################################################################

        # 画图

        fig, axes = plt.subplots(figsize=(32, 16), nrows=2, ncols=2)

        # Plot 1: 各组历史累计收益曲线图
        # x-axis: Trade_Date, y-axis: compound return from the start date, legend: Group
        group_daily_rtn_df = group_daily_rtn_df.sort_values(by=['Group', 'Trade_Date'])
        print(group_daily_rtn_df)

        sns.lineplot(data=group_daily_rtn_df, x='Trade_Date',
                     y='Daily_Return', hue='Group', lw=1.5, ax=axes[0, 0], palette='tab20')
        axes[0, 0].set_xlabel('日期', labelpad=15)
        axes[0, 0].set_ylabel('累计收益', labelpad=15)
        axes[0, 0].set_title('各组累计收益')
        # 显示图例
        axes[0, 0].legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1), handlelength=2)

        # TODO: 加一条有指数累积复合收益的曲线，这个比较是直接和指数比呢？还是要挑出指数的成分股来配比？
        # 因为有可能ST和上市未满1年的股票已经被我们的策略剔除但是指数中仍然包含这些股票？
        # TODO: 需要加一个相应的stock_universe_return_df收录相应的指数日收益

        # Plot 2: 因子值分组后的各组周期平均收益柱状图
        # x-axis: Group, y-axis: mean of period-end compound return
        if year_interested is not None and month_interested is not None:
            group_period_rtn_df = group_period_rtn_df.loc[
                (group_period_rtn_df.Period_End_Date.dt.year == year_interested) &
                (group_period_rtn_df.Period_End_Date.dt.month == month_interested)
            ]
            title_prefix = f'{year_interested}年{month_interested}月'
        elif year_interested is not None:
            group_period_rtn_df = group_period_rtn_df.loc[
                group_period_rtn_df.Period_End_Date.dt.year == year_interested]
            title_prefix = f'{year_interested}年'
        else:
            title_prefix = f'{self.start_date}到{self.end_date}之间'

        # Group by Group to get the aggregation of the mean of all period-end compound return for each group
        group_period_rtn_df = group_period_rtn_df.groupby('Group')[['Daily_Return']] \
            .agg('mean') \
            .reset_index()

        sns.barplot(data=group_period_rtn_df, x='Group',
                    y='Daily_Return', ax=axes[1, 0])
        axes[1, 0].set_xlabel('组', labelpad=15)
        axes[1, 0].set_ylabel('周期平均收益', labelpad=15)
        axes[1, 0].set_title(title_prefix + '根据因子值分组后的各组股票周期平均收益')

        # Plot 3, 4: 累计IC曲线
        dates = ic_df['Trade_Date']

        axes[0, 1].bar(dates, ic_df['Normal_IC'])
        axes[0, 1].plot(dates, ic_df['Cum_Normal_IC'])
        axes[0, 1].set_title(
            f'累计Normal IC曲线（全周期IC均值{ic_df.Normal_IC.mean():.4f}）')

        axes[1, 1].bar(dates, ic_df['Rank_IC'])
        axes[1, 1].plot(dates, ic_df['Cum_Rank_IC'])
        axes[1, 1].set_title(f'累计Rank IC曲线（全周期IC均值{ic_df.Rank_IC.mean():.4f}）')

        if type(self.factor_table) is str:
            # 添加总标题
            plt.suptitle(f'{self.freq}_{self.factor_table}_{self.stock_universe}因子', fontsize=28)
            # 保存图片
            plt.savefig(self.hostname + f'{self.factor_table}_{self.stock_universe}因子.png')
        else:
            return plt

        # TODO: 我们一般常用的就是三个指数,000300/000852/000905，对应的是沪深300/中证1000/中证500
        # todo: monthlyweights000300, etc. datasets in stockmultifactor, 上面的表是指数公司每个月末出来的成分股权重数据
        # todo: 在hthink_royalflush上面有个tiny_database你可以看看, 这个表每一列都是单独一个因子值，可以测试一下这些因子在全市场、沪深300成分股、中证500成分股、中证1000成分股上的表现

        # TODO: Always sort values by period end date, trade date , stock code
        # todo Always reset index if a function is returning a data frame in the end
        # todo Remove the sort option in all groupby function calls
        # todo Change all pivot_table to groupby to save time
        # todo Try to avoid using apply function, try with transform, etc. with the numba engine
        # todo Use multiprocessing
        # todo maybe can use pandas rolling for computing cumprod of return


if __name__ == '__main__':
    start = time.time()
    backtest = FactorBacktesting(start_date='2018-01-31',
                                  end_date='2023-01-31',
                                  factor_table='xueqiu_hour',
                                  group_counts=10,
                                  freq='M',
                                  post_ipo_days=250,
                                  remove_bj=True,
                                  drop_na_percent=0.3,
                                  stock_universe='沪深300')
    backtest.plot()
    end = time.time()
    print('time taken to read data is ' + str(end - start))
