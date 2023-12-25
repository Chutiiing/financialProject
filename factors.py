from scipy import stats
import datetime
import pandas as pd
import numpy as np
from database_reader import DatabaseReader


class FactorsLibrary:

    # 方差
    def real_var(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算方差
        df['var'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].transform(lambda x: x.var(skipna=True))
        # 删除重复行
        df = df.drop_duplicates(subset=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'var']]
        return df
    # 偏度
    def real_skew(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算偏度值
        df['skew'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].transform(lambda x: x.skew(skipna=True))
        # 删除重复行
        df = df.drop_duplicates(subset=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'skew']]
        return df

    # 峰度
    def real_kurtosis(self, df_origin):
        df = df_origin.copy(deep=True)
        # 计算峰度并直接赋值,计算超额峰度
        df['kurt'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].transform(lambda x: x.kurt(skipna=True))
        # 删除重复行
        df = df.drop_duplicates(subset=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'kurt']]
        return df

    def real_upvar(self, df_origin):
        df = df_origin.copy(deep=True)
        # 仅考虑收益率大于0
        df = df[df['Ret'] > 0]
        # 计算方差
        df['upvar'] = df.groupby(['Stock_Code', 'Trade_Date'])['Ret'].transform(lambda x: x.var(skipna=True))
        # 删除重复行
        df = df.drop_duplicates(subset=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df.sort_values(by=['Stock_Code', 'Trade_Date']).reset_index(drop=True)
        df = df[['Stock_Code', 'Trade_Date', 'upvar']]
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


if __name__ == '__main__':
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
                mikuang_202203
        '''
    )
    df_origin = pd.DataFrame(data)
    # 将 'Trade_Date' 列转换为日期时间格式，同时转化为年月日格式
    df_origin['Trade_Date'] = pd.to_datetime(df_origin['Trade_Date']).dt.strftime('%Y-%m-%d')
    # 先按股票代码排序，然后在每个股票内部按照时间排序
    df_origin = df_origin.sort_values(by=['Stock_Code', 'Trade_Date'])
    # 添加变化率行
    df_origin['Ret'] = df_origin.groupby(['Stock_Code'])['Close'].pct_change(1)

    df_skew = factor_tool.real_skew(df_origin)
    print(df_skew)
    df_kurt = factor_tool.real_kurtosis(df_origin)
    print(df_kurt)
    df_maxdrawdown = factor_tool.intraday_maxdrawdown(df_origin)
    print(df_maxdrawdown)
