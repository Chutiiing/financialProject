from DatabaseReader import DatabaseReader
import pandas as pd
import openpyxl

class Stock_Picking:

    def __init__(self):
        # 创建 DatabaseReader 实例
        self.database_reader = DatabaseReader()
        self.hostname = 'C:/Users/chuti/desktop/'  # 文件保存地址

    # report_year: 报告年份
    # tacp: 总市值限定为多少以内（单位：万元）
    # cash:货币资金高于多少（单位：元）
    # Adjusted_Net_Profit_2022: 扣非净利润超过多少（单位：元）
    # Net_Profit_2023: 净利润的值（判断是否盈利)
    # GoodWill: 合并商誉的值小于多少（单位：元）
    def stock_picking(self, report_year=None, tacp=None, cash=None, Adjusted_Net_Profit_2022=None, Net_Profit_2023=None, GoodWill=None):

        # 获取最近的交易日期
        date = self.database_reader.read_from_db(
            "gogoalest3",
            f"""
                SELECT MAX(trade_date) AS max_date FROM qt_stk_daily;
            """
        )
        max_date_str = date['max_date'].dt.strftime('%Y-%m-%d').iloc[0]
        print("交易日期：" + max_date_str)

        # 筛选出为沪深主板的数据
        data = self.database_reader.read_from_db(
            "gogoalest3",
            f"""
                SELECT stock_code,stock_name,exchange,tcap
                FROM qt_stk_daily
                WHERE Trade_Date = '{max_date_str}'
                AND exchange IN ('001001','001002')
            """
        )
        stock_data = pd.DataFrame(data)
        print("筛选出沪深主板：")
        print(stock_data)

        # 获取货币资金 （fin_balance_sheet_gen表中）
        # 选取自资产负债表
        data_cash = self.database_reader.read_from_db(
            "gogoalest3",
            f"""
                    SELECT stock_code,ta_ca1
                    FROM fin_balance_sheet_gen
                    WHERE report_year = {report_year} 
                    AND report_quarter = 1003
                    AND report_type = 1001
            """
        )
        df_cash = pd.DataFrame(data_cash).rename(columns={'ta_ca1': 'cash'})
        stock_data = pd.merge(stock_data, df_cash, on='stock_code', how='inner')
        print("获取货币资金:")
        print(data_cash)

        # 计算2022年扣非净利润,获取自利润表，取2022年的年报
        # 利润表的数据是按照季度来递增的， 例如三季报的净利润=前三季度的净利润总和
        df_net_profit_2022 = pd.read_excel('Net_Profit_2022.xlsx')
        df_net_profit_2022_result = df_net_profit_2022.iloc[:, [0, -4]].rename(
            columns={df_net_profit_2022.columns[0]: 'stock_code', df_net_profit_2022.columns[-4]: 'Adjusted_Net_Profit_2022'})
        df_net_profit_2022_result['stock_code'] = df_net_profit_2022_result['stock_code'].str[:6]
        stock_data = pd.merge(stock_data, df_net_profit_2022_result, on='stock_code', how='inner')

        # 判断2023年1-9月是否盈利，即取三个季度的数据
        # 在fin_income_gen表中获取净利润 字段为np
        # 利润表的数据是按照季度来递增的，获取三季度报表
        data_net_profit_2023 = self.database_reader.read_from_db(
            "gogoalest3",
            f"""
                SELECT stock_code,np
                FROM fin_income_gen
                WHERE report_year = {report_year}
                AND report_quarter = 1003
                AND report_type = 1001
            """
        )
        df_net_profit_2023 = pd.DataFrame(data_net_profit_2023).rename(columns={'np': 'Net_Profit_2023'})
        stock_data = pd.merge(stock_data, df_net_profit_2023, on='stock_code', how='inner')

        # 计算合并报表商誉（商誉选取在资产负债表，取距离当前日期最新的报告期里面的数据）
        # 读取商誉表的数据，获取股票代码和商誉
        df_goodwill = pd.read_excel('GoodWill.xlsx')
        df_goodwill_result = df_goodwill.iloc[:, [0, -1]].rename(columns={df_goodwill.columns[0]: 'stock_code', df_goodwill.columns[-1]: 'Goodwill'})
        df_goodwill_result['stock_code'] = df_goodwill_result['stock_code'].str[:6]
        stock_data = pd.merge(stock_data, df_goodwill_result, on='stock_code', how='inner')

        # 筛选条件
        condition = (
                (stock_data['tcap'] <= tacp) &
                (stock_data['cash'] >= cash) &
                (stock_data['Adjusted_Net_Profit_2022'] > Adjusted_Net_Profit_2022) &
                (stock_data['Net_Profit_2023'] > Net_Profit_2023) &
                (stock_data['Goodwill'] < GoodWill)
        )

        # 根据条件筛选股票
        selected_stocks = stock_data[condition]

        # 输出筛选结果
        result = selected_stocks[['stock_code', 'stock_name', 'exchange', 'tcap', 'cash', 'Adjusted_Net_Profit_2022', 'Net_Profit_2023', 'Goodwill']]
        # 保存csv
        result.to_csv(self.hostname + '选股结果.csv', index=True, encoding='gb2312')


# 测试选股结果
test = Stock_Picking()
test.stock_picking(2023, 300000, 1e8, 10000e4, 0, 5000e4)
