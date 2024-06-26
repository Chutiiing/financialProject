import numpy as np
import pandas as pd
from factors import FactorsLibrary
import qlib
from qlib.data import D
from qlib.config import REG_CN
from tqdm import tqdm
import xgboost as xgb
from scipy.stats import spearmanr


class pop:
    # 合并多个股票因子，因子值预处理
    def merge_factors_days(self, df_origin, factor_list, n_days, freq):
        #合并因子集合
        factor_lib = FactorsLibrary()

        df_merged = df_origin[['Stock_Code', 'Trade_Date', 'Ret']]

        # df_merged = df_origin.copy(deep=True)

        if(freq == 'day'):
            for i in np.arange(0, len(factor_list)):
                df_tmp = getattr(factor_lib, factor_list[i])(df_origin, n_days)
                df_merged = pd.merge(df_tmp, df_merged, on=['Stock_Code', 'Trade_Date'], how='inner')
        if(freq == 'min'):
            for i in np.arange(0, len(factor_list)):
                df_tmp = getattr(factor_lib, factor_list[i])(df_origin)
                df_merged = pd.merge(df_tmp, df_merged, on=['Stock_Code', 'Trade_Date'], how='inner')



        print("合并因子未进行处理")
        print(df_merged)

        # 对每一天的数据根据因子值进行排序并归一化到0-1区间
        def normalize(df):
            df_notnan = df.dropna(axis=0, how='any')
            return (df_notnan.rank(method='min') - 1) / (df_notnan.count() - 1)

        df_ranked = df_merged.groupby('Trade_Date', group_keys=False).apply(
            lambda df: df.assign(**{
                col: normalize(df[col])
                for col in df.columns if col not in ['Stock_Code', 'Trade_Date', 'Ret']
            })
        )
        df_ranked = df_ranked.reset_index(drop=True)

        # 对收益率处理(对收益率进行排序,设置标签）
        df_ranked['Next_Ret'] = df_ranked.groupby(['Stock_Code'])['Ret'].shift(-1)   # 下一期的收益率
        df_ranked = df_ranked.dropna(subset=['Next_Ret'])  # 去除收益率为空的行
        df_ranked['return_percentile'] = df_ranked.groupby('Trade_Date')['Next_Ret'] \
            .rank(pct=True, ascending=False)
        df_ranked['label'] = -1  # 初始化标签列
        df_ranked.loc[df_ranked['return_percentile'] <= 0.3, 'label'] = 1  # 前30%的股票标为1
        df_ranked.loc[df_ranked['return_percentile'] >= 0.7, 'label'] = 0  # 后30%的股票标为0

        df_ranked = df_ranked.drop(['Next_Ret' ,'Ret'], axis=1)
        # df_ranked = df_ranked.drop('Next_Ret', axis=1)

        print("处理合并的因子")
        print(df_ranked)

        return df_ranked


    # 划分数据集
    def prepare_dataset(self, df, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df['Trade_Date'] = pd.to_datetime(df['Trade_Date'])
        # 创建布尔索引掩码
        mask = (df['Trade_Date'] >= start_date) & (df['Trade_Date'] <= end_date)
        return df.loc[mask]

    # 初次训练
    def regress_days_train(self, df_train):
        # 数据准备
        # df_pure = df_train.drop(['Next_Ret', 'return_percentile'], axis=1)
        df_pure = df_train
        df_pure.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_pure = df_pure.dropna(axis=0, how='any')
        print("训练的数据是什么样子？")
        # 提取特征和标签
        X_train = df_pure.drop(['Stock_Code', 'Trade_Date', 'label', 'return_percentile'], axis=1)
        y_train = df_pure['return_percentile']

        # 初始化模型

        #LightGBM
        # # 创建数据集
        # lgb_train = lgb.Dataset(X_train, y_train)
        #
        # # 设置参数
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'regression',
        #     'metric': {'l2', 'l1'},
        #     'num_leaves': 31,
        #     'learning_rate': 0.05,
        #     'feature_fraction': 0.9,
        #     'bagging_fraction': 0.8,
        #     'bagging_freq': 5,
        #     'verbose': 0
        # }
        #
        # # 训练模型
        # model = lgb.train(params,
        #                 lgb_train,
        #                 num_boost_round=100)

        #XBoost
        # 创建数据集
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # 设置参数
        params = {
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 31,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9
        }

        # 训练模型
        model = xgb.train(params, dtrain, num_boost_round=100)



        return model

    def regress_days_test(self, df_test, model):

        #获取测试的起始时间和结束时间
        start_date = df_test['Trade_Date'].min()
        end_date = df_test['Trade_Date'].max()

        # LightBGM
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'regression',
        #     'metric': {'l2', 'l1'},
        #     'num_leaves': 31,
        #     'learning_rate': 0.05,
        #     'feature_fraction': 0.9,
        #     'bagging_fraction': 0.8,
        #     'bagging_freq': 5,
        #     'verbose': 0
        # }

        # XBoost
        params = {
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 31,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9
        }

        # 结果显示
        accuracy_scores = []
        percent = []
        rankic = []

        # 滚动预测和增量更新模型
        for current_date in tqdm(pd.date_range(start=start_date + pd.Timedelta(days=1), end=end_date)):
            # 准备当前日期的数据
            current_day_data = df_merged[df_merged['Trade_Date'] == current_date.strftime('%Y-%m-%d')]
            # le = LabelEncoder()

            # 数据不为空
            if not current_day_data.empty:
                df_pure = current_day_data.copy()
                df_pure.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_pure = df_pure.dropna(axis=0, how='any')

                print("预测集的原始数据是什么样子")
                print(df_pure)

                # 提取测试集特征
                X_current = df_pure.drop(['Stock_Code', 'Trade_Date', 'return_percentile', 'label'], axis=1)
                y_true = df_pure['return_percentile']

                # LightBGM
                # 获得预测类别的概率（认为预测的正类概率排序在前30% 并且正类的概率超过80%的为预测标签为1的）
                # y_proba = model.predict(X_current)

                # XBoost
                # 将测试集特征转换为 DMatrix 格式
                dtest = xgb.DMatrix(X_current)
                # 获得预测类别的概率
                y_proba = model.predict(dtest)

                print("y_prob")
                print(y_proba)
                df_pure.loc[:, 'prob_positive'] = y_proba   # 获取正样本的概率

                # 计算IC
                # 计算两个列向量的排名
                rank_x = np.argsort(y_proba)
                rank_y = np.argsort(y_true)

                # 使用斯皮尔曼相关系数计算RankIC
                rank_ic, _ = spearmanr(rank_x, rank_y)
                print("???")
                print(rank_ic)
                rankic.append((current_date, rank_ic))

                # 计算预测的准确度
                df_pure_sorted = df_pure.sort_values(by='prob_positive', ascending=True)
                top_30_percent_index = int(len(df_pure_sorted) * 0.3)         # 按概率排序取前30%的数据，和样本对其
                print(top_30_percent_index)
                df_top = df_pure_sorted.head(top_30_percent_index)
                y_pred = df_top[df_top['prob_positive'] < 0.5]    # 取概率大于80%的为1，看原本标签是否为1
                print("y_pred")
                print(y_pred)
                high_prob_percentage = (len(y_pred) / len(df_pure) / 0.3)     # 符合条件的占总数的多少
                percent.append((current_date, high_prob_percentage))

                accuracy = y_pred['label'].value_counts(normalize=True).get(1, 0)    # 预测的准确度
                accuracy_scores.append((current_date, accuracy))

                # 处理预测数据进行增量学习
                df_retrain = df_pure[df_pure['label'] != -1]
                X_add = df_retrain.drop(['Stock_Code', 'Trade_Date', 'return_percentile', 'label', 'prob_positive'], axis=1)
                y_add = df_retrain['return_percentile']

                # LightBGM
                # lgb_new = lgb.Dataset(X_add, y_add)
                # # 使用新数据更新模型
                # model = lgb.train(params,
                #                 lgb_new,
                #                 num_boost_round=10,  # 可能需要更少的轮次来进行微调
                #                 init_model=model)  # 从已有模型继续训练

                # # XBoost
                # dadd = xgb.DMatrix(X_add, label=y_add)
                # # 使用新数据更新模型
                # model = xgb.train(params, dadd, num_boost_round=10, xgb_model=model)  # 指定xgb_model以从已有模型继续训练

        # # 解包日期和准确率
        # dates, accuracies = zip(*accuracy_scores)
        # print(accuracies)
        # # 绘制准确率随时间的变化
        # plt.figure(figsize=(10, 5))
        # plt.plot(dates, accuracies, marker='o')
        # plt.title('Model Accuracy Over Time')
        # plt.xlabel('Date')
        # plt.ylabel('Accuracy')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
        #
        # # 解包日期和占比
        # dates2, percents = zip(*percent)
        # print(percents)
        # # 绘制准确率随时间的变化
        # plt.figure(figsize=(10, 5))
        # plt.plot(dates2, percents, marker='o')
        # plt.title('percent')
        # plt.xlabel('Date')
        # plt.ylabel('percent')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
        print("!!!")
        dates3, ri = zip(*rankic)
        print(ri)

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
    df_origin = D.features(instruments=stock_list, start_time=start_time, end_time=end_time, freq=freq, fields=fields).reset_index()
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
    # adaboost分类
    pop = pop()

    # 合并因子
    # df_merged = pop.merge_factors_days(df_origin,
    #                                         factor_list=['corr_VP', 'corr_VR', 'corr_VRlead', 'corr_VRlag',
    #                                                      'mihud_illiq', 'ret_open2AH1', 'ret_open2AL1', 'ret_H8',
    #                                                      'real_skewlarge', 'corr_VPlarge', 'corr_VRlaglarge'],
    #                                         n_days=3,
    #                                         freq='min')

    df_merged = pop.merge_factors_days(df_origin,
                                            factor_list=['ar', 'br', 'wvad', 'vroc_n'],
                                            n_days=3,
                                            freq='day')

    df_merged['Trade_Date'] = pd.to_datetime(df_merged['Trade_Date']).dt.strftime('%Y-%m-%d')

    # 训练的数据只保留前30%和后30%
    df_merged_train = df_merged[df_merged['label'] != -1]

    # train
    df_train = pop.prepare_dataset(df_merged_train, "2010-08-25", "2020-08-25")
    model = pop.regress_days_train(df_train)
    # test
    df_test = pop.prepare_dataset(df_merged, "2020-08-25", "2020-09-02")
    pop.regress_days_test(df_test,model)