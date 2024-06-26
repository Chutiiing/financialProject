from scipy import stats
import datetime
import pandas as pd
import numpy as np
# from database_reader import DatabaseReader
import qlib
from qlib.data import D
from qlib.config import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import json
import os
from typing import Optional, Tuple
from datetime import datetime
import fire
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.my_factors_library import FactorsLibrary
from alphagen_qlib.my_stock_data import MyStockData



class Adaboost:
    def merge_factors_days(self, df_origin, factor_list, n_days):
        factor_lib = FactorsLibrary()
        df_merged = df_origin[['Stock_Code', 'Trade_Date', 'Close']]
        for i in np.arange(0, len(factor_list)):
            df_tmp = getattr(factor_lib, factor_list[i])(df_origin, n_days)
            df_merged = pd.merge(df_tmp, df_merged, on=['Stock_Code', 'Trade_Date'], how='inner')
        return df_merged

    def regress_days(self, df_origin, factor_list, n_days):
        df_merged = self.merge_factors_days(df_origin, factor_list, n_days)
        df_pure = df_merged.drop(['Stock_Code', 'Trade_Date'], axis=1)
        df_pure.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_pure = df_pure.dropna(axis=0, how='any')
        X = df_pure.drop(['Close'], axis=1)
        y = df_pure['Close']
        regressor = AdaBoostRegressor(n_estimators=300, random_state=np.random.RandomState(0))
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        print(regressor.score(X, y))
        plt.figure()
        plt.scatter(np.arange(0, len(y)), y, c="k", s=15, label="Training Samples")
        plt.scatter(np.arange(0, len(y)), y_pred, c="r", s=15, alpha=0.5, label="Adaboost")  # 红色：AdaBoost回归
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Boosted Decision Tree Regression")
        plt.legend()
        plt.show()
        return

    def merge_factors_1min(self, df_origin, factor_list):
        factor_lib = FactorsLibrary()
        df_merged = df_origin[['Stock_Code', 'Trade_Date', 'Close']]
        for i in np.arange(0, len(factor_list)):
            df_tmp = getattr(factor_lib, factor_list[i])(df_origin)
            df_merged = pd.merge(df_tmp, df_merged, on=['Stock_Code', 'Trade_Date'], how='inner')
        return df_merged

    def regress_1min(self, df_origin, factor_list):
        df_merged = self.merge_factors_1min(df_origin, factor_list)
        df_pure = df_merged.drop(['Stock_Code', 'Trade_Date'], axis=1)
        df_pure.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_pure = df_pure.dropna(axis=0, how='any')
        X = df_pure.drop(['Close'], axis=1)
        y = df_pure['Close']
        regressor = AdaBoostRegressor(n_estimators=300, random_state=np.random.RandomState(0))
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        plt.figure()
        plt.scatter(np.arange(0, len(y)), y, c="dimgray", marker='o', s=15, label="training samples")  # 黑色：训练集
        plt.scatter(np.arange(0, len(y)), y_pred, c="darkred", marker='o', s=15, label="Adaboost")  # 红色：AdaBoost回归
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Boosted Decision Tree Regression")
        plt.legend()
        plt.show()
        return


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 valid_calculator: AlphaCalculator,
                 test_calculator: AlphaCalculator,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)
        self.logger.record('test/ic', ic_test)
        self.logger.record('test/rank_ic', rank_ic_test)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000
):
    reseed_everything(seed)

    device = torch.device('cuda:0')
    close = Feature(MyFeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    # data_train = MyStockData(instrument=instruments,
    #                        start_time='2018-01-01',
    #                        end_time='2018-01-31', 
    #                        freq='1min')
    data_train = MyStockData(instrument=instruments,
                        start_time='2021-04-12',
                        end_time='2021-04-16', 
                        freq='1min')
    # data_valid = MyStockData(instrument=instruments,
    #                        start_time='2019-01-01',
    #                        end_time='2019-01-31')
    print("- TRAINED -")
    data_valid = MyStockData(instrument=instruments,
                           start_time='2021-04-17',
                           end_time='2021-04-21', 
                           freq='1min')
    print("- VALID -")
    data_test = MyStockData(instrument=instruments,
                          start_time='2021-04-23',
                          end_time='2021-04-27',
                          freq='1min')
    print("- TESTED -")
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3
    )
    
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    name_prefix = f"new_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./path/for/checkpoints',
        valid_calculator=calculator_valid,
        test_calculator=calculator_test,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=16,
        tensorboard_log='./path/for/tb/log',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )


def fire_helper(
    seed: Union[int, Tuple[int]],
    code: str,
    pool: int,
    step: int = None
):
    if isinstance(seed, int):
        seed = (seed, )
    default_steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for _seed in seed:
        main(_seed,
             code,
             pool,
             default_steps[int(pool)] if step is None else int(step)
             )


if __name__ == '__main__':
    fire.Fire(fire_helper)

    ############################################

    # # # 获取每日级别数据
    # qlib.init(provider_uri='C:\\Users\\14782\\.qlib\qlib_data\cn_data', region=REG_CN)
    # freq = "day"

    # # 获取市场上所有股票的列表
    # stock_list = D.instruments(market='all')
    # # 定义时间范围
    # start_time = "2020-09-15"  # 例如 '2020-01-01'
    # end_time = "2020-09-18"  # 例如 '2020-12-31'
    # # $开头的是 Qlib 内置的字段，表示股票市场数据
    # fields = ["$open", "$high", "$low", "$close", "$volume"]

    # df_origin = D.features(instruments=stock_list, start_time=start_time, end_time=end_time, freq=freq, fields=fields).reset_index()
    # # 修改字段名
    # new_column_names = {
    #     "$open": "Open",
    #     "$high": "High",
    #     "$low": "Low",
    #     "$close": "Close",
    #     "$volume": "Volume",
    #     "instrument": "Stock_Code",  # 修改股票代码列的字段名
    #     "datetime": "Trade_Date"  # 修改时间列的字段名
    # }

    # df_origin = df_origin.rename(columns=new_column_names)
    # df_origin['Trade_Date'] = pd.to_datetime(df_origin['Trade_Date']).dt.strftime('%Y-%m-%d')
    # # 先按股票代码排序，然后在每个股票内部按照时间排序
    # df_origin = df_origin.sort_values(by=['Stock_Code', 'Trade_Date'])
    # # 添加变化率行
    # df_origin['Ret'] = df_origin.groupby(['Stock_Code'])['Close'].pct_change(1)
    # # 调用因子库
    # # factor_tool = FactorsLibrary()
    # # dff = factor_tool.intraday_maxdrawdown(df_origin)
    # # dff = factor_tool.rank(df_origin['Open'])
    # # dff = factor_tool.rank(df_origin['Open'])
    # # dff = factor_tool.vroc_n(df_origin, 3)
    # # dff = factor_tool.ar(df_origin, 3)
    # # dff = factor_tool.br(df_origin, 3)
    # # dff = factor_tool.wvad(df_origin, 3)
    # # print(dff.tail(20))
    # adaboost = Adaboost()

    # # # 日间价格因子
    # end_time = "2020-09-18"
    # adaboost.regress_days(df_origin=df_origin, factor_list=['ar', 'br', 'wvad', 'vroc_n'], n_days=3)

    # # # 日内价格相关因子
    # # end_time = "2020-09-17"
    # # adaboost.regress_1min(df_origin=df_origin, factor_list=['real_var', 'real_kurtosis', 'real_skew',
    # #                                                         'intraday_maxdrawdown', 'vwap', 'twap',
    # #                                                         'real_upvar', 'true_range', 'dtm', 'dbm',
    # #                                                         'hd', 'ld', 'ret_intraday', 'ratio_volumeH8'])

    # # # 日内价量相关因子
    # # end_time = "2020-09-16"
    # # adaboost.regress_1min(df_origin=df_origin, factor_list=['corr_VP', 'corr_VR', 'corr_VRlead',
    # #                                                         'corr_VRlag', 'Amihud_illiq',
    # #                                                         'ret_open2AH1', 'ret_open2AL1', 'ret_H8',
    # #                                                         'real_skewlarge', 'corr_VPlarge', 'corr_VRlaglarge'])

    # # # 使用isnull()方法标识空值
    # # null_values = dff['maxdrawdown'].isnull()
    # # # 计算空值的数量
    # # null_count = null_values.sum()
    # # # 计算空值的占比
    # # null_percentage = (null_count / len(dff)) * 100
    # # print(f"{null_percentage:.2f}%")

