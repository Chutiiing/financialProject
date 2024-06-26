from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch
from alphagen_qlib.my_factors_library import FactorsLibrary


class MyFeatureType(IntEnum):
    #################### 基础因子 ######################
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4

    #################### Minutes Factors ######################
    VWAP = 5
    REAL_VAR = 6
    REAL_KURTOSIS = 7
    REAL_SKEW = 8
    INTRADAY_MAXDRAWDOWN = 9
    TWAP = 10
    REAL_UPVAR = 11
    TRUE_RANGE = 12
    DTM = 13
    DBM = 14
    HD = 15
    LD = 16
    RET_INTRADAY = 17
    RATIO_VOLUMEH8 = 18
    CORR_VP = 19
    CORR_VR = 20
    CORR_VRlag = 21
    CORR_VRlead = 22
    AMIHUD_ILLIQ = 23

    #################### Daily Factors ######################
    # VROC_N = 24
    # AR = 25
    # BR = 26
    # WVAD = 27


class MyStockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 freq: str = 'day',
                 max_backtrack_days: int = 20,
                 max_future_days: int = 10,
                 features: Optional[List[MyFeatureType]] = None,
                 device: torch.device = torch.device('cuda:0')) -> None:
        self._init_qlib()

        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(MyFeatureType)
        self.device = device
        self.freq = freq
        self.data, self._dates, self._stock_ids = self._get_data()

    @classmethod
    def _init_qlib(cls) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN
        # qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_rolling", region=REG_CN)
        qlib.init(provider_uri="/root/autodl-tmp/.qlib/qlib_data/cn_data_1min", region=REG_CN)

        cls._qlib_initialized = True

    # Merge Daily Factors
    def merge_factors_days(self, df_origin, factor_list):
        factor_lib = FactorsLibrary()
        df_merged = df_origin.copy(deep=False)
        for i in np.arange(0, len(factor_list)):
            factor_name = str(factor_list[i]).lstrip("$")
            df_tmp = getattr(factor_lib, factor_name)(df_origin)
            df_merged = pd.merge(df_tmp, df_merged, on=['instrument', 'datetime'], how='inner')
        return df_merged
    
    # Merge Minutes Factors
    def merge_factors_1min(self, df_origin, factor_list):
        factor_lib = FactorsLibrary()
        df_merged = df_origin.copy(deep=False)
        for i in np.arange(0, len(factor_list)):
            factor_name = str(factor_list[i]).lstrip("$")
            df_tmp = getattr(factor_lib, factor_name)(df_origin)
            df_merged = pd.merge(df_tmp, df_merged, on=['instrument', 'datetime'], how='inner')
        print(df_merged)
        return df_merged

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        # Use Basic Exprs For QlibDataLoader
        basic_exprs = ['$open', '$close', '$high', '$low', '$volume']
        original_exprs = list(set(exprs).intersection(basic_exprs))
        additional_exprs = list(set(exprs)-set(basic_exprs))

        cal: np.ndarray = D.calendar()
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        real_start_time = cal[start_index - self.max_backtrack_days]
        if cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        real_end_time = cal[end_index + self.max_future_days]

        df_origin = QlibDataLoader(config=original_exprs,
                                    freq=self.freq).load(self._instrument, real_start_time, real_end_time).reset_index()
        
        # 先按股票代码排序，然后在每个股票内部按照时间排序
        df_origin = df_origin.sort_values(by=['instrument', 'datetime'])
        # 添加变化率行
        df_origin['ret'] = df_origin.groupby(['instrument'])['$close'].pct_change(1)
        if self.freq == 'days':
            df = self.merge_factors_days(df_origin, factor_list=additional_exprs)
        else:
            df = self.merge_factors_1min(df_origin, factor_list=additional_exprs)
        df = df.drop(columns=['ret'])
        return df

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        df_indexed = df.reset_index(drop=True).set_index(['datetime', 'instrument'])
        dates = df_indexed.index.levels[0]                                    # type: ignore
        df_modified = df_indexed.stack(dropna=False).unstack(level=1)
        stock_ids = df_modified.columns
        values = df_modified.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        values = list(np.array(values))
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
