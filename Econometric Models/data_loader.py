
from qlib.data.dataset.loader import QlibDataLoader

def load_data(start_time, end_time, instruments):
    data_loader = QlibDataLoader(
        instruments=instruments,
        fields=['close', 'high', 'low', 'volume', 'vwap'],
        start_time=start_time,
        end_time=end_time,
        freq='day'
    )
    return data_loader
