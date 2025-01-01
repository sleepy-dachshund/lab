import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class StockPlotter:
    def __init__(self, data: pd.DataFrame, prices: pd.DataFrame, selected_symbols: list = None):
        self.data = data.sort_values(['date', 'market_cap'], ascending=[False, False]).copy()
        self.data_last = self.data.loc[self.data['date'] == self.data['date'].max()].copy()
        self.prices = prices.loc[:, ['date', 'symbol', 'price']].copy()
        self.returns = prices.loc[:, ['date', 'symbol', 'return', 'return_log']].copy()
        self.selected_symbols = selected_symbols
