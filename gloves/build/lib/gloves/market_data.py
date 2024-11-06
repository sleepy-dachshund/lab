import yfinance as yf
import datetime


class YahooData:
    def __init__(self, start_date, end_date, ticker_list: list):
        self.ticker_list = ticker_list
        self.start_date = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
        self.end_date = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')

        self.data = None
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.adj_close = None
        self.returns = None
        self.volume = None

    def fetch_data(self):
        self.data = yf.download(self.ticker_list,
                                start=self.start_date,
                                end=self.end_date)
        self.adj_close = self.data['Adj Close'][self.ticker_list]
        self.open = self.data['Open'][self.ticker_list]
        self.high = self.data['High'][self.ticker_list]
        self.low = self.data['Low'][self.ticker_list]
        self.close = self.data['Close'][self.ticker_list]
        self.returns = self.adj_close.pct_change()[self.ticker_list]
        self.volume = self.data['Volume'][self.ticker_list]

    def get_data(self, stacked_output: bool = False):
        if self.data is None:
            self.fetch_data()
        return self.data.stack() if stacked_output else self.data

    def get_prices(self, price_type: str = 'Adj Close', stacked_output: bool = False):
        if self.data is None:
            self.fetch_data()
        if price_type == 'Adj Close':
            return self.adj_close.stack().to_frame(price_type).reset_index() if stacked_output else self.adj_close
        elif price_type == 'Open':
            return self.open.stack().to_frame(price_type).reset_index() if stacked_output else self.open
        elif price_type == 'High':
            return self.high.stack().to_frame(price_type).reset_index() if stacked_output else self.high
        elif price_type == 'Low':
            return self.low.stack().to_frame(price_type).reset_index() if stacked_output else self.low
        elif price_type == 'Close':
            return self.close.stack().to_frame(price_type).reset_index() if stacked_output else self.close
        else:
            raise ValueError('Invalid "price_type" param. Please use: "Adj Close", "Open", "High", "Low", or "Close"')

    def get_returns(self, stacked_output: bool = False):
        if self.data is None:
            self.fetch_data()
        return self.returns.stack().to_frame('Return').reset_index() if stacked_output else self.returns

    def get_volume(self, stacked_output: bool = False):
        if self.data is None:
            self.fetch_data()
        return self.volume.stack().to_frame('Volume').reset_index() if stacked_output else self.volume


if __name__ == '__main__':
    start_date = '2020-01-01'
    end_date = datetime.datetime.today()
    yd = YahooData(start_date=start_date,
                   end_date=end_date,
                   ticker_list=['AAPL', 'GOOGL', 'META', 'ASML'])
    data = yd.get_data()
    prices = yd.get_prices(price_type='Adj Close')
    prices_stacked = yd.get_prices(price_type='Adj Close', stacked_output=True)
    returns = yd.get_returns()
    returns_stacked = yd.get_returns(stacked_output=True)
    volume = yd.get_volume()
    volume_stacked = yd.get_volume(stacked_output=True)
