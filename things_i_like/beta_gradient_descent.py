import yfinance as yf
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
# could use this repo as an example
# import gloves as gv


class StockBetaCalculator:
    def __init__(self, ticker, region, lookback_months=5*12, learning_rate=0.01, normalize=False, iterations=1000):

        # Params: ticker, region, gradient descent params
        self.ticker = ticker
        self.region = region
        self.index_ticker = self.map_region_to_index(region)
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.iterations = iterations

        # Basics
        self.today = datetime.datetime.today()
        self.end_date = self.today - datetime.timedelta(days=1)
        self.start_date = self.end_date - relativedelta(months=lookback_months)

        # Attributes: store price data and returns
        self.stock_prices = None
        self.index_prices = None
        self.stock_returns = None
        self.index_returns = None

        # Attributes: regression data
        self.X = None
        self.y = None
        self.m = 1  # slope to initiate model
        self.b = 0  # intercept to initiate the model
        self.y_pred = None

        self.beta_gradient_descent = None  # this will be the output
        self.beta_sklearn = None  # this will be the check
        self.beta_yfinance = None  # another check

    def map_region_to_index(self, region):
        # Maps the region to the corresponding index ticker (e.g., 'US' -> 'SPY')
        region_map = {
            'US': 'SPY',
            'EU': 'STOXX',
            'AP': 'VPL'
        }
        return region_map.get(region, 'SPY')

    def fetch_daily_data(self):
        # Pull daily adjusted close prices for the stock and index
        ticker_list = [self.ticker, self.index_ticker]

        # prices = gv.YahooData(self.start_date, self.end_date, ticker_list).get_prices()
        prices = yf.download(ticker_list,
                             start=self.start_date.strftime('%Y-%m-%d'),
                             end=self.end_date.strftime('%Y-%m-%d'))[
            'Adj Close'][ticker_list]

        self.stock_prices = prices[self.ticker]
        self.index_prices = prices[self.index_ticker]

        # returns = gv.YahooData(self.start_date, self.end_date, ticker_list).get_returns()
        returns = prices.pct_change().dropna()
        self.stock_returns = returns[self.ticker]
        self.index_returns = returns[self.index_ticker]

    def gen_regression_data(self):
        # Generate data for regression
        overlapping_dates = self.stock_returns.dropna().index.intersection(self.index_returns.dropna().index)
        self.X = self.index_returns.loc[overlapping_dates].values
        self.y = self.stock_returns.loc[overlapping_dates].values

    def pred_y(self, m, b):
        # Define the linear regression model for historical beta
        return m * self.X + b

    def cost(self, m, b):
        # average of sum of squared errors
        return 1 / (2 * len(self.y)) * np.sum((self.pred_y(m, b) - self.y) ** 2)

    def deriv_cost_m(self, m, b):
        # derivative of cost function with respect to m
        return 1 / len(self.y) * np.sum((self.pred_y(m, b) - self.y) * self.X)

    def deriv_cost_b(self, m, b):
        # derivative of cost function with respect to b
        return 1 / len(self.y) * np.sum(self.pred_y(m, b) - self.y)

    def gradient_descent(self, print_cost=False):
        # iterate to find optimal m and b
        for iteration in range(self.iterations):
            self.m -= self.learning_rate * self.deriv_cost_m(self.m, self.b)
            self.b -= self.learning_rate * self.deriv_cost_b(self.m, self.b)
            if print_cost:
                if iteration in range(0, self.iterations, self.iterations // 3):
                    print(f"Cost after iteration {iteration}: {self.cost(self.m, self.b)}")
        self.beta_gradient_descent = self.m

    def check_beta(self):
        # Calculate beta checks using sklearn and yfinance
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(self.X[:, np.newaxis], self.y[:, np.newaxis])
        self.beta_sklearn = reg.coef_[0][0]
        self.beta_yfinance = yf.Ticker(self.ticker).info.get('beta')


# Example usage
if __name__ == "__main__":
    ticker = "NVDA"
    region = "US"

    sbc = StockBetaCalculator(ticker=ticker, region=region,
                              # gradient descent params
                              learning_rate=0.5, normalize=False, iterations=100000)
    sbc.fetch_daily_data()
    sbc.gen_regression_data()
    sbc.gradient_descent(print_cost=True)
    sbc.check_beta()

    print(f"Beta calcs for {ticker} ({region})")
    print(f"Raw Gradient Descent Beta (5y daily): {round(sbc.beta_gradient_descent, 3)}")
    print(f"Sklearn Beta (5y daily): {round(sbc.beta_sklearn, 3)}")
    print(f"YFinance Beta (5y monthly): {sbc.beta_yfinance}")


