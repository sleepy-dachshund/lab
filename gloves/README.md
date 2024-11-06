# gloves
when you go into a lab the first thing you do is put on gloves (or, import gloves as gv)

this is a very basic util package. i'd like to expand it if i had free time.

## instructions
pull the repo and navigate to outer "gloves" directory. "pip install ."

## example
import gloves as gv
yd = gv.YahooData(start_date='2020-01-01', end_date=datetime.datetime.today(), ticker_list=['AAPL', 'SPY'])
prices = yd.get_prices(price_type='Adj Close')
returns = yd.get_returns()
