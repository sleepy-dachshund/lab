import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
from config.universe import UNIVERSE
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class DataFetcher:
    def __init__(self, additional_symbols: list = None):
        self.symbols = additional_symbols
        self.api_key = os.getenv('VANTAGE_API_KEY')
        self.rpm = int(os.getenv('VANTAGE_RPM'))
        self.base_url = 'https://www.alphavantage.co/query'

        self.start_date = '2017-01-01'

        self.master_ticker_list = []

        self.etf_list = []
        self.etf_dict = {}
        self.headsup_universe = []

        self.overviews = pd.DataFrame()

        self.prices_tuple = (pd.DataFrame(), pd.DataFrame())
        self.prices = pd.DataFrame()

        self.financials_is = pd.DataFrame()
        self.financials_bs = pd.DataFrame()
        self.financials_cf = pd.DataFrame()
        self.financials = pd.DataFrame()

        self.data = pd.DataFrame()

        self.pulled_etf = False
        self.pulled_overviews = False
        self.pulled_prices = False
        self.pulled_is = False
        self.pulled_bs = False
        self.pulled_cf = False

    def fetch_etf_constituents(self, etf_list: list = None):
        if etf_list is None:
            etf_list = ['IWV', 'SPY', 'QQQ', 'VUG', 'SMH', 'SPSM', 'IWM', 'DUHP', 'DFLV', 'DFAT', 'DFSV']
        self.etf_list = etf_list

        self.master_ticker_list = []
        for etf in tqdm(etf_list, desc='Fetching ETF Constituents'):

            url = f'https://www.alphavantage.co/query?function=ETF_PROFILE&symbol={etf}&apikey=' + self.api_key
            r = requests.get(url)
            response = r.json()

            etf_holdings = pd.DataFrame(response['holdings'])
            etf_holdings['weight'] = pd.to_numeric(etf_holdings['weight'])
            etf_holdings['date'] = datetime.now().strftime('%Y-%m-%d')
            etf_holdings['etf'] = etf

            self.etf_dict[etf] = etf_holdings

            for ticker in etf_holdings['symbol']:
                if ticker not in self.master_ticker_list:
                    self.master_ticker_list.append(ticker)

        if self.symbols is None:
            self.symbols = self.master_ticker_list

        self.pulled_etf = True

        return self.etf_dict, self.master_ticker_list

    def get_headsup_universe(self, etf_list: list = None):
        """Choose some ETF(s) as the base for the headsup universe."""
        if etf_list is None:
            etf_list = ['SPY']

        for etf in etf_list:
            self.headsup_universe += self.etf_dict[etf]['symbol'].tolist()
        for ticker in self.symbols:
            if ticker not in self.headsup_universe:
                self.headsup_universe += [ticker]
        self.headsup_universe = list(set(self.headsup_universe))
        return self.headsup_universe

    def fetch_company_overviews(self, symbols: list = None):
        return self._fetch_data(
            symbols=symbols,
            function='OVERVIEW',
            desc='Fetching Company Overview Data',
            result_attr='overviews',
            result_pulled_attr='pulled_overviews',
            exclude_etf=True,
            parser=self._parse_company_overview
        )

    def _parse_company_overview(self, data):
        symbol = data['Symbol']
        overview = pd.DataFrame({
            'name': data['Name'],
            'description': data['Description'],
            'sector': data['Sector'],
            'industry': data['Industry'],
            'market_cap': pd.to_numeric(data['MarketCapitalization'], errors='coerce'),
            'ebitda': pd.to_numeric(data['EBITDA'], errors='coerce'),
            'earnings_growth': pd.to_numeric(data['QuarterlyEarningsGrowthYOY'], errors='coerce'),
            'revenue_growth': pd.to_numeric(data['QuarterlyRevenueGrowthYOY'], errors='coerce'),
            'roa': pd.to_numeric(data['ReturnOnAssetsTTM'], errors='coerce'),
            'roe': pd.to_numeric(data['ReturnOnEquityTTM'], errors='coerce'),
            'pe_trailing': pd.to_numeric(data['TrailingPE'], errors='coerce'),
            'pe_forward': pd.to_numeric(data['ForwardPE'], errors='coerce'),
            'peg': pd.to_numeric(data['PEGRatio'], errors='coerce'),
            'ps_trailing': pd.to_numeric(data['PriceToSalesRatioTTM'], errors='coerce'),
            'pb_trailing': pd.to_numeric(data['PriceToBookRatio'], errors='coerce'),
            'evr': pd.to_numeric(data['EVToRevenue'], errors='coerce'),
            'evebitda': pd.to_numeric(data['EVToEBITDA'], errors='coerce'),
            'profit_margin': pd.to_numeric(data['ProfitMargin'], errors='coerce'),
            '52w_high': pd.to_numeric(data['52WeekHigh'], errors='coerce'),
            '52w_low': pd.to_numeric(data['52WeekLow'], errors='coerce'),
            '50dma': pd.to_numeric(data['50DayMovingAverage'], errors='coerce'),
            '200dma': pd.to_numeric(data['200DayMovingAverage'], errors='coerce'),
            'analyst_target_price': pd.to_numeric(data['AnalystTargetPrice'], errors='coerce'),
            'analyst_strong_buy': pd.to_numeric(data['AnalystRatingStrongBuy'], errors='coerce'),
            'analyst_buy': pd.to_numeric(data['AnalystRatingBuy'], errors='coerce'),
            'analyst_hold': pd.to_numeric(data['AnalystRatingHold'], errors='coerce'),
            'analyst_sell': pd.to_numeric(data['AnalystRatingSell'], errors='coerce'),
            'analyst_strong_sell': pd.to_numeric(data['AnalystRatingStrongSell'], errors='coerce'),
        }, index=[symbol])

        return overview

    def fetch_daily_prices(self, symbols: list = None):
        return self._fetch_data(
            symbols=symbols,
            function='TIME_SERIES_DAILY_ADJUSTED',
            desc='Fetching Daily Prices and Volumes',
            result_attr='prices',
            result_pulled_attr='pulled_prices',
            include_etf=True,
            parser=self._parse_daily_prices
        )

    def _parse_daily_prices(self, data):
        symbol = data['Meta Data']['2. Symbol']
        df = pd.DataFrame(data['Time Series (Daily)']).T
        prices = pd.to_numeric(df['5. adjusted close'], errors='coerce')
        prices.name = symbol
        volumes = pd.to_numeric(df['6. volume'], errors='coerce')
        volumes.name = symbol
        return prices, volumes

    def fetch_quarterly_financials_is(self, symbols: list = None):
        return self._fetch_data(
            symbols=[symbol for symbol in symbols if symbol not in self.etf_list],
            function='INCOME_STATEMENT',
            desc='Fetching Income Statement Data',
            result_attr='financials_is',
            result_pulled_attr='pulled_is',
            parser=self._parse_financials_is
        )

    def _parse_financials_is(self, data):
        df = pd.DataFrame(data['quarterlyReports'])
        df['tax_rate'] = (pd.to_numeric(df['incomeTaxExpense'], errors='coerce').fillna(0)
                          / pd.to_numeric(df['incomeBeforeTax'], errors='coerce').fillna(0))
        df['nopat'] = (pd.to_numeric(df['ebit'], errors='coerce').fillna(0)
                       * (1 - df['tax_rate']))
        financials = pd.DataFrame({
            'revenue': pd.to_numeric(df['totalRevenue'], errors='coerce'),
            'gross_profit': pd.to_numeric(df['grossProfit'], errors='coerce'),
            'operating_income': pd.to_numeric(df['operatingIncome'], errors='coerce'),
            'nopat': pd.to_numeric(df['nopat'], errors='coerce'),
            'research_and_development': pd.to_numeric(df['researchAndDevelopment'], errors='coerce'),
            'net_income': pd.to_numeric(df['netIncome'], errors='coerce'),
        })
        financials.index = pd.to_datetime(df['fiscalDateEnding'])
        financials = financials.loc[financials.index >= self.start_date]
        return financials

    def fetch_quarterly_financials_bs(self, symbols: list = None):
        return self._fetch_data(
            symbols=[symbol for symbol in symbols if symbol not in self.etf_list],
            function='BALANCE_SHEET',
            desc='Fetching Balance Sheet Data',
            result_attr='financials_bs',
            result_pulled_attr='pulled_bs',
            parser=self._parse_financials_bs
        )

    def _parse_financials_bs(self, data):
        df = pd.DataFrame(data['quarterlyReports'])

        df['working_capital'] = (pd.to_numeric(df['totalAssets'], errors='coerce').fillna(0)
                                 - pd.to_numeric(df['totalLiabilities'], errors='coerce').fillna(0))

        df['net_working_capital'] = ((pd.to_numeric(df['totalCurrentAssets'], errors='coerce').fillna(0)
                                     - pd.to_numeric(df['cashAndShortTermInvestments'], errors='coerce').fillna(0))
                                    - (pd.to_numeric(df['totalCurrentLiabilities'], errors='coerce').fillna(0)
                                       - pd.to_numeric(df['currentDebt'], errors='coerce').fillna(0)))

        df['net_ppe'] = (pd.to_numeric(df['propertyPlantEquipment'], errors='coerce').fillna(0)
                         - pd.to_numeric(df['accumulatedDepreciationAmortizationPPE'], errors='coerce').fillna(0))

        df['total_intangibles'] = pd.to_numeric(df['intangibleAssets'], errors='coerce').fillna(0)
        df['other'] = pd.to_numeric(df['otherNonCurrentAssets'], errors='coerce').fillna(0)

        df['invested_capital_op'] = (df['net_working_capital']
                                     + df['net_ppe']
                                     + df['total_intangibles']
                                     + df['other'])

        df['invested_capital_fi'] = (pd.to_numeric(df['longTermDebt'], errors='coerce').fillna(0)
                                     + pd.to_numeric(df['shortTermDebt'], errors='coerce').fillna(0)
                                     + pd.to_numeric(df['totalShareholderEquity'], errors='coerce').fillna(0)
                                     - pd.to_numeric(df['cashAndShortTermInvestments'], errors='coerce').fillna(0))

        financials = pd.DataFrame({
            'assets': pd.to_numeric(df['totalAssets'], errors='coerce'),
            'liabilities': pd.to_numeric(df['totalLiabilities'], errors='coerce'),
            'shareholder_equity': pd.to_numeric(df['totalShareholderEquity'], errors='coerce'),
            'cash': pd.to_numeric(df['cashAndShortTermInvestments'], errors='coerce'),
            'current_assets': pd.to_numeric(df['totalCurrentAssets'], errors='coerce'),
            'current_liabilities': pd.to_numeric(df['totalCurrentLiabilities'], errors='coerce'),
            'retained_earnings': pd.to_numeric(df['retainedEarnings'], errors='coerce'),
            'shares_out': pd.to_numeric(df['commonStockSharesOutstanding'], errors='coerce'),
            'working_capital': pd.to_numeric(df['working_capital'], errors='coerce'),
            'net_working_capital': pd.to_numeric(df['net_working_capital'], errors='coerce'),
            'invested_capital_op': pd.to_numeric(df['invested_capital_op'], errors='coerce'),
            'invested_capital_fi': pd.to_numeric(df['invested_capital_fi'], errors='coerce'),
            'ppe': pd.to_numeric(df['propertyPlantEquipment'], errors='coerce'),
        })

        financials.index = pd.to_datetime(df['fiscalDateEnding'])
        financials = financials.loc[financials.index >= self.start_date]
        return financials

    def fetch_quarterly_financials_cf(self, symbols: list = None):
        return self._fetch_data(
            symbols=[symbol for symbol in symbols if symbol not in self.etf_list],
            function='CASH_FLOW',
            desc='Fetching Cash Flow Statement Data',
            result_attr='financials_cf',
            result_pulled_attr='pulled_cf',
            parser=self._parse_financials_cf
        )

    def _parse_financials_cf(self, data):
        df = pd.DataFrame(data['quarterlyReports'])

        df['fcf'] = (pd.to_numeric(df['operatingCashflow'], errors='coerce').fillna(0)
                     - pd.to_numeric(df['capitalExpenditures'], errors='coerce').fillna(0))

        financials = pd.DataFrame({
            'capex': pd.to_numeric(df['capitalExpenditures'], errors='coerce'),
            'operating_cash_flow': pd.to_numeric(df['operatingCashflow'], errors='coerce'),
            'fcf': pd.to_numeric(df['fcf'], errors='coerce'),
            'stock_buybacks': pd.to_numeric(df['paymentsForRepurchaseOfCommonStock'], errors='coerce'),
            'stock_sales': pd.to_numeric(df['proceedsFromIssuanceOfCommonStock'], errors='coerce'),
        })

        financials.index = pd.to_datetime(df['fiscalDateEnding'])
        financials = financials.loc[financials.index >= self.start_date]
        return financials

    def _fetch_data(self, symbols: list = None, function: str = None, desc: str = None, result_attr: str = None,
                    result_pulled_attr: str = None, exclude_etf: bool = False, include_etf: bool = False,
                    parser=None):
        """Fetches data from the Alpha Vantage API."""
        all_data = {}

        if symbols is None:
            symbols = self.symbols

        if include_etf:
            symbols += self.etf_list

        if exclude_etf:
            for symbol in tqdm(symbols, desc=desc):
                if symbol in self.etf_list:
                    continue
                try:
                    params = {
                        'function': function,
                        'symbol': symbol,
                        'apikey': self.api_key
                    }

                    response = requests.get(self.base_url, params=params)
                    data = response.json()
                    all_data[symbol] = parser(data)
                    time.sleep(60 / self.rpm)

                except Exception as e:
                    print(f"Error fetching {desc} for {symbol}: {e}")
        else:
            for symbol in tqdm(symbols, desc=desc):
                try:
                    params = {
                        'function': function,
                        'symbol': symbol,
                        'apikey': self.api_key,
                        'outputsize': 'full'
                    }

                    response = requests.get(self.base_url, params=params)
                    data = response.json()
                    all_data[symbol] = parser(data)
                    time.sleep(60 / self.rpm)
                except Exception as e:
                    print(f"Error fetching {desc} for {symbol}: {e}")

        if all_data:
            if result_attr == 'overviews':
                df = pd.concat(all_data.values(), axis=0, keys=all_data.keys())
                df.reset_index(drop=False, inplace=True)
                df = df.drop(columns=['level_0']).rename(columns={'level_1': 'symbol'})
                df = df.sort_values(['market_cap'], ascending=False).reset_index(drop=True)
                df.drop_duplicates(subset=['symbol'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                setattr(self, result_attr, df.copy())
                setattr(self, result_pulled_attr, True)
                return df
            elif result_attr == 'prices':
                prices_df = pd.DataFrame({symbol: data[0] for symbol, data in all_data.items()})
                volumes_df = pd.DataFrame({symbol: data[1] for symbol, data in all_data.items()})
                prices_df.index = pd.to_datetime(prices_df.index)
                volumes_df.index = pd.to_datetime(volumes_df.index)
                prices_df = prices_df.loc[prices_df.index >= self.start_date]
                volumes_df = volumes_df.loc[volumes_df.index >= self.start_date]

                self.prices_tuple = (prices_df, volumes_df)

                setattr(self, result_attr, (prices_df, volumes_df))
                setattr(self, result_pulled_attr, True)
                return prices_df, volumes_df
            else:
                financials_all = {}
                for symbol in all_data:
                    financials = all_data[symbol]
                    financials['symbol'] = symbol
                    financials = financials.reset_index().set_index(['fiscalDateEnding', 'symbol'])
                    financials_all[symbol] = financials

                financials_df = pd.concat(financials_all.values(), axis=0, keys=financials_all.keys())
                financials_df = financials_df.reset_index()
                financials_df = financials_df.set_index(['fiscalDateEnding', 'symbol'])
                financials_df.index.names = ['fiscalDateEnding', 'symbol']
                financials_df.reset_index(drop=False, inplace=True)
                financials_df = financials_df.loc[financials_df['fiscalDateEnding'] >= self.start_date]
                for col in financials_df.columns:
                    if 'level_' in col:
                        financials_df = financials_df.drop(columns=[col])
                financials_df = financials_df.sort_values(['symbol', 'fiscalDateEnding']).reset_index(drop=True)

                setattr(self, result_attr, financials_df.copy())
                setattr(self, result_pulled_attr, True)
                return financials_df
        else:
            return None

    def combine_financials(self):
        """Combines the income statement, balance sheet, and cash flow data."""
        if self.pulled_is and self.pulled_bs and self.pulled_cf:
            self.financials = self.financials_is.merge(self.financials_bs, on=['fiscalDateEnding', 'symbol'],
                                                       how='outer', suffixes=('', '_bs'))
            self.financials = self.financials.merge(self.financials_cf, on=['fiscalDateEnding', 'symbol'],
                                                       how='outer', suffixes=('', '_cf'))
            self.financials.drop_duplicates(subset=['fiscalDateEnding', 'symbol'], inplace=True)
            self.financials = self.financials.loc[self.financials['fiscalDateEnding'] >= self.start_date]
            self.financials.reset_index(drop=True, inplace=True)
            return self.financials
        else:
            return None

    def format_financials(self):
        """Formats the financials DataFrame by calculating annual values and scaling."""
        bs_cols = [col for col in self.financials_bs.columns[2:]]
        fq_ex_bs = self.financials.drop(columns=bs_cols).copy()

        # Calculate rolling annual sums for relevant columns
        financials_annual = (fq_ex_bs.sort_values('fiscalDateEnding', ascending=True)
                             .set_index(['fiscalDateEnding'])
                             .groupby('symbol')
                             .rolling(4)
                             .sum()
                             .reset_index())

        # Merge annualized is/cf data with original bs data
        self.financials = financials_annual.merge(self.financials.loc[:, ['fiscalDateEnding', 'symbol'] + bs_cols],
                                                  on=['fiscalDateEnding', 'symbol'], how='left')

        # Scale financial values to billions
        self.financials = (self.financials.set_index(['fiscalDateEnding', 'symbol']) / 1e9).reset_index(drop=False)
        self.financials = self.financials.rename(columns={'fiscalDateEnding': 'date'})

        # Calculate YoY changes
        self.financials.sort_values(['date', 'symbol'], inplace=True)
        yoy_cols = ['revenue', 'gross_profit', 'operating_income', 'nopat',
                    'net_income', 'fcf', 'capex', 'shares_out']
        for value in yoy_cols:
            self.financials[f'{value}_yoy'] = (self.financials.groupby('symbol')[value]
                                               .transform(lambda x: x / x.shift(4) - 1)
                                               .replace([np.inf, -np.inf], np.nan))

    def format_prices_volumes(self):
        """Formats and combines prices and volumes data."""
        prices, volumes = self.prices_tuple

        # Calculate simple and log returns
        returns_simple = prices.pct_change()
        returns_log = np.log(prices / prices.shift(1))

        # Stack prices, volumes, returns
        prices_stack = (prices.stack()
                        .reset_index()
                        .rename(columns={'level_0': 'date', 'level_1': 'symbol', 0: 'price'})
                        .sort_values(['date', 'symbol'])
                        .reset_index(drop=True))
        volumes_stack = (volumes.stack()
                         .reset_index()
                         .rename(columns={'level_0': 'date', 'level_1': 'symbol', 0: 'volume'})
                         .sort_values(['date', 'symbol'])
                         .reset_index(drop=True))
        returns_stack = (returns_simple.stack()
                         .reset_index()
                         .rename(columns={'level_0': 'date', 'level_1': 'symbol', 0: 'return'})
                         .sort_values(['date', 'symbol'])
                         .reset_index(drop=True))
        returns_log_stack = (returns_log.stack()
                             .reset_index()
                             .rename(columns={'level_0': 'date', 'level_1': 'symbol', 0: 'return_log'})
                             .sort_values(['date', 'symbol'])
                             .reset_index(drop=True))

        # Combine prices, volumes, and returns
        self.data = (prices_stack
                     .merge(returns_stack, on=['date', 'symbol'], how='left')
                     .merge(returns_log_stack, on=['date', 'symbol'], how='left')
                     .merge(volumes_stack, on=['date', 'symbol'], how='left'))
        self.prices = self.data.copy()

    def combine_prices_financials(self):
        """Combines the formatted prices and financials data."""
        self.data.sort_values(['date', 'symbol'], inplace=True)
        self.financials.sort_values(['date', 'symbol'], inplace=True)

        # Use merge_asof to align financial data based on the closest date
        self.data = pd.merge_asof(self.data, self.financials, on='date', by='symbol', direction='backward')
        self.data = self.overviews.loc[:, ['symbol', 'sector']].merge(self.data, how='left', on='symbol', validate='1:m')
        self.data = self.data.loc[self.data.date >= self.start_date]
        self.data.sort_values(['date', 'symbol'], inplace=True)

    def calc_additional_data(self):
        """Calculates final metrics based on the combined data."""
        self.data['market_cap'] = self.data['price'] * self.data['shares_out']
        self.data['roa'] = self.data['net_income'] / self.data['assets']
        self.data['roe'] = self.data['net_income'] / self.data['shareholder_equity']
        self.data['roic'] = self.data['nopat'] / np.where(self.data['invested_capital_op'] == 0, np.nan, self.data['invested_capital_op'])
        self.data['roic_fi'] = self.data['nopat'] / np.where(self.data['invested_capital_fi'] == 0, np.nan, self.data['invested_capital_fi'])
        self.data['gross_margin'] = self.data['gross_profit'] / self.data['revenue']
        self.data['operating_margin'] = self.data['operating_income'] / self.data['revenue']
        self.data['net_margin'] = self.data['net_income'] / self.data['revenue']
        self.data['fcf_margin'] = self.data['fcf'] / self.data['revenue']
        self.data['btm'] = (self.data['assets'] - self.data['liabilities']) / self.data['market_cap']
        self.data['ps'] = self.data['market_cap'] / self.data['revenue']
        self.data['pe'] = self.data['market_cap'] / self.data['net_income']
        self.data['fcf_yield'] = self.data['fcf'] / self.data['market_cap']

        # clip some ratios to +/- 200% to avoid extreme values
        for ratio in ['roa', 'roe', 'roic', 'roic_fi']:
            self.data[ratio] = self.data[ratio].clip(-2, 2)

    def save_data(self, file_path: str = 'data/'):
        """Saves all data to Excel and Pickle files."""
        self.overviews.to_excel(f'{file_path}overviews.xlsx')
        self.overviews.to_pickle(f'{file_path}overviews.pkl')
        self.financials.to_excel(f'{file_path}financials.xlsx')
        self.financials.to_pickle(f'{file_path}financials.pkl')
        self.prices.to_pickle(f'{file_path}prices.pkl')
        self.data.loc[self.data.date == self.data.date.max()].to_excel(f'{file_path}data.xlsx')
        self.data.to_pickle(f'{file_path}data.pkl')
        for etf in self.etf_dict:
            self.etf_dict[etf].to_excel(f'{file_path}{etf}.xlsx')

    def read_data(self, file_path: str = 'data/'):
        """Reads all data from pickle files."""
        self.overviews = pd.read_pickle(f'{file_path}overviews.pkl')
        self.financials = pd.read_pickle(f'{file_path}financials.pkl')
        self.prices = pd.read_pickle(f'{file_path}prices.pkl')
        self.data = pd.read_pickle(f'{file_path}data.pkl')
        for etf in UNIVERSE['major_indices'] + UNIVERSE['sector_etfs']:
            try:
                self.etf_dict[etf] = pd.read_excel(f'{file_path}{etf}.xlsx')
            except FileNotFoundError:
                continue

    def fetch_all_data(self, etf_list: list = None, read_cache: bool = False):
        """Fetches and processes all data."""
        if read_cache:
            self.read_data()
            return self.overviews, self.prices, self.financials, self.data, self.etf_dict
        else:
            self.fetch_etf_constituents(etf_list=etf_list)
            self.get_headsup_universe(etf_list=etf_list)
            self.fetch_company_overviews(symbols=self.headsup_universe)
            # todo: map 'sector' col to UNIVERSE['sector_etfs'] constituents
            self.fetch_daily_prices(symbols=self.headsup_universe)
            self.fetch_quarterly_financials_is(symbols=self.headsup_universe)
            self.fetch_quarterly_financials_bs(symbols=self.headsup_universe)
            self.fetch_quarterly_financials_cf(symbols=self.headsup_universe)
            self.combine_financials()
            self.format_financials()
            self.format_prices_volumes()
            self.combine_prices_financials()
            self.calc_additional_data()
            self.save_data()
            return self.overviews, self.prices, self.financials, self.data, self.etf_dict
