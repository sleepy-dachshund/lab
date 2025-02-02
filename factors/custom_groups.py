

"""
Setup & Run Instructions:
------------------------
1. Directory Structure:
   - Place 'custom_groups.py' in your working directory
   - Create a 'data_dump/' subdirectory (or modify DATA_DIRECTORY if desired, line 46)

2. Required Input Files (in data_dump/):
   - XLY.csv: Consumer Discretionary ETF holdings
   - XLP.csv: Consumer Staples ETF holdings
   Download from sectorspdrs.com (e.g., https://www.sectorspdrs.com/mainfund/xly)
   Remove the top information row before saving.

3. Generated Cache Files:
   - prices.csv: Historical price data from yfinance
   - raw_data.pkl: Processed daily return data
   - groups.xlsx: Final output showing clustered symbols

4. Running:
   Simply execute custom_groups.py. First run will take longer to generate cache files.
   Check groups.xlsx for final clustering results.
"""

'''
WHAT THE BELOW CODE DOES:

Many investors believe that the standard sector classifications are inaccurate and
would prefer to generate their own that more tightly track securities that are like one another.

Approach: Develop a method that generates these custom groups -- *using only historical returns.*

Example set -- Consumer stocks (S&P500 Consumer Staples and Discretionary).

1. Pull ETF holdings data for Consumer Staples (XLP) and Consumer Discretionary (XLY).
2. Pull daily prices for all stocks in the ETFs.
3. Calculate raw returns for all stocks.
4. Combine data into one dataframe.
5. Calculate sector residuals for Consumer Staples and Discretionary (sector return unexplained by market).
6. Calculate alpha returns for all symbols using a model that includes market and sector residuals.
7. Cluster symbols based on alpha returns using a distance matrix approach.
8. Output the clustered symbols.

PLEASE NOTE
- This is not a complete custom grouping model, just an example.
- Assume any choices made are just there to illustrate that the choice exists and therefore must be studied.

'''

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path

DATA_DIRECTORY = 'data_dump/'


''' =====================================================================================================
    Data Pull
===================================================================================================== '''


def pull_cached_etf_holdings(etfs: list) -> Tuple[pd.DataFrame, list]:
    """
    pull cached ETF holdings data from local csv files
    """
    # todo: this doesn't have to be so static, but priority is not high
    sector_mapping = {'XLP': 'Consumer Staples', 'XLY': 'Consumer Discretionary'}
    theme_mapping = {'XLP': 'Consumer', 'XLY': 'Consumer'}

    df = []
    for etf in etfs:
        try:
            temp = pd.read_csv(f'{DATA_DIRECTORY}{etf}.csv')
            temp['etf'] = etf
            temp['sector'] = sector_mapping[etf]
            temp['theme'] = theme_mapping[etf]
            df.append(temp)
        except FileNotFoundError:
            pass
    df = pd.concat(df)
    df.columns = [x.lower().replace(' ', '_') for x in df.columns]
    df = df.loc[:, ['theme', 'sector', 'etf', 'symbol', 'company_name']].copy()
    df['symbol'] = df['symbol'].str.replace('.', '-').str.upper()
    ticker_list = df.symbol.unique().tolist()
    ticker_list = [t for t in ticker_list if t != 'MRP']  # delist, bad data
    return df, ticker_list


def pull_daily_prices(tickers: list, start='2020-01-01', end=None, try_cache=True, data_dir=DATA_DIRECTORY) -> pd.DataFrame:
    """
    pull daily adj. close prices from yfinance. check local cache first by default.
    """
    # todo: note if params are changed in the future then the cache will be invalid -- missing tickers, dates, etc.

    def pull_fresh_prices(tickers, start, end) -> pd.DataFrame:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        if len(tickers) == 1:
            data = data.to_frame()
        data.to_csv(data_dir+'prices.csv')
        return data

    if not tickers:
        return pd.DataFrame()

    if try_cache:
        try:
            data = pd.read_csv(data_dir+'prices.csv', index_col=0, parse_dates=True)
            data = data[tickers]
            data = data.loc[start:end] if end else data.loc[start:]
            return data
        except FileNotFoundError:
            data = pull_fresh_prices(tickers, start, end)
            return data
    else:
        data = pull_fresh_prices(tickers, start, end)
        return data


def calc_raw_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    calculate raw returns from prices
    """
    return prices.pct_change()


def combine_raw_data(returns, index_rets, sector_data, sector_etfs) -> pd.DataFrame:
    """
    combine raw data into one dataframe
    """
    df = returns.stack().reset_index()
    df.columns = ['date', 'symbol', 'return']
    df = df.merge(index_rets, left_on='date', right_index=True, suffixes=('', '_index'))
    df = df.merge(sector_data, left_on='symbol', right_on='symbol', how='left')
    df = df.loc[:, ['date', 'theme', 'symbol', 'return', 'SPY']+sector_etfs].copy()
    return df


''' =====================================================================================================
    Approximate Alpha Returns
===================================================================================================== '''


def calc_rolling_sector_residuals(df: pd.DataFrame, sector_col: str, market_col: str = 'SPY', window: int = 126) -> pd.DataFrame:
    """
    get portion of sector returns unexplained by market.
    for a single sector, run a rolling regression of sector returns on market returns
    using a 126-day window.
    return a series of the daily sector residuals.
    """
    df_sorted = df.sort_values('date').copy()
    residuals = []

    # loop through each index beyond the warm-up period
    for i in range(len(df_sorted)):
        if i < window:
            residuals.append(np.nan)
        else:
            # data in the window
            window_data = df_sorted.iloc[i-window:i]
            y = window_data[sector_col].values
            X = sm.add_constant(window_data[market_col].values)
            model = sm.OLS(y, X).fit()

            # pred current day
            current_market = df_sorted[market_col].iloc[i]
            predicted_sector = model.predict([1, current_market])
            actual_sector = df_sorted[sector_col].iloc[i]
            residuals.append(actual_sector - predicted_sector[0])

    # create new column for residual
    df_sorted[f'{sector_col}_resid'] = residuals
    return df_sorted[['date', f'{sector_col}_resid']]


def calc_rolling_alpha(df: pd.DataFrame, symbol: str, window: int = 126) -> pd.DataFrame:
    """
    for a single symbol, run a rolling regression:
      return_stock ~ return_market + sector1_resid + sector2_resid
    to compute predicted return and alpha (residual)
    returns a df with columns [date, symbol, alpha]
    """
    # filter to symbol data
    data_sym = df[df['symbol'] == symbol].sort_values('date').copy()
    data_sym = data_sym.dropna(subset=['return', 'SPY', 'XLP_resid', 'XLY_resid'])
    alphas = []

    for i in range(len(data_sym)):
        if i < window:
            alphas.append(np.nan)
        else:
            window_data = data_sym.iloc[i - window:i]
            y = window_data['return'].values
            X = window_data[['SPY', 'XLP_resid', 'XLY_resid']].values
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            # pred current day
            factors_t = data_sym[['SPY', 'XLP_resid', 'XLY_resid']].iloc[i].values
            factors_t = np.insert(factors_t, 0, 1)
            predicted = model.predict(factors_t)
            actual = data_sym['return'].iloc[i]
            alphas.append(actual - predicted[0])

    data_sym['alpha_ret'] = alphas
    return data_sym[['date', 'symbol', 'alpha_ret']]


@dataclass
class DataParams:
    """params for data prep and alpha calculation"""
    sector_etfs: List[str] = field(default_factory=lambda: ['XLP', 'XLY'])
    market_etf: str = 'SPY'
    start_date: str = '2016-01-01'
    cache_file: str = 'raw_data.pkl'

    @property
    def all_etfs(self) -> List[str]:
        """list all ETFs including market ETF"""
        return [self.market_etf] + self.sector_etfs


class DataPreparation:
    """calss to handle data prep and alpha rets -- with caching"""

    def __init__(
            self,
            params: DataParams,
            data_directory: Path,
            force_refresh: bool = False
    ):
        self.params = params
        self.cache_path = data_directory / params.cache_file
        self.force_refresh = force_refresh

    def load_cached_data(self) -> Optional[pd.DataFrame]:
        """attempt to load cached data if available"""
        try:
            if self.force_refresh:
                return None
            return pd.read_pickle(self.cache_path)
        except FileNotFoundError:
            return None

    def prepare_base_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """pull and prepare price and return data"""

        # pull ETF holdings -- CURRENTLY MUST BE FROM CACHE
        sector_data, tickers = pull_cached_etf_holdings(etfs=self.params.sector_etfs)

        # get daily prices
        all_tickers = tickers + self.params.all_etfs
        prices = pull_daily_prices(all_tickers, start=self.params.start_date)

        # calc returns
        all_returns = calc_raw_returns(prices)

        # split into index and single-name returns
        index_returns = all_returns[self.params.all_etfs].copy()
        security_returns = all_returns.drop(self.params.all_etfs, axis=1)

        # combine data
        df = combine_raw_data(
            security_returns,
            index_returns,
            sector_data,
            self.params.sector_etfs
        )
        df.dropna(axis=0, inplace=True)

        return df, index_returns

    def calculate_market_adjusted_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """calculate market-adjusted returns"""
        df['return_mktadj'] = df['return'] - df[self.params.market_etf]
        return df

    def calculate_sector_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """calc sector residuals for all sector ETFs"""
        market_cols = ['date'] + self.params.all_etfs
        base_data = df.loc[:, market_cols].drop_duplicates()

        for sector_col in self.params.sector_etfs:
            sector_residuals = calc_rolling_sector_residuals(base_data, sector_col)
            df = pd.merge(df, sector_residuals, on='date', how='left')

        return df

    def calculate_alpha_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """calc alpha returns for all symbols"""
        symbols = df['symbol'].unique()
        alpha_results = []

        for symbol in symbols:
            alpha_df = calc_rolling_alpha(df, symbol)
            alpha_results.append(alpha_df)

        df_alpha = pd.concat(alpha_results, axis=0)
        return pd.merge(df, df_alpha, on=['date', 'symbol'], how='left')

    def prepare_data(self) -> pd.DataFrame:
        """primary method to prep all data with caching"""

        # check cache first
        if (df := self.load_cached_data()) is not None:
            return df

        # prep base data
        df, _ = self.prepare_base_data()

        # calc return measures
        df = self.calculate_market_adjusted_returns(df)
        df = self.calculate_sector_residuals(df)
        df = self.calculate_alpha_returns(df)

        # cache results
        df.to_pickle(self.cache_path)

        return df


def prepare_clustering_data(
        data_params: Optional[DataParams] = None,
        data_directory: Optional[Path] = None,
        force_refresh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    function to prep data for clustering analysis

    return: (prepared_data, index_returns)
    """
    # use default params if none provided
    data_params = data_params or DataParams()
    data_directory = data_directory or Path(DATA_DIRECTORY)

    # init data prep class
    prep = DataPreparation(
        params=data_params,
        data_directory=data_directory,
        force_refresh=force_refresh
    )

    # get base data first (needed for both cached and non-cached paths)
    df, index_returns = prep.prepare_base_data()

    # prep full dataset
    df = prep.prepare_data()

    return df, index_returns


''' =====================================================================================================
    Clustering Functions
===================================================================================================== '''


@dataclass
class ClusteringParams:
    """organize all parameters in one place"""
    return_type: str = 'alpha_ret'  # returns to use for clustering
    avg_names_per_group: int = 5  # average number of names per group
    linkage: str = 'complete'  # clustering algo param: 'complete' or 'average'

    # optional vol period params
    include_market_vol_periods: bool = False
    vol_period_count: int = 3  # number of high vol periods to include/up-weight in distance matrix
    vol_window: int = 30  # length of window for vol period calculation
    vol_start_date: str = '2020-01-01'  # start date for vol period inclusion

    # recent period weighting
    recent_period_days: int = 365  # number of days to consider as recent period


def prepare_return_data(df: pd.DataFrame) -> pd.DataFrame:
    """get relevant return cols as inputs for clustering"""
    return df.loc[:, ['date', 'symbol', 'return', 'return_mktadj', 'alpha_ret']].copy()


def find_highest_vol_periods(index_rets, n=10, window=30, start_date='2020-01-01') -> List[pd.Timestamp]:
    """
    find the n highest volatility periods across the market indicies (e.g. average rolling vol of SPY, XLP, XLY)
    """
    temp_index_rets = index_rets.loc[start_date:].copy()
    vol_periods = []
    for i in range(n):
        vol_period = temp_index_rets.rolling(window, min_periods=10).std().sum(axis=1).nlargest(n).index[0]
        vol_periods.append(vol_period)
        # remove that event
        temp_index_rets = temp_index_rets.loc[(temp_index_rets.index < vol_period - pd.Timedelta(days=3 * window)) |
                                              (temp_index_rets.index > vol_period + pd.Timedelta(days=2* window))]

    # consider looking into / adding alpha vol periods for each symbol
    # -- i.e., when it experienced high alpha vol, how were other names trading?
    #  -- this may be driven by out of model factors, spurious correlations, etc. but it's worth a look

    return vol_periods


def get_distance_matrix(df: pd.DataFrame, return_type: str = 'return', force_symbols: list = None) -> pd.DataFrame:
    """
    pivot df by date x symbol -> returns, then compute distance = 1 - corr
    """
    pivoted = df.pivot(index='date', columns='symbol', values=return_type)
    corr = pivoted.corr()

    if force_symbols is not None:
        missing_symbols = set(force_symbols) - set(corr.columns)
        for sym in missing_symbols:
            corr[sym] = 0
            corr.loc[sym] = 0
        corr.fillna(0, inplace=True)

    dist = 1 - corr
    return dist


def calculate_distance_matrices(
        df_rets: pd.DataFrame,
        params: ClusteringParams,
        index_rets: Optional[pd.DataFrame] = None,
        symbols: Optional[list] = None
) -> pd.DataFrame:
    """handles all distance matrix calculations"""

    # calculate full period and recent period distance matrices
    dist_full = get_distance_matrix(df_rets, params.return_type)
    recent_mask = df_rets.date >= df_rets.date.max() - pd.Timedelta(days=params.recent_period_days)
    dist_last = get_distance_matrix(df_rets.loc[recent_mask], params.return_type)

    if not params.include_market_vol_periods or index_rets is None:
        return (dist_full + dist_last) / 2

    # calculate distance matrices for volatile periods
    vol_periods = find_highest_vol_periods(
        index_rets,
        n=params.vol_period_count,
        window=params.vol_window,
        start_date=params.vol_start_date
    )

    dist_market_vols = {}
    for vol_period in vol_periods:
        period_mask = (
                (df_rets.date >= vol_period - pd.Timedelta(days=2 * params.vol_window)) &
                (df_rets.date <= vol_period + pd.Timedelta(days=params.vol_window))
        )
        temp_df_rets = df_rets.loc[period_mask].copy()
        dist_market_vols[vol_period] = get_distance_matrix(
            temp_df_rets,
            params.return_type,
            force_symbols=symbols
        )

    # our distance matrix input for clustering will be the simple average of all distance matrices
    return (dist_full + dist_last + sum(dist_market_vols.values())) / (2 + len(dist_market_vols))


def perform_clustering(
        distance_matrix: pd.DataFrame,
        params: ClusteringParams
) -> pd.DataFrame:
    """execute clustering algorithm"""

    num_symbols = len(distance_matrix.columns)
    num_clusters = int(num_symbols / params.avg_names_per_group)

    # prep distance matrix for clustering
    dist_values = distance_matrix.fillna(distance_matrix.mean()).values

    # perform clustering
    clusterer = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric='precomputed',  # param used to be 'affinity', not 'metric'
        linkage=params.linkage
    )
    labels = clusterer.fit_predict(dist_values)

    # create cluster assignments
    return pd.DataFrame({
        'symbol': distance_matrix.columns,
        'cluster': labels
    })


def format_cluster_output(cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    """format cluster assignments into a readable dataframe"""

    largest_cluster = cluster_assignments.cluster.value_counts().max()
    cluster_df = pd.DataFrame()

    for cluster in cluster_assignments.cluster.unique():
        cluster_symbols = cluster_assignments.loc[
            cluster_assignments.cluster == cluster
            ].symbol.tolist()

        # pad with empty strings to make all columns same length
        cluster_symbols.extend([''] * (largest_cluster - len(cluster_symbols)))
        cluster_df[f'Cluster {cluster:03d}'] = cluster_symbols

    return cluster_df.loc[:, sorted(cluster_df.columns)]


def run_clustering_analysis(
        df: pd.DataFrame,
        params: ClusteringParams,
        index_rets: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    main workflow function that orchestrates the process

    return: dict containing key dataframes from the analysis
    """

    # prepare return data
    df_rets = prepare_return_data(df)

    # calculate distance matrices and combine
    dist_avg = calculate_distance_matrices(df_rets, params, index_rets)

    # cluster symbols based on distance martix
    cluster_assignments = perform_clustering(dist_avg, params)

    # format output
    cluster_df = format_cluster_output(cluster_assignments)

    # save results if path provided
    if output_path:
        cluster_df.to_excel(output_path, index=False)

    # return key dataframes
    return {
        'cluster_df': cluster_df,
        'cluster_assignments': cluster_assignments,
        'distance_matrix': dist_avg
    }


if __name__ == '__main__':

    # ============ DEFINE DATA PULL PARAMS ============ #
    data_params = DataParams(
        sector_etfs=['XLP', 'XLY'],  # must be cached in DATA_DIRECTORY, e.g. XLY.csv -- download .csv from SPDR website
        start_date='2016-01-01'
    )

    # ============ DEFINE CLUSTERING PARAMETERS ============ #
    params = ClusteringParams(
        return_type='alpha_ret',  # alpha_ret, return, return_mktadj
        avg_names_per_group=5,  # want somewhat tight groups
        include_market_vol_periods=False  # option to up-weight relationships in high vol periods
    )

    # ============ PREPARE DATA ============ #
    df, index_rets = prepare_clustering_data(
        data_params=data_params,
        data_directory=Path(DATA_DIRECTORY),
        force_refresh=False
    )

    # ============ RUN ANALYSIS ============ #
    results = run_clustering_analysis(
        df=df,
        params=params,
        index_rets=index_rets,
        output_path=f'{DATA_DIRECTORY}groups.xlsx'
    )

    # ============ RESULTS ============ #
    cluster_df = results['cluster_df']
    cluster_assignments = results['cluster_assignments']
    distance_matrix = results['distance_matrix']
