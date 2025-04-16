import os
import io
import requests
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging
from matplotlib.figure import Figure
from dateutil.relativedelta import relativedelta
import base64
from io import BytesIO

''' =======================================================================================================
    Config
======================================================================================================= '''

try:  # Try to import local config first (for local development)
    import config
    VANTAGE_API_KEY = config.VANTAGE_API_KEY
    SENDER_EMAIL = config.SENDER_EMAIL
    EMAIL_APP_PASSWORD = config.EMAIL_APP_PASSWORD
    RECIPIENT_EMAIL = config.RECIPIENT_EMAIL
except ImportError:  # Fall back to environment variables (for GitHub Actions) -- repo secrets from .yml file env
    VANTAGE_API_KEY = os.environ.get('VANTAGE_API_KEY')
    SENDER_EMAIL = os.environ.get('SENDER_EMAIL')
    EMAIL_APP_PASSWORD = os.environ.get('EMAIL_APP_PASSWORD')
    RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL')

    # Validate required environment variables
    required_vars = ['VANTAGE_API_KEY', 'SENDER_EMAIL', 'EMAIL_APP_PASSWORD', 'RECIPIENT_EMAIL']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

''' =======================================================================================================
    Data Fetching Functions
======================================================================================================= '''


def fetch_alpha_vantage_prices(symbol: str, output_size: str = 'full') -> pd.DataFrame:
    """
    Fetch daily stock data (adj. close prices) from Alpha Vantage API.

    Parameters
    ----------
    symbol : str
        The stock symbol to fetch data for.
    output_size : str, optional
        The amount of data to fetch, by default 'full' (up to 20 years).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the historical stock data.
        Index: pd.DatetimeIndex
        Columns: ['adj_close', 'volume', 'date']
    """
    logger.info(f"Fetching price data for {symbol}...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={output_size}&apikey={VANTAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=30)  # Add timeout
        r.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = r.json()

        if "Error Message" in data:
            logger.error(f"API Error fetching price data for {symbol}: {data['Error Message']}")
            return pd.DataFrame()
        if "Time Series (Daily)" not in data:
            # Handle potential rate limiting message or other unexpected formats
            if "Information" in data:
                 logger.warning(f"API Info for {symbol}: {data['Information']}. Might be rate limited.")
            else:
                logger.error(f"Unexpected price response format for {symbol}: {data}")
            return pd.DataFrame()

        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = df.apply(pd.to_numeric)
        df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adj_close',
            '6. volume': 'volume',
        }, inplace=True)
        df = df[['adj_close', 'volume']].copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df['date'] = df.index
        logger.info(f"Successfully fetched price data for {symbol}, {len(df)} rows")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching price data for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching price data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_alpha_vantage_overviews(symbol: str) -> pd.DataFrame:
    """
    Fetch stock company overview data from Alpha Vantage API.

    Parameters
    ----------
    symbol : str
        The stock symbol to fetch data for.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the company data.
        Index: Symbol
        Columns: Name, Sector, MarketCapitalization, AnalystTargetPrice, EPS,
                 TrailingPE, ForwardPE, PEGRatio, QuarterlyEarningsGrowthYOY, EVToEBITDA, PriceToSalesRatioTTM,
                 ProfitMargin, OperatingMarginTTM, ReturnOnAssetsTTM, ReturnOnEquityTTM,
                 RevenueTTM, GrossProfitTTM
    """
    logger.info(f"Fetching overview data for {symbol}...")
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={VANTAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=30)  # Add timeout
        r.raise_for_status()
        data = r.json()

        if "Error Message" in data:
            logger.error(f"API Error fetching overview for {symbol}: {data['Error Message']}")
            return pd.DataFrame()
        if not data or 'Symbol' not in data:  # Check if data is empty or lacks essential key
            if "Information" in data:
                logger.warning(f"API Info for {symbol}: {data['Information']}. Might be rate limited.")
            else:
                logger.error(f"Unexpected or empty overview response for {symbol}: {data}")
            return pd.DataFrame()

        id_cols = ['Symbol', 'Name', 'Sector']
        numeric_cols = ['MarketCapitalization', 'AnalystTargetPrice', 'EPS',
                        'TrailingPE', 'ForwardPE', 'PEGRatio', 'QuarterlyEarningsGrowthYOY',
                        'EVToEBITDA', 'PriceToSalesRatioTTM', 'ProfitMargin', 'OperatingMarginTTM',
                        'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM', 'GrossProfitTTM']

        # Ensure all expected columns exist in the response, fill with None if missing
        overview_data = {col: data.get(col) for col in id_cols + numeric_cols}

        df = pd.DataFrame([overview_data])  # Create DataFrame from dict
        df.set_index('Symbol', inplace=True)

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' turns errors into NaT/NaN

        logger.info(f"Successfully fetched overview data for {symbol}")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching overview data for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching overview data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_realtime_bulk_quotes(symbols: List[str]) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[float]]]:
    """
    Fetch realtime bulk quotes from Alpha Vantage for a list of symbols.

    Prioritizes 'close' price (live market price) if available,
    otherwise uses 'extended_hours_quote'.

    Parameters
    ----------
    symbols : List[str]
        List of stock symbols (max 100 per call recommended by Alpha Vantage).

    Returns
    -------
    Dict[str, Tuple[Optional[pd.Timestamp], Optional[float]]]
        Dictionary mapping each symbol to a tuple containing:
        (Timestamp of the quote (Eastern Time), Price).
        Returns (None, None) if no valid quote found for a symbol.
    """
    logger.info(f"Fetching realtime bulk quotes for {len(symbols)} symbols...")
    live_quotes = {}
    # Alpha Vantage suggests max 100 symbols per bulk request
    symbols_string = ','.join(symbols)
    url = f'https://www.alphavantage.co/query?function=REALTIME_BULK_QUOTES&symbol={symbols_string}&apikey={VANTAGE_API_KEY}'

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "Error Message" in data:
            logger.error(f"API Error fetching bulk quotes: {data['Error Message']}")
            return {s: (None, None) for s in symbols} # Return None for all on error
        if "data" not in data or not data["data"]:
            logger.warning(f"No 'data' field in bulk quote response or empty: {data}")
            return {s: (None, None) for s in symbols}

        quote_df = pd.DataFrame(data['data'])
        # Convert relevant columns, coercing errors
        quote_df['timestamp'] = pd.to_datetime(quote_df['timestamp'], errors='coerce')
        numeric_cols = ['close', 'extended_hours_quote']
        for col in numeric_cols:
            quote_df[col] = pd.to_numeric(quote_df[col], errors='coerce')

        for index, row in quote_df.iterrows():
            symbol = row['symbol']
            timestamp = row['timestamp']
            live_price = row['close']
            extended_price = row['extended_hours_quote']

            price_to_use = None
            if pd.notna(live_price):
                price_to_use = live_price
                logger.debug(f"Using live quote for {symbol}: {price_to_use} at {timestamp}")
            elif pd.notna(extended_price):
                price_to_use = extended_price
                logger.debug(f"Using extended hours quote for {symbol}: {price_to_use} at {timestamp}")
            else:
                logger.warning(f"No valid live or extended quote found for {symbol}.")

            if pd.notna(timestamp) and price_to_use is not None:
                live_quotes[symbol] = (timestamp, price_to_use)
            else:
                live_quotes[symbol] = (None, None) # Store None if data invalid

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching bulk quotes: {e}")
        return {s: (None, None) for s in symbols}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching bulk quotes: {e}")
        return {s: (None, None) for s in symbols}

    # Ensure all requested symbols have an entry, even if fetching failed for them
    for s in symbols:
        if s not in live_quotes:
            live_quotes[s] = (None, None)
            logger.warning(f"Symbol {s} not found in bulk quote response.")

    logger.info(f"Finished fetching realtime quotes. Found data for {len([q for q in live_quotes.values() if q[0] is not None])} symbols.")
    return live_quotes


''' =======================================================================================================
    Calculation Functions
======================================================================================================= '''


def calculate_trailing_returns(prices: pd.DataFrame, column: str = 'adj_close') -> pd.Series:
    """
    Calculates trailing returns for various periods.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with a datetime index and a price column (e.g., 'adj_close').
    column : str, optional
        The column name containing the prices, by default 'adj_close'.

    Returns
    -------
    pd.Series
        Series containing trailing returns for 1d, 1w, 1m, 3m, 6m, 1y, 2y.
        Index: ['Return_1d', 'Return_1w', 'Return_1m', 'Return_3m', 'Return_6m', 'Return_1y', 'Return_2y']
    """
    if prices.empty or column not in prices.columns or len(prices) < 2:
        logger.warning(f"Cannot calculate returns for empty or insufficient price data. Column: {column}")
        return pd.Series(index=['Return_1d', 'Return_1w', 'Return_1m', 'Return_3m', 'Return_6m', 'Return_1y', 'Return_2y'], dtype=float)

    returns = {}
    periods = {'1d': 1, '1w': 5, '1m': 21, '3m': 63, '6m': 126, '1y': 252, '2y': 504}
    latest_price = prices[column].iloc[-1]

    for name, days in periods.items():
        if len(prices) > days:
            past_price = prices[column].iloc[-days-1]
            if pd.notna(past_price) and past_price != 0:
                returns[f'Return_{name}'] = (latest_price / past_price) - 1
            else:
                returns[f'Return_{name}'] = np.nan  # Avoid division by zero or NaN
        else:
            returns[f'Return_{name}'] = np.nan  # Not enough data

    return pd.Series(returns)


def calculate_rsi(prices: pd.DataFrame, window: int = 14, column: str = 'adj_close') -> Optional[float]:
    """
    Calculates the Relative Strength Index (RSI).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with a datetime index and a price column (e.g., 'adj_close').
    window : int, optional
        The lookback period for RSI calculation, by default 14.
    column : str, optional
        The column name containing the prices, by default 'adj_close'.

    Returns
    -------
    Optional[float]
        The latest RSI value, or None if calculation is not possible.
    """
    if prices.empty or column not in prices.columns or len(prices) < window + 1:
        logger.warning(f"Cannot calculate RSI for empty or insufficient price data. Required: {window+1} points, Have: {len(prices)}. Column: {column}")
        return None

    delta = prices[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    if rsi.empty:
        return None

    latest_rsi = rsi.iloc[-1]
    return latest_rsi if pd.notna(latest_rsi) else None


def calculate_moving_averages(prices: pd.DataFrame, column: str = 'adj_close') -> pd.Series:
    """
    Calculates moving averages for 50, 100, and 200 days.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with a datetime index and a price column (e.g., 'adj_close').
    column : str, optional
        The column name containing the prices, by default 'adj_close'.

    Returns
    -------
    pd.Series
        Series containing the latest moving average values.
        Index: ['DMA_50', 'DMA_100', 'DMA_200']
    """
    if prices.empty or column not in prices.columns:
        logger.warning(f"Cannot calculate moving averages for empty price data. Column: {column}")
        return pd.Series(index=['DMA_50', 'DMA_100', 'DMA_200'], dtype=float)

    mas = {}
    windows = {'50': 50, '100': 100, '200': 200}

    for name, period in windows.items():
        if len(prices) >= period:
            ma = prices[column].rolling(window=period).mean().iloc[-1]
            mas[f'DMA_{name}'] = ma
        else:
            mas[f'DMA_{name}'] = np.nan

    return pd.Series(mas)


def calculate_high_water_mark_and_drawdown(prices: pd.DataFrame, lookback_years: int = 2, column: str = 'adj_close') -> pd.Series:
    """
    Calculates the high-water mark price, date, and current drawdown over a lookback period.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with a datetime index and a price column (e.g., 'adj_close').
    lookback_years : int, optional
        Number of years to look back for the high-water mark, by default 2.
    column : str, optional
        The column name containing the prices, by default 'adj_close'.

    Returns
    -------
    pd.Series
        Series containing high-water mark details.
        Index: ['LastPrice', 'HighWaterMarkPrice', 'HighWaterMarkDate', 'CurrentDrawdown']
    """
    if prices.empty or column not in prices.columns or prices.index.empty:
        logger.warning(f"Cannot calculate HWM/Drawdown for empty price data. Column: {column}")
        return pd.Series(index=['LastPrice', 'HighWaterMarkPrice', 'HighWaterMarkDate', 'CurrentDrawdown'], dtype=object) # Use object for date

    end_date = prices.index.max()
    start_date = end_date - relativedelta(years=lookback_years)
    relevant_prices = prices.loc[start_date:, column].dropna()

    if relevant_prices.empty:
        logger.warning(f"No price data found in the last {lookback_years} years for HWM/Drawdown calculation.")
        return pd.Series(index=['LastPrice', 'HighWaterMarkPrice', 'HighWaterMarkDate', 'CurrentDrawdown'], dtype=object)

    last_price = relevant_prices.iloc[-1]
    high_water_mark_price = relevant_prices.max()
    high_water_mark_date = relevant_prices.idxmax()
    current_drawdown = (last_price / high_water_mark_price) - 1 if high_water_mark_price > 0 else np.nan

    return pd.Series({
        'LastPrice': last_price,
        'HighWaterMarkPrice': high_water_mark_price,
        'HighWaterMarkDate': high_water_mark_date,
        'CurrentDrawdown': current_drawdown
    })


def calculate_correlation_matrix(price_data: Dict[str, pd.DataFrame], period: str = '1y', column: str = 'adj_close') -> Optional[pd.DataFrame]:
    """
    Calculates the correlation matrix of returns for a set of tickers over a specified period.

    Parameters
    ----------
    price_data : Dict[str, pd.DataFrame]
        Dictionary where keys are symbols and values are price DataFrames.
    period : str, optional
        The return period to use for correlation ('1d', '1w', '1m', etc.), by default '1y'.
        Alternatively, can specify number of days like '252d'.
    column : str, optional
        The price column to use, by default 'adj_close'.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame containing the correlation matrix, or None if calculation fails.
        Index: Symbols
        Columns: Symbols
    """
    logger.info(f"Calculating correlation matrix for period: {period}")
    returns_dfs = {}
    num_days = 0

    # Determine number of days for the period
    period_map = {'1d': 1, '1w': 5, '1m': 21, '3m': 63, '6m': 126, '1y': 252, '2y': 504}
    if period in period_map:
        num_days = period_map[period]
    elif period.endswith('d'):
         try:
             num_days = int(period[:-1])
         except ValueError:
             logger.error(f"Invalid period format for correlation: {period}")
             return None
    else:
         logger.error(f"Invalid period specified for correlation: {period}")
         return None

    if num_days <= 0:
        logger.error(f"Number of days for correlation must be positive: {num_days}")
        return None

    # Calculate returns for each ticker
    valid_symbols = []
    for symbol, df in price_data.items():
        if df.empty or column not in df.columns or len(df) <= num_days:
            logger.warning(f"Skipping {symbol} for correlation: insufficient data (need > {num_days}, have {len(df)}).")
            continue
        # Calculate percentage change over the period
        period_return = df[column].pct_change(periods=num_days).dropna()
        if not period_return.empty:
            returns_dfs[symbol] = period_return
            valid_symbols.append(symbol)
        else:
            logger.warning(f"Skipping {symbol} for correlation: calculated returns are empty.")

    if len(returns_dfs) < 2:
        logger.warning("Need at least two valid tickers to calculate correlation matrix.")
        return None

    # Combine returns into a single DataFrame, aligning by date
    combined_returns = pd.concat(returns_dfs, axis=1).dropna()  # Drop rows where any ticker has NaN return

    if combined_returns.empty:
        logger.warning("Combined returns DataFrame is empty after alignment and dropping NaNs.")
        return None

    correlation_matrix = combined_returns.corr()

    return correlation_matrix


''' =======================================================================================================
    Filtering Logic
======================================================================================================= '''


# todo: decide if we want to use this script as a screener rather than a summary of input tickers
def filter_tickers_for_summary(all_data: pd.DataFrame, metric: str, n_top: int = 10, n_bottom: int = 10) -> List[str]:
    """
    Placeholder function to select tickers for focused tables/plots.
    Currently returns all tickers. Implement specific logic here (e.g., top/bottom movers).

    Parameters
    ----------
    all_data : pd.DataFrame
        DataFrame containing combined data for all tickers (prices, fundamentals, calculated metrics).
        Index: Symbol
    metric : str
        The column name in `all_data` to sort by for selecting top/bottom tickers (e.g., 'Return_1w').
    n_top : int, optional
        Number of top performers to select, by default 10.
    n_bottom : int, optional
        Number of bottom performers to select, by default 10.

    Returns
    -------
    List[str]
        List of symbols selected for the summary.
    """
    logger.info(f"Filtering tickers based on metric: {metric} (Top {n_top}, Bottom {n_bottom})")

    if all_data.empty or metric not in all_data.columns:
        logger.warning(f"Cannot filter tickers. DataFrame empty or metric '{metric}' not found.")
        return all_data.index.tolist()  # Return all available tickers if filtering fails

    # Ensure metric column is numeric for sorting
    if not pd.api.types.is_numeric_dtype(all_data[metric]):
         logger.warning(f"Metric column '{metric}' is not numeric. Cannot sort for filtering.")
         return all_data.index.tolist()

    # Sort and select
    sorted_data = all_data.sort_values(by=metric, ascending=False, na_position='last')

    top_tickers = sorted_data.head(n_top).index.tolist()
    # Ensure we don't select the same tickers if n_top + n_bottom > total tickers
    bottom_tickers = sorted_data.tail(n_bottom).index.tolist()

    # Combine and remove duplicates, maintaining some order
    selected_tickers = list(dict.fromkeys(top_tickers + bottom_tickers))

    logger.info(f"Selected {len(selected_tickers)} tickers after filtering.")

    # --- For now, return all tickers until specific logic is decided ---
    # return all_data.index.tolist()
    return selected_tickers


''' =======================================================================================================
    Plotting Functions
======================================================================================================= '''


def fig_to_base64(fig: Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string for embedding in HTML.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')  # Use bbox_inches='tight'
    plt.close(fig) # Close the figure to free memory
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


def plot_rsi_vs_target_return(df: pd.DataFrame, sector_colors: Dict[str, str]) -> Optional[Figure]:
    """
    Generates a scatter plot of RSI vs. Analyst Target Price Return.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'RSI', 'AnalystTargetPrice', 'LastPrice', 'Sector', 'Name'. Index should be Symbol.
    sector_colors : Dict[str, str]
        Mapping of Sector names to color codes.

    Returns
    -------
    Optional[Figure]
        Matplotlib Figure object, or None if data is insufficient.
    """
    required_cols = ['RSI', 'AnalystTargetPrice', 'LastPrice', 'Sector', 'Name']
    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing required columns for RSI vs Target Return plot.")
        return None

    plot_df = df[required_cols].dropna()
    if plot_df.empty:
        logger.warning("No valid data points for RSI vs Target Return plot after dropping NaNs.")
        return None

    plot_df['TargetReturn'] = (plot_df['AnalystTargetPrice'] / plot_df['LastPrice']) - 1

    # Filter out extreme outliers for better visualization (optional)
    plot_df = plot_df[plot_df['TargetReturn'].between(-1, 5)] # Example filter

    if plot_df.empty:
        logger.warning("No valid data points remain after filtering for RSI vs Target Return plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign colors based on sector
    colors = plot_df['Sector'].map(sector_colors).fillna('grey') # Use grey for missing sectors

    scatter = ax.scatter(plot_df['RSI'], plot_df['TargetReturn'], c=colors, alpha=0.7)

    # Annotate points with symbol
    for i, symbol in enumerate(plot_df.index):
        ax.annotate(symbol, (plot_df['RSI'].iloc[i], plot_df['TargetReturn'].iloc[i]), fontsize=8, alpha=0.8)

    ax.set_xlabel('RSI (14-day)')
    ax.set_ylabel('Analyst Target Price Return (%)')
    ax.set_title('RSI vs. Analyst Target Price Return by Sector')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Format y-axis as percentage
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create a legend for sectors
    # Use only sectors present in the plot_df
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=sector,
                           markerfacecolor=color, markersize=10)
               for sector, color in sector_colors.items() if sector in plot_df['Sector'].unique()]
    if handles: # Only add legend if there are handles
       ax.legend(handles=handles, title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    return fig


def plot_recent_return_heatmap(df: pd.DataFrame) -> Optional[Figure]:
    """
    Generates a heatmap of recent trailing returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing return columns (e.g., 'Return_1d', 'Return_1w', ...). Index should be Symbol.

    Returns
    -------
    Optional[Figure]
        Matplotlib Figure object, or None if data is insufficient.
    """
    return_cols = ['Return_1d', 'Return_1w', 'Return_1m', 'Return_3m', 'Return_6m', 'Return_1y', 'Return_2y']
    plot_df = df[[col for col in return_cols if col in df.columns]].dropna(how='all')  # Keep rows with at least one return

    # Ensure data is numeric before plotting heatmap
    plot_df = plot_df.apply(pd.to_numeric, errors='coerce')

    if plot_df.empty:
        logger.warning("No valid data for Recent Return Heatmap.")
        return None

    fig, ax = plt.subplots(figsize=(10, max(6, int(len(plot_df) * 0.4))))  # Adjust height based on number of tickers
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["red", "white", "green"])  # Red-White-Green colormap
    center_val = 0 # Center the colormap at 0% return

    # Determine symmetric color limits around 0
    max_abs_val = plot_df.abs().max().max() # Find max absolute return value
    vmin = -max_abs_val if pd.notna(max_abs_val) else -0.1
    vmax = max_abs_val if pd.notna(max_abs_val) else 0.1

    sns.heatmap(plot_df * 100, annot=True, fmt=".1f", cmap=cmap, linewidths=.5, ax=ax,
                cbar_kws={'label': 'Return (%)'}, center=center_val, vmin=vmin, vmax=vmax)
    ax.set_title('Recent Trailing Returns Heatmap')
    ax.set_xlabel('Period')
    ax.set_ylabel('Symbol')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def plot_fundamental_scatter(df: pd.DataFrame, x_metric: str, y_metric: str, sector_colors: Dict[str, str]) -> Optional[Figure]:
    """
    Generates a scatter plot for two fundamental metrics, colored by sector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing x_metric, y_metric, 'Sector', 'Name'. Index should be Symbol.
    x_metric : str
        Column name for the x-axis.
    y_metric : str
        Column name for the y-axis.
    sector_colors : Dict[str, str]
        Mapping of Sector names to color codes.


    Returns
    -------
    Optional[Figure]
        Matplotlib Figure object, or None if data is insufficient.
    """
    required_cols = [x_metric, y_metric, 'Sector', 'Name']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for fundamental scatter plot ({x_metric} vs {y_metric}).")
        return None

    plot_df = df[required_cols].dropna()

    # Optional: Filter outliers if necessary, e.g., based on quantiles
    # q_low = plot_df[x_metric].quantile(0.01)
    # q_high = plot_df[x_metric].quantile(0.99)
    # plot_df = plot_df[plot_df[x_metric].between(q_low, q_high)]
    # Similar filtering for y_metric if needed

    if plot_df.empty:
        logger.warning(f"No valid data points for fundamental scatter plot ({x_metric} vs {y_metric}) after dropping NaNs.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign colors based on sector
    colors = plot_df['Sector'].map(sector_colors).fillna('grey')

    ax.scatter(plot_df[x_metric], plot_df[y_metric], c=colors, alpha=0.7)

    # Annotate points with symbol
    for i, symbol in enumerate(plot_df.index):
        ax.annotate(symbol, (plot_df[x_metric].iloc[i], plot_df[y_metric].iloc[i]), fontsize=8, alpha=0.8)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f'{y_metric} vs. {x_metric} by Sector')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create a legend for sectors
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=sector,
                           markerfacecolor=color, markersize=10)
               for sector, color in sector_colors.items() if sector in plot_df['Sector'].unique()]
    if handles:
        ax.legend(handles=handles, title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout

    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> Optional[Figure]:
    """
    Generates a heatmap of the stock return correlation matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix DataFrame. Index/Columns should be Symbols.

    Returns
    -------
    Optional[Figure]
        Matplotlib Figure object, or None if data is insufficient.
    """
    if corr_matrix is None or corr_matrix.empty or not isinstance(corr_matrix, pd.DataFrame):
        logger.warning("Invalid or empty correlation matrix provided for heatmap.")
        return None
    if len(corr_matrix) < 2:
         logger.warning("Correlation matrix has less than 2 tickers. Cannot plot heatmap.")
         return None

    fig, ax = plt.subplots(figsize=(max(8, int(len(corr_matrix)*0.5)), max(6, int(len(corr_matrix)*0.5))))  # Adjust size
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue-Red colormap

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=.5, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'}, vmin=-1, vmax=1, center=0)
    ax.set_title('Stock Return Correlation Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


''' =======================================================================================================
    HTML Table Generation Functions
======================================================================================================= '''


def format_value(value: Any, format_type: str = 'float') -> str:
    """Helper to format values for HTML tables."""
    if pd.isna(value):
        return "N/A"
    try:
        if format_type == 'percent':
            return f"{value:.2%}"
        elif format_type == 'float':
            return f"{value:.2f}"
        elif format_type == 'integer':
            return f"{int(value):,d}"  # Format as integer with commas
        elif format_type == 'currency':
            return f"${value:,.2f}"  # Format as currency
        elif format_type == 'large_number':  # Format large numbers (e.g., Market Cap)
            if abs(value) >= 1e12:
                return f"${value/1e12:.2f} T"
            elif abs(value) >= 1e9:
                return f"${value/1e9:.2f} B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f} M"
            else:
                 return f"${value:,.0f}"
        elif format_type == 'date':
            if isinstance(value, pd.Timestamp):
                return value.strftime('%Y-%m-%d')
            else:
                return str(value)  # Fallback
        else:
            return str(value)
    except (ValueError, TypeError):
        return str(value)  # Fallback if formatting fails


def create_html_table(df: pd.DataFrame, columns_formats: Dict[str, Tuple[str, str]], title: str) -> str:
    """
    Generates an HTML table from a DataFrame with specified column formatting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert to HTML table. Index is assumed to be Symbol or similar identifier.
    columns_formats : Dict[str, Tuple[str, str]]
        Dictionary mapping DataFrame column names to a tuple: (Display Name, Format Type).
        Format Types: 'percent', 'float', 'integer', 'currency', 'large_number', 'date', 'string'.
    title : str
        Title for the table.

    Returns
    -------
    str
        HTML string representation of the table.
    """
    if df.empty:
        return f"<h2>{title}</h2><p>No data available.</p>"

    html = f"<h2>{title}</h2>\n"
    html += "<table>\n<thead>\n<tr>\n"
    # Add Index Header if index has a name or is meaningful (e.g., 'Symbol')
    if df.index.name:
        html += f"<th>{df.index.name}</th>\n"
    else:
        html += f"<th>Identifier</th>\n" # Default header

    # Add Column Headers
    display_names = [cf[0] for cf in columns_formats.values()]
    html += "".join(f"<th>{name}</th>\n" for name in display_names)
    html += "</tr>\n</thead>\n<tbody>\n"

    # Add Rows
    for index, row in df.iterrows():
        html += "<tr>\n"
        html += f"<td>{index}</td>\n" # Add index value first
        for col, (display_name, fmt) in columns_formats.items():
            if col in row:
                # Add color styling for return columns
                cell_style = ""
                if 'Return' in col or 'Drawdown' in col:
                     value = row[col]
                     if pd.notna(value):
                        if value > 0:
                            cell_style = ' style="color: green;"'
                        elif value < 0:
                            cell_style = ' style="color: red;"'
                html += f"<td{cell_style}>{format_value(row[col], fmt)}</td>\n"
            else:
                html += "<td>N/A</td>\n"  # Column not present in this row
        html += "</tr>\n"

    html += "</tbody>\n</table>\n"
    return html


''' =======================================================================================================
    Email Functions
======================================================================================================= '''


def send_email_report(
        html_content: str,
        image_cids: Dict[str, str], # Dictionary mapping placeholder CID to base64 string
        subject_text: str = 'Daily Ticker Update'
) -> None:
    """
    Sends an email with the market analysis report including embedded images.

    Parameters
    ----------
    html_content : str
        The complete HTML body of the email, including <img src="cid:..."> tags.
    image_cids : Dict[str, str]
        Dictionary where keys are CIDs used in html_content (e.g., 'plot1')
        and values are the base64 encoded image strings.
    """
    logger.info("Preparing email report...")

    msg = MIMEMultipart('related')
    msg['Subject'] = f'{subject_text} - {datetime.now().strftime("%Y-%m-%d")}'
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL

    # Attach HTML part
    msg_alternative = MIMEMultipart('alternative')
    msg.attach(msg_alternative)
    msg_html = MIMEText(html_content, 'html')
    msg_alternative.attach(msg_html)

    # Attach images using CIDs
    for cid, img_base64 in image_cids.items():
        try:
            img_data = base64.b64decode(img_base64)
            img = MIMEImage(img_data, 'png')
            img.add_header('Content-ID', f'<{cid}>')
            img.add_header('Content-Disposition', 'inline', filename=f'{cid}.png') # Helps some clients
            msg.attach(img)
            logger.info(f"Attached image with CID: {cid}")
        except Exception as e:
            logger.error(f"Failed to decode or attach image with CID {cid}: {e}")

    # Send email
    try:
        logger.info("Connecting to SMTP server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_APP_PASSWORD)
        logger.info("Sending email...")
        server.send_message(msg)
        server.quit()
        logger.info("Email sent successfully!")
    except smtplib.SMTPAuthenticationError as e:
         logger.error(f"SMTP Authentication failed: {e}. Check email/password/app password.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


''' =======================================================================================================
    Main Execution Function
======================================================================================================= '''


def main(symbol_list: List[str], market_indices: List[str], update_name: str) -> None:
    """
    Main function to run the analysis and generate the report.
    """
    logger.info("Starting the daily ticker update process...")

    # --- Configuration ---
    # API Rate Limit: 150 calls per minute
    MAX_VANTAGE_API_RPM = 148
    API_CALLS_PER_TICKER = 2

    # Use a smaller list for testing, expand as needed
    INPUT_TICKERS = symbol_list
    MAJOR_INDICES = market_indices
    ALL_SYMBOLS = list(set(INPUT_TICKERS + MAJOR_INDICES))  # Ensure unique symbols

    # Sector Colors (Example - customize as needed)
    SECTOR_COLORS = {
        'TECHNOLOGY': '#1f77b4', 'COMMUNICATION SERVICES': '#ff7f0e', 'HEALTH CARE': '#2ca02c',
        'FINANCIALS': '#d62728', 'CONSUMER CYCLICAL': '#9467bd', 'INDUSTRIALS': '#8c564b',
        'CONSUMER DEFENSIVE': '#e377c2', 'ENERGY': '#7f7f7f', 'REAL ESTATE': '#bcbd22',
        'UTILITIES': '#17becf', 'BASIC MATERIALS': '#aec7e8',
        'MANUFACTURING': '#ff9896',
        'LIFE SCIENCES': '#c5b0d5',
        'TRADE & SERVICES': '#c49c94',
        'FINANCE': '#f7b6d2',
        'ETF': '#ffbb78',  # Example for ETFs like SPY, QQQ
        'OTHER': 'grey'  # Default
    }

    # --- Data Fetching ---
    price_data: Dict[str, pd.DataFrame] = {}
    overview_data: Dict[str, pd.DataFrame] = {}
    failed_symbols: List[str] = []

    logger.info("Fetching real-time quotes for all symbols...")
    latest_quotes = fetch_realtime_bulk_quotes(ALL_SYMBOLS)
    logger.info("Real-time quotes fetched.")

    for symbol in ALL_SYMBOLS:
        logger.info(f"Processing symbol: {symbol}")
        try:
            # Fetch historical price data from Alpha Vantage
            prices = fetch_alpha_vantage_prices(symbol)

            if not isinstance(prices.index, pd.DatetimeIndex):
                if not prices.empty:
                    logger.error(f"Price data for {symbol} does not have a DatetimeIndex. Skipping.")
                    failed_symbols.append(symbol)
                    continue

            # --- Append Live Quote Logic ---
            if symbol in latest_quotes:
                live_timestamp, live_price = latest_quotes[symbol]

                # Proceed only if we have a valid price and historical data exists to compare against
                if live_price is not None and not prices.empty:
                    if not isinstance(live_timestamp, pd.Timestamp):
                        try:
                            live_timestamp = pd.Timestamp(live_timestamp)
                        except Exception as e:
                            logger.warning(f"Could not convert live_timestamp for {symbol} to Timestamp: {live_timestamp}. Error: {e}. Skipping live quote.")
                            live_timestamp = None

                    if live_timestamp:
                        try:
                            live_quote_date = live_timestamp.tz_convert(None).normalize()
                        except TypeError:
                            live_quote_date = live_timestamp.normalize()

                        last_historical_date = prices.index.max().normalize()  # Already DatetimeIndex, just normalize

                        # --- Core Conditional Logic ---
                        if live_quote_date > last_historical_date:
                            logger.info(
                                f"Live quote date {live_quote_date.strftime('%Y-%m-%d')} is newer than last historical date {last_historical_date.strftime('%Y-%m-%d')} for {symbol}.")

                            if live_quote_date not in prices.index:

                                new_row_data = {'adj_close': live_price}

                                for col in prices.columns:
                                    if col not in new_row_data:
                                        new_row_data[col] = np.nan

                                new_row = pd.DataFrame(new_row_data, index=[live_quote_date])
                                new_row.index.name = prices.index.name

                                # Concatenate the historical prices with the new row
                                prices = pd.concat([prices, new_row])
                                prices.sort_index(inplace=True)
                                logger.info(
                                    f"Appended live quote for {symbol} for date {live_quote_date.strftime('%Y-%m-%d')}")
                            else:
                                logger.warning(
                                    f"Skipping live quote append for {symbol} - date {live_quote_date.strftime('%Y-%m-%d')} unexpectedly found in index despite being after max date.")
                        else:
                            logger.info(
                                f"Skipping live quote append for {symbol}. Live quote date {live_quote_date.strftime('%Y-%m-%d')} is not strictly later than the last historical date {last_historical_date.strftime('%Y-%m-%d')}.")
                    else:
                        logger.warning(f"Could not process live_timestamp for {symbol}. Skipping live quote append.")

                elif live_price is None:
                    logger.info(f"No valid live price available for {symbol} in latest_quotes. Skipping append.")

            else:
                logger.info(f"No real-time quote found for {symbol} in the fetched batch.")

            if prices.empty:
                logger.warning(f"Failed to fetch price data for {symbol} and no live quote appended. Skipping.")
                failed_symbols.append(symbol)
                continue

            # Store the price data
            price_data[symbol] = prices

            # Fetch overview data
            overview = fetch_alpha_vantage_overviews(symbol)

            if overview.empty:
                logger.warning(f"Failed to fetch overview data for {symbol}. Proceeding with price data only.")
                # Create a minimal overview entry, setting index name to match convention if possible
                overview_data[symbol] = pd.DataFrame([{'Name': symbol, 'Sector': 'OTHER'}],
                                                     index=pd.Index([symbol], name='symbol'))
            else:
                # Ensure the 'Sector' column exists, default to 'OTHER' if not provided by API
                if 'Sector' not in overview.columns:
                    logger.debug(f"Adding default 'OTHER' Sector for {symbol}")
                    overview['Sector'] = 'OTHER'
                # Ensure index name consistency if possible (optional but good practice)
                if overview.index.name is None:
                    overview.index.name = 'symbol'
                overview_data[symbol] = overview

            # Add small delay to avoid hitting API rate limits
            import time
            delay_seconds = 60.0 / (MAX_VANTAGE_API_RPM / API_CALLS_PER_TICKER)
            logger.debug(f"Sleeping for {delay_seconds:.2f} seconds before next symbol...")
            time.sleep(delay_seconds)

        except Exception as e:
            logger.error(f"An unexpected error occurred processing symbol {symbol}: {e}", exc_info=True)
            failed_symbols.append(symbol)


    # Remove failed symbols from the list to process
    active_symbols = [s for s in ALL_SYMBOLS if s not in failed_symbols]
    logger.info(f"Successfully fetched data for {len(active_symbols)} symbols: {active_symbols}")
    if not active_symbols:
        logger.error("No data fetched successfully. Exiting.")
        return
    if failed_symbols:
        logger.warning(f"Failed to process symbols: {failed_symbols}")

    # --- Combine Overview Data ---
    combined_overview = pd.concat(overview_data.values())
    # Ensure sector colors cover all fetched sectors, defaulting unknowns to 'OTHER' color
    for sector in combined_overview['Sector'].unique():
        if sector not in SECTOR_COLORS:
            logger.warning(f"Sector '{sector}' not in SECTOR_COLORS map. Assigning default color.")
            SECTOR_COLORS[sector] = SECTOR_COLORS.get('OTHER', 'grey')

    # --- Calculations ---
    calculated_metrics: Dict[str, pd.Series] = {}
    for symbol in active_symbols:
        logger.info(f"Calculating metrics for {symbol}...")
        prices = price_data[symbol]
        metrics = pd.Series(dtype=object) # Use object to hold mixed types initially

        # Returns
        returns = calculate_trailing_returns(prices)
        metrics = pd.concat([metrics, returns])

        # RSI
        rsi = calculate_rsi(prices)
        metrics['RSI'] = rsi

        # Moving Averages
        mas = calculate_moving_averages(prices)
        metrics = pd.concat([metrics, mas])

        # High Water Mark & Drawdown
        hwm = calculate_high_water_mark_and_drawdown(prices, lookback_years=2) # Look back 2 years
        metrics = pd.concat([metrics, hwm])

        calculated_metrics[symbol] = metrics

    combined_metrics = pd.DataFrame(calculated_metrics).T # Transpose to have symbols as index

    # --- Combine All Data ---
    # Merge overview and calculated metrics
    # Use outer join to keep all symbols even if one part is missing, though we filter earlier
    all_data = pd.merge(combined_overview, combined_metrics, left_index=True, right_index=True, how='inner') # Inner join ok now as we filter failed symbols

    # Convert return/metric columns to numeric after merge, coercing errors to NaN
    metric_cols_to_convert = ['Return_1d', 'Return_1w', 'Return_1m', 'Return_3m', 'Return_6m', 'Return_1y', 'Return_2y',
                              'RSI', 'DMA_50', 'DMA_100', 'DMA_200', 'LastPrice', 'HighWaterMarkPrice',
                              'CurrentDrawdown']
    for col in metric_cols_to_convert:
        if col in all_data.columns:
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')

    # --- Filtering for Summaries ---
    # Example: Filter based on 1-week return for Top Movers table
    # Adjust metric and n_top/n_bottom as needed for different sections
    top_movers_metric = 'Return_1w'
    filtered_tickers_movers = filter_tickers_for_summary(
         all_data[all_data.index.isin(INPUT_TICKERS)], # Filter only from the input list, not indices
         metric=top_movers_metric,
         n_top=10,
         n_bottom=10
     )
    summary_data_movers = all_data.loc[filtered_tickers_movers]

    # For general plots/tables, maybe use all INPUT_TICKERS if list is small, or apply other filters
    filtered_tickers_general = INPUT_TICKERS # Use all input tickers for now
    summary_data_general = all_data.loc[filtered_tickers_general]

    # --- Generate Plots ---
    plots: Dict[str, Figure] = {}
    logger.info("Generating plots...")

    # Plot 1: RSI vs Target Return
    fig_rsi_target = plot_rsi_vs_target_return(summary_data_general, SECTOR_COLORS)
    if fig_rsi_target:
        plots['rsi_vs_target'] = fig_rsi_target

    # Plot 2: Recent Return Heatmap
    fig_return_heatmap = plot_recent_return_heatmap(summary_data_general)
    if fig_return_heatmap:
        plots['return_heatmap'] = fig_return_heatmap

    # Plot 3: PEGRatio vs ForwardPE
    fig_peg_fpe = plot_fundamental_scatter(summary_data_general, 'ForwardPE', 'PEGRatio', SECTOR_COLORS)
    if fig_peg_fpe:
        plots['peg_vs_fpe'] = fig_peg_fpe

    # Plot 4: ROE vs Profit Margin
    fig_roe_margin = plot_fundamental_scatter(summary_data_general, 'ProfitMargin', 'ReturnOnEquityTTM', SECTOR_COLORS)
    if fig_roe_margin:
        plots['roe_vs_margin'] = fig_roe_margin

    # Plot 5: Correlation Matrix (use only INPUT_TICKERS' price data)
    input_price_data = {sym: price_data[sym] for sym in INPUT_TICKERS if sym in price_data}
    corr_matrix = calculate_correlation_matrix(input_price_data, period='1y') # Use 1-year returns for correlation
    fig_corr_heatmap = plot_correlation_heatmap(corr_matrix)
    if fig_corr_heatmap:
         plots['correlation_heatmap'] = fig_corr_heatmap

    # --- Convert Plots to Base64 for Email Embedding ---
    plot_cids: Dict[str, str] = {}
    for name, fig in plots.items():
        plot_cids[name] = fig_to_base64(fig)
        logger.info(f"Generated base64 for plot: {name}")

    # --- Generate HTML Tables ---
    html_tables: Dict[str, str] = {}
    logger.info("Generating HTML tables...")

    # Table 1: Header Dashboard (Major Indices)
    index_data = all_data.loc[MAJOR_INDICES]
    header_formats = {
        'LastPrice': ('Last Price', 'currency'),
        'Return_1d': ('1 Day', 'percent'),
        'Return_1w': ('1 Week', 'percent'),
        'Return_1m': ('1 Month', 'percent'),
        'Return_3m': ('3 Month', 'percent'),
        'Return_6m': ('6 Month', 'percent'),
        'Return_1y': ('1 Year', 'percent'),
        'Return_2y': ('2 Year', 'percent'),
        'CurrentDrawdown': ('Drawdown (2Y)', 'percent'),
        'RSI': ('RSI (14d)', 'float'),
        'DMA_50': ('50 DMA', 'currency'),
        'DMA_100': ('100 DMA', 'currency'),
        'DMA_200': ('200 DMA', 'currency'),
    }
    html_tables['header_dashboard'] = create_html_table(index_data, header_formats, "Major Index Performance")

    # Table 2: Top Movers (Gainers & Losers based on 1w return)
    movers_data = summary_data_movers.sort_values(by=top_movers_metric, ascending=False)
    gainers = movers_data.head(10)
    losers = movers_data.tail(10)
    if gainers.equals(losers): # Handle cases with few tickers where head and tail overlap
         losers = pd.DataFrame() # Avoid duplicate table if gainers == losers

    top_movers_formats = {
        'Name': ('Name', 'string'),
        'Return_1w': ('1w Return', 'percent'),
        'CurrentDrawdown': ('% from HWM (2Y)', 'percent'),
        'RSI': ('RSI', 'float'),
        'AnalystTargetPrice': ('Analyst Target', 'currency'),
        'TrailingPE': ('Trailing PE', 'float'),
        'PEGRatio': ('PEG Ratio', 'float'),
    }
    html_tables['top_gainers'] = create_html_table(gainers, top_movers_formats, f"Top {len(gainers)} Gainers (Last Week)")
    if not losers.empty:
        html_tables['top_losers'] = create_html_table(losers.sort_values(by=top_movers_metric, ascending=True), top_movers_formats, f"Bottom {len(losers)} Losers (Last Week)")

    # Table 3: Valuation
    valuation_formats = {
        'Name': ('Name', 'string'),
        'LastPrice': ('Last Price', 'currency'),
        'EPS': ('EPS (TTM)', 'float'),
        'TrailingPE': ('Trailing PE', 'float'),
        'ForwardPE': ('Forward PE', 'float'),
        'PEGRatio': ('PEG Ratio', 'float'),
        'QuarterlyEarningsGrowthYOY': ('Quarterly Earnings Growth (YoY)', 'percent'),
    }
    html_tables['valuation'] = create_html_table(summary_data_general, valuation_formats, "Valuation Metrics")

    # Table 4: Efficiency
    efficiency_formats = {
        'Name': ('Name', 'string'),
        'EVToEBITDA': ('EV/EBITDA', 'float'),
        'PriceToSalesRatioTTM': ('Price/Sales (TTM)', 'float'),
        'ProfitMargin': ('Profit Margin', 'percent'),
        'OperatingMarginTTM': ('Operating Margin (TTM)', 'percent'),
    }
    html_tables['efficiency'] = create_html_table(summary_data_general, efficiency_formats, "Efficiency Metrics")

    # Table 5: Returns
    returns_formats = {
        'Name': ('Name', 'string'),
        'ReturnOnAssetsTTM': ('ROA (TTM)', 'percent'),
        'ReturnOnEquityTTM': ('ROE (TTM)', 'percent'),
        'RevenueTTM': ('Revenue (TTM)', 'large_number'),
        'GrossProfitTTM': ('Gross Profit (TTM)', 'large_number'),
    }
    html_tables['returns'] = create_html_table(summary_data_general, returns_formats, "Return & Profitability Metrics")

    # Table 6: Risk / Drawdown
    drawdown_formats = {
        'Name': ('Name', 'string'),
        'LastPrice': ('Last Price', 'currency'),
        'HighWaterMarkPrice': ('Max Price (2Y)', 'currency'),
        # 'HighWaterMarkDate': ('Max Price Date', 'date'), # Optional: Add date
        'CurrentDrawdown': ('Current Drawdown', 'percent'),
        'RSI': ('RSI (14d)', 'float'),
    }
    html_tables['drawdown'] = create_html_table(summary_data_general.sort_values(by='CurrentDrawdown'), drawdown_formats, "Drawdown from 2-Year High")

    # --- Construct Email HTML Body ---
    # Basic structure, can be enhanced with CSS, sections, etc.
    # Note: Complex CSS and interactivity (like collapsible sections) have limited support across email clients.
    # Keep styling relatively simple using inline styles or basic CSS in <style> tag.
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f8f9fa; }}
            h1 {{ color: #1a0dab; border-bottom: 2px solid #1a0dab; padding-bottom: 5px; }}
            h2 {{ color: #343a40; margin-top: 30px; border-bottom: 1px solid #dee2e6; padding-bottom: 3px; }}
            table {{ border-collapse: collapse; width: 95%; margin-bottom: 25px; border: 1px solid #dee2e6; font-size: 14px; background-color: #ffffff; }}
            th, td {{ padding: 10px 12px; text-align: left; border: 1px solid #dee2e6; }}
            th {{ background-color: #e9ecef; font-weight: bold; color: #495057; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .plot-container {{ margin: 25px auto; text-align: center; background-color: #ffffff; padding: 15px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; }}
            .section {{ margin-bottom: 30px; }}
            /* Simple Table of Contents Styling (Optional) */
            .toc {{ margin-bottom: 25px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
            .toc ul {{ list-style: none; padding: 0; margin: 0; }}
            .toc li {{ margin-bottom: 5px; }}
            .toc a {{ text-decoration: none; color: #1a0dab; }}
        </style>
    </head>
    <body>
        <h1>{update_name} - {datetime.now().strftime("%Y-%m-%d")}</h1>

        <div class="toc">
            <h3>Contents</h3>
            <ul>
                <li><a href="#executive_summary">1. Executive Summary</a></li>
                <li><a href="#technical_analysis">2. Technical Analysis</a></li>
                <li><a href="#fundamental_analysis">3. Fundamental Analysis</a></li>
                <li><a href="#risk_analysis">4. Risk Analysis</a></li>
            </ul>
        </div>


        <div class="section" id="executive_summary">
            <h2>1. Executive Summary</h2>
            {html_tables.get('header_dashboard', '<p>Header dashboard data missing.</p>')}
            {html_tables.get('top_gainers', '<p>Top gainers data missing.</p>')}
            {html_tables.get('top_losers', '<p>Top losers data missing.</p>')}
        </div>

        <div class="section" id="technical_analysis">
            <h2>2. Technical Analysis</h2>
            <div class="plot-container">
                <h3>RSI vs Analyst Target Return</h3>
                {'<img src="cid:rsi_vs_target">' if 'rsi_vs_target' in plot_cids else '<p>RSI vs Target Return plot missing.</p>'}
            </div>
            <div class="plot-container">
                <h3>Recent Return Heatmap</h3>
                 {'<img src="cid:return_heatmap">' if 'return_heatmap' in plot_cids else '<p>Return Heatmap plot missing.</p>'}
            </div>
        </div>

        <div class="section" id="fundamental_analysis">
            <h2>3. Fundamental Analysis</h2>
             <div class="plot-container">
                <h3>PEGRatio vs ForwardPE</h3>
                 {'<img src="cid:peg_vs_fpe">' if 'peg_vs_fpe' in plot_cids else '<p>PEGRatio vs ForwardPE plot missing.</p>'}
            </div>
             <div class="plot-container">
                <h3>ROE vs Profit Margin</h3>
                 {'<img src="cid:roe_vs_margin">' if 'roe_vs_margin' in plot_cids else '<p>ROE vs Profit Margin plot missing.</p>'}
            </div>
            {html_tables.get('valuation', '<p>Valuation table data missing.</p>')}
            {html_tables.get('efficiency', '<p>Efficiency table data missing.</p>')}
            {html_tables.get('returns', '<p>Returns table data missing.</p>')}
        </div>


        <div class="section" id="risk_analysis">
            <h2>4. Risk Analysis</h2>
             {html_tables.get('drawdown', '<p>Drawdown table data missing.</p>')}
             <div class="plot-container">
                <h3>Correlation Matrix (1Y Returns)</h3>
                 {'<img src="cid:correlation_heatmap">' if 'correlation_heatmap' in plot_cids else '<p>Correlation Heatmap plot missing.</p>'}
            </div>
        </div>

        <p style="font-size: 12px; color: #6c757d; text-align: center;">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Bruce McNair. Data sourced from Alpha Vantage. Financial data is for informational purposes only.
        </p>

    </body>
    </html>
    """

    # --- Send Email ---
    if not SENDER_EMAIL or not EMAIL_APP_PASSWORD or not RECIPIENT_EMAIL:
         logger.error("Email credentials or recipient not configured. Skipping email sending.")
    else:
        send_email_report(html_body, plot_cids, subject_text=update_name)


''' =======================================================================================================
    Execute Main Function
======================================================================================================= '''


if __name__ == "__main__":

    # Index ETFs to include in the daily update
    major_indices = ['SPY', 'QQQ', 'IWM', 'AGG', 'GLD']

    # Stocks to cover in the daily update(s)
    coverage_set = ['AMZN', 'GOOG', 'META', 'BRK-B', 'NVDA', 'TSM', 'JNJ',
                    'MELI', 'MSFT', 'WMT', 'SE',
                    'NFLX', 'AAPL', 'ASML',
                    'PM', 'ABT', 'INTC', 'TSLA', 'CDNS', 'PG', 'CAT']
    watchlist_set = ['TTWO', 'CRWD', 'DE', 'KR', 'UNH', 'ALK', 'DD',
                     'AMT', 'CCI',
                     'LMT', 'NOC', 'LHX', 'HII',
                     'V', 'AXP', 'JPM',
                     'GILD', 'MRK',
                     'COP', 'XOM',
                     'COST', 'NKE', 'LLY', 'VZ']

    # todo: add sector ETFs to data pull, market adjust their returns, and use to adjust stock returns
    # todo: for coverage set, add ability to input dictionary w/ share count for each stock to calculate portfolio return

    main(symbol_list=coverage_set, market_indices=major_indices, update_name="Daily Coverage Update")  # update_name is just the subject line of the email
    main(symbol_list=watchlist_set, market_indices=major_indices, update_name="Daily Watchlist Update")

    logger.info("Script execution finished.")