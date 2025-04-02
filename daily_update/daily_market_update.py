import os
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging
from matplotlib.figure import Figure
from dateutil.relativedelta import relativedelta

try:  # Try to import local config first (for local development)
    import config
    VANTAGE_API_KEY = config.VANTAGE_API_KEY
    SENDER_EMAIL = config.SENDER_EMAIL
    EMAIL_APP_PASSWORD = config.EMAIL_APP_PASSWORD
    RECIPIENT_EMAIL = config.RECIPIENT_EMAIL
    MAX_VANTAGE_API_RPM = config.MAX_VANTAGE_API_RPM
except ImportError:  # Fall back to environment variables (for GitHub Actions) -- repo secrets from .yml file env
    VANTAGE_API_KEY = os.environ.get('VANTAGE_API_KEY')
    MAX_VANTAGE_API_RPM = 148
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


def fetch_alpha_vantage_data(symbol: str, output_size: str = 'full') -> pd.DataFrame:
    """
    Fetch daily stock data from Alpha Vantage API.

    Parameters
    ----------
    symbol : str
        The stock symbol to fetch data for.
    output_size : str, optional
        The amount of data to fetch, by default 'full' (up to 20 years)

    Returns
    -------
    pd.DataFrame
        DataFrame containing the historical stock data with datetime index.
    """
    logger.info(f"Fetching data for {symbol}...")

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={output_size}&apikey={VANTAGE_API_KEY}"

    r = requests.get(url)
    data = r.json()

    if "Error Message" in data:
        logger.error(f"Error fetching data for {symbol}: {data['Error Message']}")
        return pd.DataFrame()

    if "Time Series (Daily)" not in data:
        logger.error(f"Unexpected response format for {symbol}: {data}")
        return pd.DataFrame()

    time_series = data["Time Series (Daily)"]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')

    # Convert columns to numeric
    df = df.apply(pd.to_numeric)

    # Rename columns
    df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. adjusted close': 'adj_close',
        '6. volume': 'volume',
        '7. dividend amount': 'dividend',
        '8. split coefficient': 'split'
    }, inplace=True)

    # Add date as column too (for convenience)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)  # Ensure chronological order
    df['date'] = df.index

    logger.info(f"Successfully fetched data for {symbol}, {len(df)} rows")

    import time
    time.sleep(60 / (MAX_VANTAGE_API_RPM))  # Sleep to avoid rate limiting

    return df


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


def filter_data_timeframe(df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    """
    Filter the DataFrame to include only data from the last N years.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    years : int, optional
        Number of years of data to keep, by default 10

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    cutoff_date = datetime.now() - relativedelta(years=years)
    return df[df.index >= cutoff_date]


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 50, 100, and 200-day moving averages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.

    Returns
    -------
    pd.DataFrame
        DataFrame with added moving average columns.
    """
    df_result = df.copy()

    # Calculate moving averages
    df_result['MA50'] = df_result['adj_close'].rolling(window=50).mean()
    df_result['MA100'] = df_result['adj_close'].rolling(window=100).mean()
    df_result['MA200'] = df_result['adj_close'].rolling(window=200).mean()

    return df_result


def calculate_returns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculate various return periods and their historical percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        DataFrame with return calculations for all periods, and a Series with the latest returns.
    """
    df_returns = df.copy()

    # Calculate different period returns
    df_returns['return_1w'] = df_returns['adj_close'].pct_change(periods=5) * 100
    df_returns['return_1m'] = df_returns['adj_close'].pct_change(periods=21) * 100
    df_returns['return_3m'] = df_returns['adj_close'].pct_change(periods=63) * 100
    df_returns['return_6m'] = df_returns['adj_close'].pct_change(periods=126) * 100
    df_returns['return_1y'] = df_returns['adj_close'].pct_change(periods=252) * 100

    # Calculate YTD return
    df_returns['year'] = df_returns.index.year
    current_year = datetime.now().year
    start_of_year = pd.Timestamp(f'{current_year}-01-01')

    if start_of_year in df_returns.index:
        start_of_year_price = df_returns.loc[start_of_year, 'adj_close']
    else:
        # Find the first trading day of the year
        start_of_year_price = df_returns[df_returns.index.year == current_year].iloc[0]['adj_close']

    df_returns['return_ytd'] = (df_returns['adj_close'] / start_of_year_price - 1) * 100

    # Get the latest returns
    latest_returns = df_returns.iloc[-1][
        ['return_ytd', 'return_1w', 'return_1m', 'return_3m', 'return_6m', 'return_1y']]

    return df_returns, latest_returns


def calculate_percentiles(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate percentiles for the latest returns compared to historical returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with return calculations.

    Returns
    -------
    Dict[str, int]
        Dictionary with return periods as keys and percentile values as values.
    """
    result = {}
    periods = {
        'return_ytd': 252,  # Approximate trading days in a year
        'return_1w': 5,
        'return_1m': 21,
        'return_3m': 63,
        'return_6m': 126,
        'return_1y': 252
    }

    for period, days in periods.items():
        # Create non-overlapping returns for historical comparison
        historical_returns = []
        for i in range(0, len(df) - days, days):
            start_idx = i
            end_idx = i + days
            if end_idx < len(df):
                start_price = df.iloc[start_idx]['adj_close']
                end_price = df.iloc[end_idx]['adj_close']
                period_return = (end_price / start_price - 1) * 100
                historical_returns.append(period_return)

        if not historical_returns:
            result[period] = None
            continue

        # Get the latest return value
        latest_return = df.iloc[-1][period]

        # Calculate percentile
        percentile = int(sum(1 for x in historical_returns if x < latest_return) / len(historical_returns) * 100)
        result[period] = percentile

    return result


def calculate_rsi(df: pd.DataFrame, window: int = 14, wilder: bool = False) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    window : int, optional
        Look-back period for RSI calculation, by default 14
    wilder : bool, optional
        If True, use Wilder's smoothing method (slow RSI), by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with added RSI column.
    """
    df_result = df.copy()
    delta = df_result['adj_close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    if wilder:
        # Wilder's smoothing
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    else:
        # Simple moving average
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    col_name = f'RSI_{window}{"_Wilder" if wilder else ""}'
    df_result[col_name] = rsi

    return df_result


def calculate_volatility(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Calculate rolling volatility for specified windows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    windows : List[int]
        List of window sizes (in days) for volatility calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with added volatility columns.
    """
    df_result = df.copy()

    # Daily returns
    df_result['daily_return'] = df_result['adj_close'].pct_change()

    # Calculate volatility for each window
    for window in windows:
        # Annualized volatility
        df_result[f'volatility_{window}d'] = df_result['daily_return'].rolling(window=window).std() * np.sqrt(252) * 100

    return df_result


def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling drawdown and drawdown expressed in terms of volatility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.

    Returns
    -------
    pd.DataFrame
        DataFrame with added drawdown and vol-adjusted drawdown columns.
    """
    df_result = df.copy()

    # Calculate rolling maximum (high-water mark)
    df_result['rolling_max'] = df_result['adj_close'].cummax()

    # Calculate drawdown in percentage
    df_result['drawdown_pct'] = (df_result['adj_close'] / df_result['rolling_max'] - 1) * 100

    # Calculate 252-day rolling volatility for x-vol calculation
    df_result['daily_return'] = df_result['adj_close'].pct_change()
    df_result['volatility_252d'] = df_result['daily_return'].rolling(window=252).std() * np.sqrt(252)

    # Calculate drawdown in terms of volatility
    df_result['drawdown_xvol'] = df_result['drawdown_pct'] / (df_result['volatility_252d'] * 100)

    # Calculate historical percentile of current drawdown
    all_drawdowns = df_result['drawdown_pct'].dropna().values
    current_drawdown = df_result['drawdown_pct'].iloc[-1]

    if len(all_drawdowns) > 0:
        df_result['drawdown_percentile'] = sum(1 for x in all_drawdowns if x < current_drawdown) / len(
            all_drawdowns) * 100
    else:
        df_result['drawdown_percentile'] = None

    # Find the global high-water mark (maximum price) and its date
    hwm_idx = df_result['adj_close'].idxmax()
    df_result['hwm_price'] = df_result.loc[hwm_idx, 'adj_close']
    df_result['hwm_date'] = hwm_idx.strftime('%Y-%m-%d')

    return df_result


def create_price_ma_plot(df: pd.DataFrame, symbol: str, years: int = 2) -> Figure:
    """
    Create a plot of price and moving averages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and moving average data.
    symbol : str
        Stock symbol for title.
    years : int, optional
        Number of years to show in the plot, by default 2

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Filter for the plotting timeframe
    cutoff_date = datetime.now() - relativedelta(years=years)
    plot_df = df[df.index >= cutoff_date].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the last values for annotation
    last_price = plot_df['adj_close'].iloc[-1]
    last_ma50 = plot_df['MA50'].iloc[-1]
    last_ma100 = plot_df['MA100'].iloc[-1]
    last_ma200 = plot_df['MA200'].iloc[-1]

    # Define colors for consistency
    price_color = 'tab:blue'
    ma50_color = 'tab:orange'
    ma100_color = 'tab:green'
    ma200_color = 'tab:red'

    # Plot price and moving averages
    ax.plot(plot_df.index, plot_df['adj_close'], label='Price', linewidth=2, color=price_color)
    ax.plot(plot_df.index, plot_df['MA50'], label='50 DMA', linewidth=1.5, alpha=0.8, color=ma50_color)
    ax.plot(plot_df.index, plot_df['MA100'], label='100 DMA', linewidth=1.5, alpha=0.8, color=ma100_color)
    ax.plot(plot_df.index, plot_df['MA200'], label='200 DMA', linewidth=1.5, alpha=0.8, color=ma200_color)

    # Get x-axis limits and set annotation x position
    x_min, x_max = ax.get_xlim()
    x_annotation = x_max + (x_max - x_min) * 0.01  # Slightly to the right of the plot

    # Add annotations for the last values
    # Calculate vertical offsets to prevent overlap
    y_range = plot_df['adj_close'].max() - plot_df['adj_close'].min()
    offset = y_range * 0.02  # 2% of y-range for offset

    # Add annotations with boxes
    ax.annotate(f"${last_ma50:.2f}", xy=(x_annotation, last_ma50), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=ma50_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"${last_ma100:.2f}", xy=(x_annotation, last_ma100), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=ma100_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"${last_ma200:.2f}", xy=(x_annotation, last_ma200), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=ma200_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"${last_price:.2f}", xy=(x_annotation, last_price), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=price_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    # Extend the x-axis to make room for annotations
    plt.xlim(x_min, x_max + (x_max - x_min) * 0.05)

    # Format the plot
    ax.set_title(f"{symbol} - Price and Moving Averages (Last {years} Years)", fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig


def create_wilder_rsi_plot(df: pd.DataFrame, symbol: str, years: int = 2) -> Figure:
    """
    Create a plot of Wilder's RSI with 70/30 lines.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with RSI data.
    symbol : str
        Stock symbol for title.
    years : int, optional
        Number of years to show in the plot, by default 2

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Filter for the plotting timeframe
    cutoff_date = datetime.now() - relativedelta(years=years)
    plot_df = df[df.index >= cutoff_date].copy()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Get the last value for annotation
    last_rsi = plot_df['RSI_14_Wilder'].iloc[-1]
    rsi_color = 'blue'

    # Plot RSI
    ax.plot(plot_df.index, plot_df['RSI_14_Wilder'], label='RSI (14)', color=rsi_color, linewidth=1.5)

    # Add 70/30 lines
    ax.axhline(y=70, color='r', linestyle='-', alpha=0.5)
    ax.axhline(y=30, color='g', linestyle='-', alpha=0.5)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # Get x-axis limits and set annotation x position
    x_min, x_max = ax.get_xlim()
    x_annotation = x_max + (x_max - x_min) * 0.01  # Slightly to the right of the plot

    # Add annotation for the last RSI value
    ax.annotate(f"{last_rsi:.1f}", xy=(x_annotation, last_rsi), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=rsi_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    # Extend the x-axis to make room for annotations
    plt.xlim(x_min, x_max + (x_max - x_min) * 0.05)

    # Format the plot
    ax.set_title(f"{symbol} - Wilder's RSI (Last {years} Years)", fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RSI', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig


def create_volatility_plot(df: pd.DataFrame, symbol: str, years: int = 2) -> Figure:
    """
    Create a plot of rolling volatility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volatility data.
    symbol : str
        Stock symbol for title.
    years : int, optional
        Number of years to show in the plot, by default 2

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Filter for the plotting timeframe
    cutoff_date = datetime.now() - relativedelta(years=years)
    plot_df = df[df.index >= cutoff_date].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Get the last values for annotation
    last_vol_10d = plot_df['volatility_10d'].iloc[-1]
    last_vol_30d = plot_df['volatility_30d'].iloc[-1]
    last_vol_60d = plot_df['volatility_60d'].iloc[-1]
    last_vol_90d = plot_df['volatility_90d'].iloc[-1]

    # Define colors for consistency
    vol_10d_color = 'tab:blue'
    vol_30d_color = 'tab:orange'
    vol_60d_color = 'tab:green'
    vol_90d_color = 'tab:red'

    # Plot volatilities
    ax.plot(plot_df.index, plot_df['volatility_10d'], label='10-day', linewidth=1.5, alpha=0.8, color=vol_10d_color)
    ax.plot(plot_df.index, plot_df['volatility_30d'], label='30-day', linewidth=1.5, color=vol_30d_color)
    ax.plot(plot_df.index, plot_df['volatility_60d'], label='60-day', linewidth=1.5, color=vol_60d_color)
    ax.plot(plot_df.index, plot_df['volatility_90d'], label='90-day', linewidth=1.5, color=vol_90d_color)

    # Get x-axis limits and set annotation x position
    x_min, x_max = ax.get_xlim()
    x_annotation = x_max + (x_max - x_min) * 0.01  # Slightly to the right of the plot

    # Add annotations for the last values
    ax.annotate(f"{last_vol_10d:.1f}%", xy=(x_annotation, last_vol_10d), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=vol_10d_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"{last_vol_60d:.1f}%", xy=(x_annotation, last_vol_60d), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=vol_60d_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"{last_vol_90d:.1f}%", xy=(x_annotation, last_vol_90d), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=vol_90d_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"{last_vol_30d:.1f}%", xy=(x_annotation, last_vol_30d), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=vol_30d_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    # Extend the x-axis to make room for annotations
    plt.xlim(x_min, x_max + (x_max - x_min) * 0.05)

    # Format the plot
    ax.set_title(f"{symbol} - Rolling Volatility (Annualized, Last {years} Years)", fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig


def create_fast_rsi_plot(df: pd.DataFrame, symbol: str, years: int = 2) -> Figure:
    """
    Create a plot of fast RSI indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with RSI data.
    symbol : str
        Stock symbol for title.
    years : int, optional
        Number of years to show in the plot, by default 2

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Filter for the plotting timeframe
    cutoff_date = datetime.now() - relativedelta(years=years)
    plot_df = df[df.index >= cutoff_date].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Get the last values for annotation
    last_rsi_9 = plot_df['RSI_9'].iloc[-1]
    last_rsi_14 = plot_df['RSI_14'].iloc[-1]
    last_rsi_30 = plot_df['RSI_30'].iloc[-1]

    # Define colors for consistency
    rsi_9_color = 'tab:blue'
    rsi_14_color = 'tab:orange'
    rsi_30_color = 'tab:green'

    # Plot RSIs
    ax.plot(plot_df.index, plot_df['RSI_9'], label='RSI-9', linewidth=1.5, color=rsi_9_color)
    ax.plot(plot_df.index, plot_df['RSI_14'], label='RSI-14', linewidth=1.5, color=rsi_14_color)
    ax.plot(plot_df.index, plot_df['RSI_30'], label='RSI-30', linewidth=1.5, color=rsi_30_color)

    # Add 70/30 lines
    ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    # Get x-axis limits and set annotation x position
    x_min, x_max = ax.get_xlim()
    x_annotation = x_max + (x_max - x_min) * 0.01  # Slightly to the right of the plot

    # Add annotations for the last values
    ax.annotate(f"{last_rsi_9:.1f}", xy=(x_annotation, last_rsi_9), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=rsi_9_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"{last_rsi_30:.1f}", xy=(x_annotation, last_rsi_30), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=rsi_30_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    ax.annotate(f"{last_rsi_14:.1f}", xy=(x_annotation, last_rsi_14), xycoords=('data', 'data'),
                xytext=(5, 0), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc=rsi_14_color, ec="none", alpha=0.8),
                color='white', fontweight='bold', ha='left', va='center')

    # Extend the x-axis to make room for annotations
    plt.xlim(x_min, x_max + (x_max - x_min) * 0.05)

    # Format the plot
    ax.set_title(f"{symbol} - Fast RSI Indicators (Last {years} Years)", fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RSI', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig


def create_drawdown_plot(df: pd.DataFrame, symbol: str, years: int = 2) -> Figure:
    """
    Create a plot of rolling drawdown.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with drawdown data.
    symbol : str
        Stock symbol for title.
    years : int, optional
        Number of years to show in the plot, by default 2

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Filter for the plotting timeframe
    cutoff_date = datetime.now() - relativedelta(years=years)
    plot_df = df[df.index >= cutoff_date].copy()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Get the last values for annotation
    last_drawdown_pct = plot_df['drawdown_pct'].iloc[-1]
    last_drawdown_xvol = plot_df['drawdown_xvol'].iloc[-1]

    # Define colors for consistency
    drawdown_color = 'tab:red'
    xvol_color = 'tab:blue'

    # Plot drawdown percentage
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Drawdown (%)', color=drawdown_color, fontsize=12)
    ax1.plot(plot_df.index, plot_df['drawdown_pct'], color=drawdown_color, label='% Drawdown')
    ax1.tick_params(axis='y', labelcolor=drawdown_color)
    ax1.set_ylim(min(plot_df['drawdown_pct'].min() * 1.1, -5), 2)  # Give some buffer at the bottom

    # Create a second y-axis for drawdown in terms of volatility
    ax2 = ax1.twinx()
    ax2.set_ylabel('Drawdown (x Vol)', color=xvol_color, fontsize=12)
    ax2.plot(plot_df.index, plot_df['drawdown_xvol'], color=xvol_color, linestyle='--', label='x Vol')
    ax2.tick_params(axis='y', labelcolor=xvol_color)

    # Get x-axis limits and set annotation x position
    x_min, x_max = ax1.get_xlim()
    x_annotation = x_max + (x_max - x_min) * 0.01  # Slightly to the right of the plot

    # Add annotations for the last values
    ax1.annotate(f"{last_drawdown_pct:.1f}%", xy=(x_annotation, last_drawdown_pct), xycoords=('data', 'data'),
                 xytext=(5, 0), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc=drawdown_color, ec="none", alpha=0.8),
                 color='white', fontweight='bold', ha='left', va='center')

    # Annotation for the second y-axis
    # We need to transform the y coordinate from ax2's coordinate system to ax1's
    y_min1, y_max1 = ax1.get_ylim()
    y_min2, y_max2 = ax2.get_ylim()

    # Linear transformation from ax2's y-coordinate to ax1's
    y_drawdown_xvol_transformed = y_min1 + (last_drawdown_xvol - y_min2) * (y_max1 - y_min1) / (y_max2 - y_min2)

    ax2.annotate(f"{last_drawdown_xvol:.2f}x", xy=(x_annotation, last_drawdown_xvol), xycoords=('data', 'data'),
                 xytext=(5, 0), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc=xvol_color, ec="none", alpha=0.8),
                 color='white', fontweight='bold', ha='left', va='center')

    # Extend the x-axis to make room for annotations
    ax1.set_xlim(x_min, x_max + (x_max - x_min) * 0.05)

    # Add a title
    plt.title(f"{symbol} - Rolling Drawdown (Last {years} Years)", fontsize=14)

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    # Format the date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Add grid
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def format_value(value: Any, format_type: str = 'float') -> str:
    """Helper to format values for HTML tables."""
    if pd.isna(value):
        return "N/A"
    try:
        if format_type == 'percent':
            # Multiply by 100 only if it's not already a percentage from pct_change
            # Assuming input like 0.05 needs to become 5.00%
            # If pct_change already multiplied by 100, remove the * 100
            return f"{value * 100:.2f}%" # Adjust if your pct_change is already * 100
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
    # Use border=1 cellpadding=5 cellspacing=0 style=border-collapse: collapse; like the other table
    html += "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\" style=\"border-collapse: collapse; width: auto; margin-bottom: 20px;\">\n<thead>\n"
    # Add header row styling
    html += "<tr style=\"background-color: #f2f2f2;\">\n"
    # Add Index Header
    index_header = df.index.name if df.index.name else "Symbol" # Use 'Symbol' if no index name
    html += f"<th>{index_header}</th>\n"
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
                # Add color styling for return/drawdown columns
                cell_style = ""
                # Assuming pct_change values are like 0.01 for 1%
                value = row[col] # Get the raw value
                if pd.notna(value) and ('Return' in col or 'Drawdown' in col):
                    # Check the raw value before formatting
                    if value > 0:
                        cell_style = ' style="color: green;"'
                    elif value < 0:
                        # For Drawdown, negative is expected, maybe don't make it red? Optional.
                        cell_style = ' style="color: red;"'

                # Apply formatting using the helper function
                formatted_val = format_value(value, fmt)
                html += f"<td{cell_style}>{formatted_val}</td>\n"
            else:
                html += "<td>N/A</td>\n" # Column not present in this row
    html += "</tr>\n"

    html += "</tbody>\n</table>\n"
    return html


def create_performance_table(
        symbol: str,
        latest_price: float,
        returns: pd.Series,
        percentiles: Dict[str, int],
        hwm_price: float = None,
        hwm_date: str = None,
        current_drawdown: float = None,
        drawdown_percentile: float = None
) -> str:
    """
    Create an HTML table of performance metrics.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    latest_price : float
        Latest price.
    returns : pd.Series
        Series with return values.
    percentiles : Dict[str, int]
        Dictionary with return percentiles.
    hwm_price : float, optional
        High-water mark price.
    hwm_date : str, optional
        Date of the high-water mark.
    current_drawdown : float, optional
        Current drawdown percentage.
    drawdown_percentile : float, optional
        Percentile of the current drawdown.

    Returns
    -------
    str
        HTML table as a string.
    """
    # Start HTML table
    html = f"""
    <h3>{symbol} Performance Metrics</h3>
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Last Price</td>
            <td>${latest_price:.2f}</td>
        </tr>
    """

    # Add High-Water Mark if available
    if hwm_price is not None and hwm_date is not None:
        html += f"""
        <tr>
            <td>High-Water Mark</td>
            <td>${hwm_price:.2f} ({hwm_date})</td>
        </tr>
        """

    # Add Current Drawdown if available
    if current_drawdown is not None and drawdown_percentile is not None:
        html += f"""
        <tr>
            <td>Current Drawdown</td>
            <td>{current_drawdown:.2f}% ({int(drawdown_percentile)}th)</td>
        </tr>
        """

    # Map of return columns to display names
    return_names = {
        'return_ytd': 'YTD Return',
        'return_1w': '1-Week Return',
        'return_1m': '1-Month Return',
        'return_3m': '3-Month Return',
        'return_6m': '6-Month Return',
        'return_1y': '1-Year Return'
    }

    # Add returns and percentiles
    for key, name in return_names.items():
        value = returns[key]
        percentile = percentiles.get(key)

        html += f"""
        <tr>
            <td>{name}</td>
            <td>{value:.2f}% ({percentile}th)</td>
        </tr>
        """

    # Close table
    html += "</table>"

    return html


def fig_to_base64(fig: Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object.

    Returns
    -------
    str
        Base64 encoded string of the image.
    """
    import base64
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


def send_email_report(
        symbol_data: Dict[str, Dict[str, Any]],
        plots: Dict[str, Dict[str, Figure]],
        summary_table_html: str  # Add new parameter
) -> None:
    """
    Send an email with the market analysis report, including a summary table.
    Parameters
    ----------
    symbol_data : Dict[str, Dict[str, Any]]
        Dictionary containing data for each symbol.
    plots : Dict[str, Dict[str, Figure]]
        Dictionary containing plots for each symbol.
    summary_table_html : str
        HTML string for the summary table to be placed at the top.
    """
    logger.info("Preparing email report...")

    msg = MIMEMultipart('related')
    msg['Subject'] = f'Daily Market Report - {datetime.now().strftime("%Y-%m-%d")}'
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL

    # --- Create the HTML content ---
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            h1 {{ color: #333366; }}
            h2 {{ color: #666699; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: auto; margin-bottom: 20px; border: 1px solid #ddd; }} /* Adjusted width */
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }} /* Adjusted padding */
            th {{ background-color: #f2f2f2; }}
            .plot-container {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Daily Market Report - {datetime.now().strftime("%Y-%m-%d")}</h1>
    """

    # --- Add the summary table at the top ---
    html_content += summary_table_html
    html_content += "<hr>"  # Add a separator

    # Add data for each symbol
    for symbol in symbol_data:
        html_content += f"<h2>{symbol} Analysis</h2>"

        # Add individual performance table (ensure keys match what process_symbol now returns for this section)
        data = symbol_data[symbol]
        # Use the correct keys based on the updated process_symbol return dict
        html_content += create_performance_table(
            symbol,
            data['latest_price'],  # Or 'LastPrice' if you standardized
            data['returns'],  # This holds the specific returns for the individual table
            data['percentiles'],
            data['hwm_price'],
            data['hwm_date'],
            data['current_drawdown_pct_individual'],  # Use the specific key for this table
            data['drawdown_percentile']
        )

        # Add plots
        symbol_plots = plots[symbol]
        plot_names = {
            'price_ma': 'Price and Moving Averages',
            'wilder_rsi': 'Wilder\'s RSI',
            'volatility': 'Rolling Volatility',
            'fast_rsi': 'Fast RSI Indicators',
            'drawdown': 'Rolling Drawdown'
        }

        for plot_key, plot_name in plot_names.items():
            img_str = fig_to_base64(symbol_plots[plot_key])
            img_id = f"{symbol}_{plot_key}"

            html_content += f"""
            <div class="plot-container">
                <h3>{plot_name}</h3>
                <img src="cid:{img_id}" style="width:100%; max-width:800px;">
            </div>
            """

    html_content += """
    </body>
    </html>
    """

    # --- Attach HTML and Images ---
    msg_html = MIMEText(html_content, 'html')
    msg.attach(msg_html)

    # Attach images (ensure plots dictionary is iterated safely)
    for symbol in plots:
        if symbol in plots:  # Double check symbol exists
            symbol_plots = plots[symbol]
            for plot_key, fig in symbol_plots.items():
                if fig is not None:  # Check if figure object is valid
                    img_id = f"{symbol}_{plot_key}"
                    try:
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100)
                        buf.seek(0)

                        img = MIMEImage(buf.read())
                        img.add_header('Content-ID', f'<{img_id}>')
                        msg.attach(img)
                    except Exception as img_err:
                        logger.error(f"Error attaching image {plot_key} for {symbol}: {img_err}")

    # --- Send email ---
    try:
        logger.info("Connecting to SMTP server...")
        # Ensure EMAIL_APP_PASSWORD and SENDER_EMAIL are correctly loaded
        if not SENDER_EMAIL or not EMAIL_APP_PASSWORD:
            logger.error("Sender email or app password not configured. Cannot send email.")
            return

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info("Email sent successfully!")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        logger.exception("Email sending failed with exception:")  # Log traceback


def process_symbol(symbol: str, latest_quotes: Dict[str, Tuple[Optional[pd.Timestamp], Optional[float]]]) -> Tuple[Dict[str, Any], Dict[str, Figure]]:
    """
    Process a single symbol to generate all required data and plots, including appending the latest quote.
    Parameters
    ----------
    symbol : str
        The stock symbol to process.
    latest_quotes : Dict[str, Tuple[Optional[pd.Timestamp], Optional[float]]]
        Dictionary containing the latest fetched real-time quotes.
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Figure]]
        Tuple containing the symbol data and the generated plots.
    """
    # Fetch historical data
    df = fetch_alpha_vantage_data(symbol)
    if df.empty:
        logger.error(f"No historical data fetched for {symbol}, skipping...")
        return {}, {}

    # --- Append Live Quote ---
    if symbol in latest_quotes:
        live_timestamp, live_price = latest_quotes[symbol]

        if live_price is not None and live_timestamp is not None and not df.empty:
            # Ensure live_timestamp is timezone naive and normalized
            live_date = live_timestamp.normalize()
            last_historical_date = df.index.max().normalize()

            # Only append if the live quote date is strictly after the last historical date
            if live_date > last_historical_date:
                logger.info(f"Attempting to append live quote for {symbol} for date {live_date.strftime('%Y-%m-%d')}")
                # Create a new row with the live price
                new_row_data = {'adj_close': live_price}
                # Add NaNs for other columns if they exist in `df`
                for col in df.columns:
                    if col not in new_row_data:
                        new_row_data[col] = np.nan

                new_row = pd.DataFrame(new_row_data, index=[live_date])

                # Check if the date *already exists* (e.g., script ran twice today)
                if live_date not in df.index:
                    # Append the new row
                    df = pd.concat([df, new_row])
                    df.sort_index(inplace=True) # Ensure order is maintained
                    logger.info(f"Successfully appended live quote for {symbol} for date {live_date.strftime('%Y-%m-%d')}")
                else:
                    # Optionally update the existing row for today instead of skipping
                    logger.warning(f"Live quote date {live_date.strftime('%Y-%m-%d')} already exists for {symbol}. Overwriting adj_close.")
                    df.loc[live_date, 'adj_close'] = live_price

            elif live_date == last_historical_date:
                logger.info(f"Live quote date {live_date.strftime('%Y-%m-%d')} matches last historical date for {symbol}. Updating adj_close.")
                df.loc[live_date, 'adj_close'] = live_price
            else:
                logger.warning(f"Live quote for {symbol} ({live_date.strftime('%Y-%m-%d')}) is not newer than last historical date ({last_historical_date.strftime('%Y-%m-%d')}). Skipping append.")
        else:
            logger.warning(f"No valid live quote data found for {symbol} in latest_quotes dictionary.")
    else:
        logger.warning(f"Symbol {symbol} not found in latest_quotes dictionary.")
    # --- End Append Live Quote ---

    # Filter to last 10 years (apply *after* potentially adding today's quote)
    df = filter_data_timeframe(df, years=10)
    if df.empty:  # Check if empty after filtering
        logger.error(f"Data for {symbol} became empty after time filtering, skipping...")
        return {}, {}

    # --- Calculate Indicators ---
    df = calculate_moving_averages(df)

    # Calculate Returns (Make sure pct_change gives fractions like 0.01 for 1%)
    df['Return_1d'] = df['adj_close'].pct_change(periods=1)  # Daily return
    df['Return_1w'] = df['adj_close'].pct_change(periods=5)
    df['Return_1m'] = df['adj_close'].pct_change(periods=21)
    df['Return_3m'] = df['adj_close'].pct_change(periods=63)
    df['Return_6m'] = df['adj_close'].pct_change(periods=126)
    df['Return_1y'] = df['adj_close'].pct_change(periods=252)
    df['Return_2y'] = df['adj_close'].pct_change(periods=504)  # Approx 2 years

    # Calculate YTD return separately if needed for the individual table, or reuse calculation from calculate_returns
    # Reuse calculate_returns logic if needed for percentiles, but extract raw returns here too
    df_returns_for_percentiles, latest_returns_for_individual_table = calculate_returns(
        df.copy())  # Use copy if calculate_returns modifies df
    percentiles = calculate_percentiles(df_returns_for_percentiles)

    # RSI calculations
    df = calculate_rsi(df, window=14, wilder=True)  # Wilder's RSI ('RSI_14_Wilder')
    df = calculate_rsi(df, window=9, wilder=False)  # Fast RSI 9 ('RSI_9')
    df = calculate_rsi(df, window=14, wilder=False)  # Fast RSI 14 ('RSI_14')
    df = calculate_rsi(df, window=30, wilder=False)  # Fast RSI 30 ('RSI_30')

    # Volatility and drawdown
    df = calculate_volatility(df, windows=[10, 30, 60, 90])
    df = calculate_drawdown(df)  # Calculates 'drawdown_pct', 'hwm_price', 'hwm_date', etc.

    # --- Store the data ---
    if df.empty:
        logger.error(f"DataFrame for {symbol} is empty before storing data. Skipping.")
        return {}, {}

    # Store all required raw data points for BOTH the summary table and individual sections
    try:
        symbol_data = {
            # Data for Summary Table
            'LastPrice': df['adj_close'].iloc[-1],
            'HighWaterMarkPrice': df['hwm_price'].iloc[-1],
            'CurrentDrawdown': df['drawdown_pct'].iloc[-1] / 100.0,  # Store as fraction for formatting consistency
            'RSI': df['RSI_14_Wilder'].iloc[-1] if 'RSI_14_Wilder' in df.columns else None,  # Use Wilder 14d RSI
            'Return_1d': df['Return_1d'].iloc[-1],
            'Return_1w': df['Return_1w'].iloc[-1],
            'Return_1m': df['Return_1m'].iloc[-1],
            'Return_3m': df['Return_3m'].iloc[-1],
            'Return_6m': df['Return_6m'].iloc[-1],
            'Return_1y': df['Return_1y'].iloc[-1],
            'Return_2y': df['Return_2y'].iloc[-1],
            'DMA_50': df['MA50'].iloc[-1],
            'DMA_100': df['MA100'].iloc[-1],
            'DMA_200': df['MA200'].iloc[-1],

            # Data for Individual Section (some might overlap, ensure consistency)
            'latest_price': df['adj_close'].iloc[-1],  # Redundant but keeps existing individual table working
            'returns': latest_returns_for_individual_table,  # For the individual performance table
            'percentiles': percentiles,  # For the individual performance table
            'hwm_price': df['hwm_price'].iloc[-1],  # Redundant
            'hwm_date': df['hwm_date'].iloc[-1],
            'current_drawdown_pct_individual': df['drawdown_pct'].iloc[-1],
            # Keep original format if needed by individual table
            'drawdown_percentile': df['drawdown_percentile'].iloc[-1] if 'drawdown_percentile' in df.columns and not df['drawdown_percentile'].empty else None
        }
    except IndexError:
        logger.error(f"IndexError accessing data for {symbol}. DataFrame might be too small after processing.")
        return {}, {}

    # Generate plots (ensure index access is valid)
    try:
        plots = {
            'price_ma': create_price_ma_plot(df, symbol),
            'wilder_rsi': create_wilder_rsi_plot(df, symbol),
            'volatility': create_volatility_plot(df, symbol),
            'fast_rsi': create_fast_rsi_plot(df, symbol),
            'drawdown': create_drawdown_plot(df, symbol)
        }
    except IndexError:
        logger.error(f"IndexError generating plots for {symbol}. DataFrame might be too small.")
        return symbol_data, {}  # Return data but empty plots if plots fail

    return symbol_data, plots


def main() -> None:
    """
    Main function to run the daily market report generation.
    """
    symbols = ['SPY', 'QQQ', 'IWM', 'RSP', 'XLI', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV']
    all_symbol_data = {}
    all_plots = {}

    logger.info("Fetching initial real-time quotes for all symbols...")
    latest_quotes = fetch_realtime_bulk_quotes(symbols)

    processed_data_list = []  # Collect data dictionaries

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        try:
            symbol_data, plots = process_symbol(symbol, latest_quotes)
            if symbol_data:  # Check if data was returned
                all_symbol_data[symbol] = symbol_data
                symbol_data['Symbol'] = symbol  # Add symbol to dict for DataFrame creation
                processed_data_list.append(symbol_data)
            if plots:  # Check if plots were returned
                all_plots[symbol] = plots
        except Exception as e:
            logger.exception(f"Unhandled error processing {symbol}: {e}")

    summary_table_html = "<h2>Summary Table</h2><p>No data generated.</p>"  # Default
    if processed_data_list:
        logger.info("Aggregating data for summary table...")

        # Create DataFrame from the list of dictionaries
        summary_df = pd.DataFrame(processed_data_list)
        summary_df.set_index('Symbol', inplace=True)

        # Define columns and formats for the summary table
        summary_columns = [
            'LastPrice', 'HighWaterMarkPrice', 'CurrentDrawdown', 'RSI',
            'Return_1d', 'Return_1w', 'Return_1m', 'Return_3m', 'Return_6m', 'Return_1y', 'Return_2y',
            'DMA_50', 'DMA_100', 'DMA_200'
        ]
        # Ensure only available columns are used
        summary_columns_present = [col for col in summary_columns if col in summary_df.columns]

        header_formats = {
            'LastPrice': ('Last Price', 'currency'),
            'HighWaterMarkPrice': ('HWM Price', 'currency'),  # Added format
            'CurrentDrawdown': ('Drawdown', 'percent'),  # Use the fractional value calculated
            'RSI': ('RSI (14d)', 'float'),
            'Return_1d': ('1 Day', 'percent'),
            'Return_1w': ('1 Week', 'percent'),
            'Return_1m': ('1 Month', 'percent'),
            'Return_3m': ('3 Month', 'percent'),
            'Return_6m': ('6 Month', 'percent'),
            'Return_1y': ('1 Year', 'percent'),
            'Return_2y': ('2 Year', 'percent'),
            'DMA_50': ('50 DMA', 'currency'),
            'DMA_100': ('100 DMA', 'currency'),
            'DMA_200': ('200 DMA', 'currency'),
        }

        # Filter formats to only include present columns
        filtered_formats = {k: v for k, v in header_formats.items() if k in summary_columns_present}

        # Select only the desired columns for the table
        summary_df_selected = summary_df[summary_columns_present]

        # Create the HTML table
        summary_table_html = create_html_table(summary_df_selected, filtered_formats, "Market Summary")

    else:
        logger.warning("No symbols processed successfully, summary table cannot be generated.")

    if all_symbol_data and all_plots:
        logger.info("Sending email report...")
        plots_to_send = {sym: all_plots[sym] for sym in all_symbol_data if sym in all_plots}
        if plots_to_send:
            # Pass the summary table HTML to the send function
            send_email_report(all_symbol_data, plots_to_send, summary_table_html)
        else:
            logger.error("No valid plots correspond to the successfully processed symbol data. Aborting email.")
    else:
        logger.error("No symbol data or plots were successfully generated, aborting email...")

    plt.close('all')


if __name__ == "__main__":
    main()