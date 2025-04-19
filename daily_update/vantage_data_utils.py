import os
import requests
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Callable


try:  # Try to import local config first (for local development)
    import config
    VANTAGE_API_KEY = config.VANTAGE_API_KEY
except ImportError:  # Fall back to environment variables (for GitHub Actions) -- repo secrets from .yml file env
    VANTAGE_API_KEY = os.environ.get('VANTAGE_API_KEY')


''' =======================================================================================================
    Prices
    - historical daily adj_close for one symbol
    - realtime bulk quotes for list of symbols
======================================================================================================= '''


def fetch_alpha_vantage_prices(symbol: str, output_size: str = 'full', logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Fetch daily stock data (adj. close prices) from Alpha Vantage API.

    Parameters
    ----------
    symbol : str
        The stock symbol to fetch data for.
    output_size : str, optional
        The amount of data to fetch, by default 'full' (up to 20 years).
    logger : logging.Logger, optional
        Logger instance for logging messages, by default None.

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


def fetch_realtime_bulk_quotes(symbols: List[str], logger: Optional[logging.Logger] = None) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[float]]]:
    """
    Fetch realtime bulk quotes from Alpha Vantage for a list of symbols.

    Prioritizes 'close' price (live market price) if available,
    otherwise uses 'extended_hours_quote'.

    Parameters
    ----------
    symbols : List[str]
        List of stock symbols (max 100 per call recommended by Alpha Vantage).
    logger : logging.Logger, optional
        Logger instance for logging messages, by default None.

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
    Overviews
    - company overview data for one symbol
======================================================================================================= '''


def fetch_alpha_vantage_overviews(symbol: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Fetch stock company overview data from Alpha Vantage API.

    Parameters
    ----------
    symbol : str
        The stock symbol to fetch data for.
    logger : logging.Logger, optional
        Logger instance for logging messages, by default None.

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


''' =======================================================================================================
    Earnings
    - Earnings Call Transcripts for one symbol & one quarter
======================================================================================================= '''

def fetch_alpha_vantage_cash_flow(symbol: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Fetch cash flow data from Alpha Vantage API (TTM figures, from quarterlyReports).

    Parameters
    ----------
    symbol : str
        The stock symbol to fetch data for.
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, a default logger is used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the TTM company cash flow data. Returns an empty DataFrame on error.
        Index:      Symbol (string, uppercase version of input symbol)
        Columns:    fiscalDateEnding (datetime64[ns]), operatingCashflow (float64),
                    capitalExpenditures (float64), netIncome (float64),
                    freeCashFlow (float64)
                    # Note: paymentsForRepurchaseOfEquity, dividendPayout are excluded as per comments
    """

    upper_symbol = symbol.upper() # Standardize symbol
    logger.info(f"Fetching cash flow data for {upper_symbol}...")
    url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={upper_symbol}&apikey={VANTAGE_API_KEY}"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = r.json()

        # --- API Response Validation ---
        if not data:
             logger.error(f"Empty response received from API for {upper_symbol}.")
             return pd.DataFrame()

        if "Error Message" in data:
            logger.error(f"API Error fetching cash flow for {upper_symbol}: {data['Error Message']}")
            return pd.DataFrame()

        # Check for rate limiting messages or other info messages indicating no data
        if "Information" in data:
             logger.warning(f"API Info for {upper_symbol}: {data['Information']}. Might be rate limited or data unavailable.")
             # Often rate limit messages don't contain the actual data keys
             if 'quarterlyReports' not in data:
                 return pd.DataFrame()

        if 'symbol' not in data or data['symbol'] != upper_symbol:
            logger.error(f"Unexpected response format or symbol mismatch for {upper_symbol}: {data.get('symbol', 'N/A')}")
            return pd.DataFrame()

        if 'quarterlyReports' not in data or not isinstance(data['quarterlyReports'], list):
            logger.error(f"Missing or invalid 'quarterlyReports' data for {upper_symbol}.")
            return pd.DataFrame()

        quarterly_reports = data['quarterlyReports']

        if len(quarterly_reports) == 0:
            logger.warning(f"No quarterly reports found for {upper_symbol}.")
            return pd.DataFrame()

        # --- Data Processing ---
        # We only want the last 4 quarterlyReports (TTM)
        # API usually returns newest first, so take the first 4 if available
        if len(quarterly_reports) < 4:
            logger.warning(f"Insufficient quarterly reports ({len(quarterly_reports)} < 4) for TTM calculation for {upper_symbol}. Using available data.")
            reports_to_process = quarterly_reports # Use whatever is available
        else:
            reports_to_process = quarterly_reports[:4]

        df_quarterly = pd.DataFrame(reports_to_process)

        # Define required columns
        date_col = 'fiscalDateEnding'
        numeric_cols = ['operatingCashflow', 'capitalExpenditures', 'netIncome']
                     # 'paymentsForRepurchaseOfEquity', 'dividendPayout'] # Add back if needed
        required_cols = [date_col] + numeric_cols

        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df_quarterly.columns]
        if missing_cols:
            logger.error(f"Missing required columns in quarterly data for {upper_symbol}: {missing_cols}")
            return pd.DataFrame()

        # Select only the necessary columns
        df_quarterly = df_quarterly[required_cols]

        # Convert 'None' strings and empty strings to NaN before numeric conversion
        df_quarterly.replace('None', np.nan, inplace=True)
        df_quarterly.replace('', np.nan, inplace=True)

        # Convert data types
        try:
            df_quarterly[date_col] = pd.to_datetime(df_quarterly[date_col], errors='coerce')
            for col in numeric_cols:
                df_quarterly[col] = pd.to_numeric(df_quarterly[col], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting data types for {upper_symbol}: {e}")
            return pd.DataFrame()

        # Drop rows where essential data (like date or operating cash flow) couldn't be converted
        # Or where the date is NaT
        df_quarterly.dropna(subset=[date_col] + numeric_cols, how='any', inplace=True)

        if df_quarterly.empty:
            logger.warning(f"No valid quarterly data remaining after cleaning for {upper_symbol}.")
            return pd.DataFrame()

        # --- TTM Aggregation ---
        # Sum the numeric values for the selected reports and take the max fiscalDateEnding
        latest_date = df_quarterly[date_col].max()
        ttm_sums = df_quarterly[numeric_cols].sum()

        # Create the single-row result DataFrame
        ttm_data = {date_col: latest_date}
        ttm_data.update(ttm_sums.to_dict())

        df_result = pd.DataFrame([ttm_data]) # Create DataFrame from a list containing the dict

        # --- Calculate freeCashFlow ---
        # Ensure required columns for calculation exist and are numeric after aggregation
        if 'operatingCashflow' in df_result.columns and 'capitalExpenditures' in df_result.columns:
             # Handle potential NaN results from sum if input data had NaNs
            op_cashflow = df_result['operatingCashflow'].iloc[0]
            cap_ex = df_result['capitalExpenditures'].iloc[0]
            if pd.notna(op_cashflow) and pd.notna(cap_ex):
                 df_result['freeCashFlow'] = op_cashflow - cap_ex
            else:
                 df_result['freeCashFlow'] = np.nan # Assign NaN if components are NaN
        else:
             logger.warning(f"Could not calculate freeCashFlow for {upper_symbol} due to missing components.")
             df_result['freeCashFlow'] = np.nan # Add column with NaN if calculation failed

        # Set the index to the symbol
        df_result.set_index(pd.Index([upper_symbol], name='Symbol'), inplace=True)

        logger.info(f"Successfully processed TTM cash flow data for {upper_symbol}.")
        return df_result

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching cash flow data for {upper_symbol}.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching cash flow data for {upper_symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching cash flow data for {upper_symbol}: {e}", exc_info=True) # Log traceback
        return pd.DataFrame()


def fetch_earnings_transcript(symbol: str, quarter: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Fetch earnings call transcript from Alpha Vantage API for a specific symbol and quarter.

    Parameters
    ----------
    symbol : str
        The stock symbol (e.g., 'IBM').
    quarter : str
        The fiscal quarter (e.g., '2024Q1').
    logger : logging.Logger, optional
        Logger instance for logging messages, by default None.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the earnings transcript data (symbol, quarter, transcript list).
        Returns an empty dictionary if fetching fails or no transcript is found.
    """
    if logger:
        logger.info(f"Fetching earnings transcript for {symbol}, quarter {quarter}...")
    else:
        print(f"Fetching earnings transcript for {symbol}, quarter {quarter}...") # Basic print if no logger

    if not VANTAGE_API_KEY:
        if logger:
            logger.error("Alpha Vantage API key not found.")
        else:
            print("Alpha Vantage API key not found.")
        return {}

    url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={symbol}&quarter={quarter}&apikey={VANTAGE_API_KEY}"

    try:
        r = requests.get(url, timeout=45)  # Increased timeout slightly for potentially larger response
        r.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = r.json()

        # Check for specific Alpha Vantage errors
        if "Error Message" in data:
            if logger:
                logger.error(f"API Error fetching transcript for {symbol} {quarter}: {data['Error Message']}")
            else:
                print(f"API Error fetching transcript for {symbol} {quarter}: {data['Error Message']}")
            return {}

        # Check for rate limit messages or other info messages
        if "Information" in data:
             if logger:
                logger.warning(f"API Info for {symbol} {quarter}: {data['Information']}. Might be rate limited or no transcript available.")
             else:
                 print(f"API Info for {symbol} {quarter}: {data['Information']}. Might be rate limited or no transcript available.")
            # Often, an info message means the desired data isn't present
             return {}

        # Check if the essential transcript data is present
        if "symbol" not in data or "quarter" not in data or "transcript" not in data:
            if logger:
                logger.error(f"Unexpected transcript response format for {symbol} {quarter}: {data}")
            else:
                 print(f"Unexpected transcript response format for {symbol} {quarter}: {data}")
            return {}

        # Check if the transcript list is empty (valid response, but no content)
        if not data["transcript"]:
             if logger:
                logger.warning(f"Transcript data found for {symbol} {quarter}, but the transcript list is empty.")
             else:
                 print(f"Transcript data found for {symbol} {quarter}, but the transcript list is empty.")
             # Still return the data, as it's a valid structure, just empty.
             return data

        if logger:
            logger.info(f"Successfully fetched transcript for {symbol}, quarter {quarter}. Transcript segments: {len(data.get('transcript', []))}")
        else:
            print(f"Successfully fetched transcript for {symbol}, quarter {quarter}. Transcript segments: {len(data.get('transcript', []))}")
        return data # Return the full JSON dictionary as requested

    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f"Network error fetching transcript for {symbol} {quarter}: {e}")
        else:
            print(f"Network error fetching transcript for {symbol} {quarter}: {e}")
        return {}
    except Exception as e:
        if logger:
            logger.error(f"An unexpected error occurred fetching transcript for {symbol} {quarter}: {e}")
        else:
            print(f"An unexpected error occurred fetching transcript for {symbol} {quarter}: {e}")
        return {}

''' ========================================================================================================
    Example Usage
======================================================================================================== '''

if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Example usage of the functions
    symbol = 'AMZN'
    transcript_quarter = '2024Q4'

    # Example usage of the fetch_earnings_transcript function
    ern_transcript = fetch_earnings_transcript(symbol, transcript_quarter, logger=logger)
    cash_flow = fetch_alpha_vantage_cash_flow(symbol, logger=logger)
    prices = fetch_alpha_vantage_prices(symbol, logger=logger)
    overview = fetch_alpha_vantage_overviews(symbol, logger=logger)

    test = pd.merge(overview, cash_flow, how='left', left_index=True, right_index=True, validate='1:1')

    ern_prompt_llm = f"""
    Analyze this earnings call transcript and provide a bullet-point summary with:
    
    - Company Overview (3-4 sentences in bullets): Core business, primary revenue segments (with % contribution where available), key cost drivers, growth vs. established segments, and main competitors.
    - Financial Performance: Key metrics from transcript (revenue, profit, EPS) with YoY growth percentages.
    - Forward Guidance: Management's outlook for upcoming quarters/year.
    - Strategic Initiatives: Product updates, market positioning, and business strategy changes.
    - Q&A Themes: Major questions raised and management's responses.
    - Major Announcements: Acquisitions, restructuring, leadership changes, or other significant news if applicable.
    
    Keep all points brief and informative. Avoid filler text.
    
    Transcript Data:
    {ern_transcript}
    """

    # todo: ping Gemini API with the prompt & earnings transcript to get text summary
