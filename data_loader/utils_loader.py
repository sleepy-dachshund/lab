import pandas as pd
import numpy as np
import logging
import os
import datetime
import csv
import zipfile
import argparse
from datetime import datetime, timedelta, timezone
import glob
import traceback
from typing import Tuple, Optional, List
from utils_logging import setup_logging


# Configurations
DATA_DIR = 'data' # directory where historical and live data is stored -- check data_gen.py in this repo to create
OUTPUT_FILE = 'raw_data.csv'
LOG_FILE = 'data_loader.log'


def load_historical_data(date_str: str, logger: logging.Logger) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Loads historical data from a zip file for the specified date.

    Args:
        date_str (str): Date in YYYYMMDD format representing the historical day for the data.
        logger: Logger object to log errors.

    Returns:
        Tuple[pd.DataFrame, Optional[List[str]]]: Combined DataFrame of all files inside the zip for the given date,
        and the header of the combined data.

    Logic:
    - The function reads data from a zip file corresponding to `date_str`.
    - The 'time' column is set as the nanoseconds since Unix epoch for 9:00 AM EST of the given day.
        - This is a questionable assumption since the knowledge date of the historical data is not clear.
    - The function returns an empty DataFrame if the historical data is not found.
    - We return the header in case we'd like to compare it with existing data.
    """
    historical_path = os.path.join(DATA_DIR, 'historical', f'{date_str}.zip')
    if not os.path.exists(historical_path):
        return pd.DataFrame(), None

    try:
        with zipfile.ZipFile(historical_path, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            data_frames = []
            header = None
            for file_name in file_names:
                with zip_ref.open(file_name) as file:
                    df = pd.read_csv(file)

                    # *** Didn't find knowledge date in these historical files ***
                    # Converting date_str + 1 day to nanoseconds since Unix epoch -- which is questionable
                    # also using 9:00 AM EST since it's not clear what time the data was downloaded + daylights saving buffer
                    dt = datetime.strptime(date_str, '%Y%m%d')
                    dt = dt.replace(hour=9, minute=0, second=0, tzinfo=timezone(timedelta(hours=-5)))
                    dt += timedelta(days=1)
                    df['time'] = int(dt.timestamp() * 1e9)  # Set 'time' as nanoseconds since Unix epoch at 9:00AM EST

                    data_frames.append(df)
                    if header is None:
                        header = df.columns.tolist()
            combined_data = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
            return combined_data, header
    except Exception as e:
        logger.error(f"Error loading historical data for {date_str}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), None


def load_live_data(date_str: str, logger: logging.Logger) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Loads live data CSV files for the specified date.

    Args:
        date_str (str): Date in YYYYMMDD format representing the live day of the data.
        logger: Logger object to log errors.

    Returns:
        Tuple[pd.DataFrame, Optional[List[str]]]: Combined DataFrame of all live CSV files for the given date,
        and the header of the combined data.

    Logic:
    - The function reads all CSV files in the directory corresponding to `date_str`.
    - The 'time' column is set as the download timestamp taken from the file name.
    - We return the header in case we'd like to compare it with existing data.
    """
    daily_path = os.path.join(DATA_DIR, date_str, '*.csv')
    files = glob.glob(daily_path)
    data_frames = []
    header = None
    try:
        for file in files:
            file_name = os.path.basename(file)
            df = pd.read_csv(file)
            df['time'] = int(file_name.split('.')[0])  # Set 'time' as the filename (download timestamp)
            data_frames.append(df)
            if header is None:
                header = df.columns.tolist()
        combined_data = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
        return combined_data, header
    except Exception as e:
        logger.error(f"Error loading live data for {date_str}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), None


def read_existing_data(logger: logging.Logger) -> pd.DataFrame:
    """
    Reads the existing raw data from the CSV file.

    Args:
        logger: Logger object to log errors.

    Returns:
        pd.DataFrame: DataFrame containing the existing data from `raw_data.csv`.

    Logic:
    - The function reads `raw_data.csv` if it exists; otherwise, it returns an empty DataFrame.
    - We want to read the existing data to align columns and prevent duplicates.
    """
    if os.path.exists(OUTPUT_FILE):
        try:
            return pd.read_csv(OUTPUT_FILE)
        except Exception as e:
            logger.error(f"Error reading existing data from {OUTPUT_FILE}: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    return pd.DataFrame()


def validate_data_types(data: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Validates and converts data types of the DataFrame columns.

    Args:
        data (pd.DataFrame): DataFrame to validate and convert.
        logger: Logger object to log errors.

    Returns:
        pd.DataFrame: DataFrame with validated and converted data types.

    Logic:
    - Converts the 'time' and 'unique_id' columns to integers.
    - Ensures columns starting with 'field_' are floats and replaces invalid entries with NaN.
    """
    try:
        # Ensure 'time' column is int
        if 'time' in data.columns:
            data['time'] = pd.to_numeric(data['time'], errors='coerce').fillna(0).astype(int)

        # Ensure 'unique_id' column is int
        if 'unique_id' in data.columns:
            data['unique_id'] = pd.to_numeric(data['unique_id'], errors='coerce').fillna(0).astype(int)

        # Ensure all 'field_' columns are float, replace any string values with NaN
        for column in data.columns:
            if column.startswith('field_'):
                data.loc[:, column] = pd.to_numeric(data[column], errors='coerce').astype('float64')
                data[column] = data[column].astype(float)

        return data
    except Exception as e:
        logger.error(f"Error validating data types: {e}")
        traceback.print_exc()
        return data


def log_data_gaps(missing_dates, message, logger: logging.Logger):
    """ Log missing dates """
    log_missing_dates = [date.strftime('%Y%m%d') for date in missing_dates]
    if missing_dates.size > 0:
        logger.warning(f"{message}: {log_missing_dates}")


def get_missing_business_days(dates, start_date, end_date):
    """ Find missing business days """
    full_range = pd.date_range(start=start_date, end=end_date, freq='B')
    return full_range.difference(dates)


def data_check(new_data: pd.DataFrame, existing_data: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Check new data fields and coverage.

    Args:
        new_data (pd.DataFrame): DataFrame to be appended.
        existing_data (pd.DataFrame): Existing data in `raw_data.csv`.
        logger: Logger object to log messages.

    Returns:
        None

    Logic:
    - Check if there is existing data and log the overlap of unique_ids and new columns.
    - Check for missing dates in existing and new data as well as in between the two.
    - Calculate the data coverage of the new data and log the percentage.
    - Note if any new features are added to the data.
    """

    if existing_data.empty:
        # Log new output file creation
        logger.info(f"New output file created.")
    else:
        # Log overlap of unique_ids
        overlap_ids = (len(np.intersect1d(existing_data['unique_id'].unique(),
                                          new_data['unique_id'].unique()))
                       / len(existing_data['unique_id'].unique()))
        logger.info(f"Overlap of unique_ids: {overlap_ids:.2%}")

        # Log new columns added
        new_cols = [col for col in new_data.columns if col not in existing_data.columns]
        if new_cols:
            logger.info(f"New columns added: {new_cols}")

        # Define and check dates
        existing_dates = pd.to_datetime(existing_data['date'].drop_duplicates(), format='%Y%m%d').sort_values()
        new_dates = pd.to_datetime(new_data['date'].drop_duplicates(), format='%Y%m%d').sort_values()

        # Log gaps in existing data
        missing_dates_existing = get_missing_business_days(existing_dates, existing_dates.min(), existing_dates.max())
        log_data_gaps(missing_dates_existing, "Existing data gap", logger)

        # Log gaps in new data
        missing_dates_new = get_missing_business_days(new_dates, new_dates.min(), new_dates.max())
        log_data_gaps(missing_dates_new, "New data gap", logger)

        # Log gaps between existing and new data
        missing_dates_between = get_missing_business_days(
            dates=pd.concat([existing_dates, new_dates]),
            start_date=existing_dates.max() + pd.Timedelta(days=1),
            end_date=new_dates.min() - pd.Timedelta(days=1)
        )
        log_data_gaps(missing_dates_between, "Missing data between existing and new data", logger)

        # Log multiple dates found in new data
        if len(new_dates) > 1:
            logger.warning("Multiple dates found in new data.")

        # Log overlapping dates
        overlapping_dates = existing_dates[existing_dates.isin(new_dates)]
        if not overlapping_dates.empty:
            log_overlapping_dates = [date.strftime('%Y%m%d') for date in overlapping_dates]
            logger.warning(f"Data already exists for dates: {log_overlapping_dates}. Will drop duplicates.")

        # Check for dates in existing_data that are after dates in new_data
        future_dates_in_existing = existing_dates[existing_dates > new_dates.max()]
        if not future_dates_in_existing.empty:
            log_future_dates = [date.strftime('%Y%m%d') for date in future_dates_in_existing]
            logger.warning(f"Existing data contains dates after the latest date in new data: {log_future_dates}")

    # Log basic data coverage
    data_coverage = new_data.iloc[3:].isna().sum().sum() / new_data.size
    logger.info(f"Data coverage: {1 - data_coverage:.2%}")


def append_to_csv(new_data: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Appends new data to the existing raw_data.csv file.

    Args:
        new_data (pd.DataFrame): DataFrame containing new data to append.
        logger: Logger object to log messages.

    Logic:
    - Reads the existing data and aligns columns between existing and new data.
    - Ensures the 'time' column is the first column, followed by 'date', 'unique_id', and other columns.
    - Drops duplicate rows based on 'time' and 'unique_id' to prevent duplicate entries. Should confirm this logic.
    """
    try:
        # Load existing data if present
        if os.path.exists(OUTPUT_FILE):
            existing_data = read_existing_data(logger)
        else:
            existing_data = pd.DataFrame()

        # Validate data types of new data
        new_data = validate_data_types(new_data, logger)

        # Ensure the 'time' column is the first column
        columns_order = ['time', 'date', 'unique_id'] + [col for col in new_data.columns if
                                                         col not in ['time', 'date', 'unique_id']]
        new_data = new_data[columns_order]

        data_check(new_data, existing_data, logger)

        # Check and align columns
        if not existing_data.empty:
            for col in new_data.columns:
                if col not in existing_data.columns:
                    existing_data[col] = pd.NA
            for col in existing_data.columns:
                if col not in new_data.columns:
                    new_data[col] = pd.NA
            # Reorder columns to match existing data
            new_data = new_data[existing_data.columns]
        else:
            # If existing data is empty, set new data columns
            existing_data = new_data

        # Combine and drop duplicates
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        # Plenty of room for improvement in drop_duplicates() logic, but I'm not sure what the requirements are
        combined_data.drop_duplicates(subset=['time', 'unique_id'], inplace=True)

        # Save back to CSV
        combined_data.to_csv(OUTPUT_FILE, index=False)
        logger.info('Data appended successfully to raw_data.csv.')
    except Exception as e:
        print(f"Error appending to {OUTPUT_FILE}: {e}")
        traceback.print_exc()


def get_previous_business_day(date: datetime) -> str:
    """
    Calculates the previous business day for a given date.

    Args:
        date (datetime): The date for which to find the previous business day.

    Returns:
        str: Previous business day in YYYYMMDD format.

    Logic:
    - Starts from the day before the given date and iterates backwards to find the last weekday (Mon-Fri).
    - Would want to confirm this logic though I think this behavior makes sense.
    """
    previous_day = date - timedelta(days=1)
    while previous_day.weekday() > 4:  # Mon-Fri are 0-4
        previous_day -= timedelta(days=1)
    return previous_day.strftime('%Y%m%d')


def main(logger: logging.Logger) -> None:
    """
    Main function to handle data pipeline logic.

    Args:
        logger: Logger object to log messages.

    Logic:
    - Parses the command line argument for a date, T.
    - Loads historical and/or live data for the previous business day, T-1.
    - Appends combined data to `raw_data.csv`.
    """
    logger.info('Starting encoder data pipeline.')
    # Argument parsing
    parser = argparse.ArgumentParser(description='Encoder data pipeline.')
    parser.add_argument('--date', type=str, required=True, help='Date in YYYYMMDD format')
    args = parser.parse_args()
    date_str = args.date
    logger.info(f"Running for input date {date_str}.")

    try:
        # Calculate the date to pull data for -- arg is T, so append data for T-1 (prev biz day)
        dt = datetime.strptime(date_str, '%Y%m%d')
        previous_business_day = get_previous_business_day(dt)

        # Load data from both historical and live sources for the previous business day
        historical_data, historical_header = load_historical_data(previous_business_day, logger)
        live_data, live_header = load_live_data(previous_business_day, logger)
        # Note we're adding both here which might not be desired. We could add a flag to only add one or the other.
        # but instead we add both and rely on the drop_duplicates() logic in append_to_csv() to handle any overlap.
        data = pd.concat([historical_data, live_data],
                         ignore_index=True) if not historical_data.empty or not live_data.empty else pd.DataFrame()

        # Append to raw_data.csv
        if not data.empty:
            append_to_csv(data, logger)
        else:
            logger.warning(f"No data found for {previous_business_day}.")

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    logger = setup_logging(LOG_FILE)
    main(logger)
