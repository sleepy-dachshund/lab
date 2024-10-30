from utils_loader import *

"""
    This is a data pipeline that fetches historical and live data for a given date, compares it to existing data, and
    appends it to a CSV file. The pipeline is designed to be run daily, and the input date is expected to a business day.
    
    The pipeline is designed to be run in a production environment, where the input date is passed as an argument.
    
    If you want to run, set test_run to False and terminal_run to True. This will allow you to run the pipeline from the
    terminal. If you want to run the pipeline for a range of dates, set test_run to True.
    
    If you need data for testing, you can use the data_gen.py script in this repo 
    to generate data for the required dates.
"""


class EncoderPipeline:
    def __init__(self, input_date: str, output_file: str = "raw_data.csv"):

        # Initial parameters and variables for the pipeline
        self.input_str = input_date
        self.output_file = output_file
        self.logger = None

        # Date and time inputs
        self.input_date = None
        self.execution_time = None
        self.previous_biz_day_str = None  # Folder path

        # Existing data
        self.existing_data = None
        self.existing_columns = None
        self.raw_data_exists = False

        # Historical data
        self.historical_data = None
        self.historical_columns = None
        self.historical_data_cleaned = False

        # Live data
        self.live_data = None
        self.live_columns = None
        self.live_data_cleaned = False

        # Data to append -- Live or Historical
        self.new_data = None
        self.new_columns = None

        # Duplicates
        self.duplicate_columns = ['time', 'unique_id']
        self.duplicates_found = False
        self.duplicates = None

        # Output file formatting
        self.first_cols = ['time', 'date', 'unique_id']

    def fetch_logger(self):
        # Initiate logger
        self.logger = setup_logging(log_file="encoder_pipeline.log")

    def convert_input_str_to_date(self):
        # Convert input_str to a datetime object
        self.input_date = datetime.strptime(self.input_str, '%Y%m%d')

    def calculate_previous_business_day(self):
        # Calculate the previous business day
        self.previous_biz_day_str = get_previous_business_day(self.input_date)

    def check_raw_data_exists(self):
        # Check if output_file exists
        if os.path.exists(self.output_file):
            try:
                self.existing_data = pd.read_csv(self.output_file)
            except Exception as e:
                logger.error(f"Error reading existing data from {self.output_file}: {e}")
                self.existing_data = pd.DataFrame()
        else:
            self.existing_data = pd.DataFrame()

        if not self.existing_data.empty:
            # todo: cleanup
            first_cols = [col for col in self.first_cols if col in self.existing_data.columns]
            self.existing_data = self.existing_data[
                first_cols + [col for col in self.existing_data.columns if col not in first_cols]]
            self.raw_data_exists = True
            self.existing_columns = self.existing_data.columns.to_list()

    def fetch_historical_data(self):
        # Load historical data for the previous business day
        self.historical_data, self.historical_columns = load_historical_data(self.previous_biz_day_str, self.logger)
        first_cols = [col for col in self.first_cols if col in self.historical_data.columns]
        self.historical_data = self.historical_data[
            first_cols + [col for col in self.historical_data.columns if col not in first_cols]]
        if not self.historical_data.empty:
            self.historical_data = validate_data_types(self.historical_data, self.logger)
            self.historical_data_cleaned = True

    def fetch_live_data(self):
        # Load live data for the previous business day
        self.live_data, self.live_columns = load_live_data(self.previous_biz_day_str, self.logger)
        first_cols = [col for col in self.first_cols if col in self.live_data.columns]
        self.live_data = self.live_data[
            first_cols + [col for col in self.live_data.columns if col not in first_cols]]
        if not self.live_data.empty:
            self.live_data = validate_data_types(self.live_data, self.logger)
            self.live_data_cleaned = True

    def compare_new_and_existing_data(self):
        # Compare new and existing data
        data_check(self.new_data, self.existing_data, self.logger)

    def check_duplicates(self):
        # Check for duplicates in the new data
        self.duplicates = self.existing_data.merge(self.new_data, how='inner', on=self.duplicate_columns)
        if not self.duplicates.empty:
            self.duplicates_found = True
        if self.new_data[self.duplicate_columns].duplicated().any():
            self.logger.error("Duplicates found in new data.")

    def drop_new_data_dupes(self):
        # Drop duplicate entries in the new data
        # todo: there is more logic to be added here... this currently blindly drops observations
        self.new_data.drop_duplicates(subset=self.duplicate_columns, inplace=True)

    def append_to_csv(self):
        # Option to create separate function here
        pass

    def run(self):
        self.fetch_logger()

        # Log execution time
        self.execution_time = datetime.now()

        # Get all the data
        self.convert_input_str_to_date()
        self.calculate_previous_business_day()
        self.check_raw_data_exists()
        self.fetch_historical_data()
        self.fetch_live_data()

        # If no data found for given input date, return
        if not self.historical_data_cleaned and not self.live_data_cleaned:
            self.logger.warning(f"No data found for {self.previous_biz_day_str}.")
            return

        # If data found in historical and live locations for given input date, return
        if self.historical_data_cleaned and self.live_data_cleaned:
            self.logger.error("Both historical and live data found for the same date.")
            return

        # If neither of the above -- we have one data source to work with
        self.new_data = self.historical_data if self.historical_data_cleaned else self.live_data
        self.new_columns = self.historical_columns if self.historical_data_cleaned else self.live_columns

        # If we have new and existing data, compare and append
        if self.raw_data_exists and not self.new_data.empty:
            self.compare_new_and_existing_data()

            # Vendor probably adds new features from time to time
            if self.existing_columns != self.new_columns:
                self.logger.debug("Columns in new data do not match existing data.")
                # todo: decide what to do here. For now, just log it.
                #  Recommend trimming new_data to existing_columns; dumping the new column data elsewhere for review
            else:
                self.logger.debug("Columns in new data match existing data.")

            # If duplicates found, drop them
            if self.duplicates_found:
                self.drop_new_data_dupes()

            # Append new data to existing data and output to output_file
            combined_data_out = pd.concat([self.existing_data, self.new_data], ignore_index=True)
            combined_data_out.drop_duplicates(subset=self.duplicate_columns, inplace=True)

            # Write data to output_file
            # todo: this is going to be a time suck in prod, couple minutes, but for now it's fine
            combined_data_out.to_csv(self.output_file, index=False)

        # If no new data found, return
        elif self.new_data.empty:
            self.logger.debug("No new data found for the date.")
            return

        # If no existing data, create new output file
        elif not self.raw_data_exists:
            self.logger.debug("Existing data not found. Creating new output file.")
            if not self.new_data.empty:
                # todo: careful here, this could be a huge re-write.
                #  probably want a staging file just in case -- or cache raw_data.csv periodically e.g. on Saturdays
                self.new_data.to_csv(self.output_file, index=False)
        else:
            self.logger.debug("Something's up. Check the logs.")
            return


# Example instantiation of the class
if __name__ == "__main__":
    test_run = True
    terminal_run = False

    if not test_run:

        # option to run in terminal
        if terminal_run:
            parser = argparse.ArgumentParser(description='Encoder data pipeline.')
            parser.add_argument('--date', type=str, required=True, help='Date in YYYYMMDD format')
            args = parser.parse_args()
            input_date = args.date
        else:
            input_date = datetime.date.today() - datetime.timedelta(days=1)
            input_date = input_date.strftime('%Y%m%d')

        ep = EncoderPipeline(input_date, output_file="raw_data_oop.csv")
        ep.run()

        # test_ex = encoder_pipeline.existing_data
        # test_hist = encoder_pipeline.historical_data
        # test_live = encoder_pipeline.live_data

    else:
        start_date = datetime.date.today()
        end_date = datetime.date.today() - datetime.timedelta(days=55)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        for date in date_range:
            ep = EncoderPipeline(date.strftime('%Y%m%d'), output_file="test_data_oop.csv")
            ep.run()
