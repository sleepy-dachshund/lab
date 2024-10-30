import os
import datetime
import zipfile
import time
import pandas as pd
import numpy as np


class SynthDataGen:
    def __init__(self, base_directory, num_business_days=25, num_unique_ids=100, random_seed=42):
        """
        Initializes the SynthDataGen with the base directory.

        :param base_directory: Path to the base directory where the 'data' folder will be created.
        :param num_business_days: Number of business days for recent and historical data.
        :param num_unique_ids: Number of unique identifiers to generate.
        :param random_seed: Seed for random number generation to ensure reproducibility.
        """
        self.base_directory = base_directory
        self.data_directory = os.path.join(self.base_directory, 'data')
        self.historical_directory = os.path.join(self.data_directory, 'historical')
        self.num_business_days = num_business_days
        self.num_unique_ids = num_unique_ids
        np.random.seed(random_seed)  # Set random seed for reproducibility
        self.unique_identifiers = [np.random.randint(1000, 9999) for _ in
                                   range(self.num_unique_ids)]  # Generate unique identifiers
        self.means_stds = self.generate_means_and_stds()
        self.df = None
        self.df_fun = None

    def create_data_directory(self):
        """
        Creates the main 'data' directory.
        """
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def create_recent_directories(self):
        """
        Creates directories named as the last `num_business_days` business days (excluding today).
        """
        today = datetime.date.today()
        business_days = []
        while len(business_days) < self.num_business_days:
            today -= datetime.timedelta(days=1)
            if today.weekday() < 5:  # Monday to Friday are business days
                business_days.append(today.strftime("%Y%m%d"))

        for day in business_days:
            day_directory = os.path.join(self.data_directory, day)
            if not os.path.exists(day_directory):
                os.makedirs(day_directory)

    def generate_means_and_stds(self):
        """
        Generates a dictionary of means and standard deviations for each unique identifier and field.
        """
        means_stds = {}
        for uid in self.unique_identifiers:
            means_stds[uid] = {}
            for i in range(1, 11):
                mean = np.random.uniform(i*5, i*6)
                std = np.random.uniform(1, i+1)
                means_stds[uid][f"field_{i}"] = (mean, std)
        return means_stds

    def generate_data_frame(self):
        """
        Generates the main DataFrame containing data for all dates.
        """
        today = datetime.date.today()
        business_days = []
        while len(business_days) < (2 * self.num_business_days):  # Recent + Historical
            today -= datetime.timedelta(days=1)
            if today.weekday() < 5:  # Monday to Friday are business days
                business_days.append(today.strftime("%Y%m%d"))

        data = []
        for as_of_date in business_days:
            for uid in self.unique_identifiers:
                row = {
                    "as_of_date": int(as_of_date),
                    "unique_identifier": uid,
                }
                for i in range(1, 11):
                    mean, std = self.means_stds[uid][f"field_{i}"]
                    row[f"field_{i}"] = np.random.normal(mean, std)
                data.append(row)

        self.df = pd.DataFrame(data)

    def make_it_fun(self, activate: bool = False):
        """
        This function makes it fun.
        """
        fun_df = self.df.copy()

        # drop some rows at random
        drop_rows = int(0.01 * len(fun_df)) if len(fun_df) > 100 else 2
        drop_indices = np.random.choice(fun_df.index, drop_rows, replace=False)
        fun_df = fun_df.drop(drop_indices)

        # drop an entire day of data
        drop_day = np.random.choice(fun_df['as_of_date'].unique())
        fun_df = fun_df[fun_df['as_of_date'] != drop_day]

        # add some duplicates
        duplicate_rows = int(0.01 * len(fun_df)) if len(fun_df) > 100 else 2
        duplicate_indices = np.random.choice(fun_df.index, duplicate_rows, replace=False)
        fun_df = pd.concat([fun_df, fun_df.loc[duplicate_indices]], ignore_index=True)

        # add some NaNs
        nan_rows = int(0.03 * len(fun_df)) if len(fun_df) > 100 else 3
        for i in range(1, 11):
            nan_indices = np.random.choice(fun_df.index, nan_rows, replace=False)
            fun_df.loc[nan_indices, f'field_{i}'] = np.nan

        # add some bad data types
        fun_df['field_1'] = fun_df['field_1'].astype('object')
        fun_df['field_1'].fillna('none', inplace=True)

        self.df_fun = fun_df.copy()

        if activate:
            self.df = fun_df.copy()

    def create_csv_files_in_recent_directories(self):
        """
        Creates a CSV file in each of the recent business day directories with a name of 9am EST for the following day.
        """
        recent_directories = [d for d in os.listdir(self.data_directory) if d != 'historical']
        recent_directories.sort()  # Ensure the order is correct

        for directory in recent_directories:
            directory_path = os.path.join(self.data_directory, directory)
            if os.path.isdir(directory_path):
                # Calculate 9am EST for the following day
                date = datetime.datetime.strptime(directory, "%Y%m%d")
                next_day = date + datetime.timedelta(days=1)
                nine_am_est = datetime.datetime(next_day.year, next_day.month, next_day.day, 9, 0)
                nine_am_epoch = int(nine_am_est.timestamp() * 1e9)  # Nanoseconds since Unix epoch

                # Filter data for the specific date
                df_day = self.df.loc[self.df['as_of_date'] == int(directory)].copy()
                df_day["knowledge_date"] = nine_am_epoch

                # Create CSV file
                csv_filename = f"{nine_am_epoch}.csv"
                csv_path = os.path.join(directory_path, csv_filename)
                df_day.to_csv(csv_path, index=False)

    def create_historical_directory(self):
        """
        Creates the 'historical' directory.
        """
        if not os.path.exists(self.historical_directory):
            os.makedirs(self.historical_directory)

    def create_historical_zip_files(self):
        """
        Creates zip files in the 'historical' directory, each containing a CSV file with the same name as the folder.
        """
        today = datetime.date.today()
        historical_days = []
        while len(historical_days) < (2 * self.num_business_days):
            today -= datetime.timedelta(days=1)
            if today.weekday() < 5:  # Monday to Friday are business days
                historical_days.append(today.strftime("%Y%m%d"))

        # Only keep the first `num_business_days` days for historical data
        historical_days = historical_days[self.num_business_days:]

        for day in historical_days:
            zip_filename = f"{day}.zip"
            zip_path = os.path.join(self.historical_directory, zip_filename)
            csv_filename = f"{day}.csv"
            csv_path = os.path.join(self.historical_directory, csv_filename)

            # Filter data for the specific date
            df_day = self.df.loc[self.df['as_of_date'] == int(day)].copy()
            df_day.to_csv(csv_path, index=False)

            # Create zip file containing only the CSV file (not a directory)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(csv_path, arcname=csv_filename)

            # Remove the CSV file after zipping
            os.remove(csv_path)


if __name__ == "__main__":
    # Example usage
    base_directory = os.getcwd()  # Use current working directory as base
    sdg = SynthDataGen(base_directory, num_business_days=20, num_unique_ids=15, random_seed=42)

    # Methods to generate data and create directories/files
    sdg.create_data_directory()
    sdg.create_recent_directories()
    sdg.generate_data_frame()
    sdg.make_it_fun(activate=True)
    sdg.create_csv_files_in_recent_directories()
    sdg.create_historical_directory()
    sdg.create_historical_zip_files()
