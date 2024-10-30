# data loader
This is an example data pipeline that fetches historical and/or live data for a given date, compares it to existing data for integrity, and
appends that new data to an existing CSV file (obviously not the best file type for this but this is just an example). 

The pipeline is designed to be run daily, and the input date is expected to a business day. But you could input any day and it will act as if 
it were running on that day by appending the data received on that day.

The pipeline is designed to be run in a production environment, where the input date is passed as an argument in the terminal.

If you want to run, set test_run to False and terminal_run to True. This will allow you to run the pipeline from the
terminal. If you want to run the pipeline for a range of dates, set test_run to True and it will pull data for 
all dates in the range and append them to one file.

If you need data for testing this, you can use the data_gen.py script in this repo to generate synthetic data in the necessary directory scheme.
