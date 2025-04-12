# Daily Ticker Update Script

## Description

This Python script automates the process of fetching financial data for a predefined list of stock tickers and major market indices, calculating various technical and fundamental metrics, generating insightful plots and summary tables, and emailing the consolidated report. It utilizes the Alpha Vantage API for data retrieval and Gmail SMTP for email dispatch.

## Features

* **Data Fetching**:
    * Retrieves daily adjusted closing prices, volume, and company overview data from Alpha Vantage.
    * Fetches real-time bulk quotes and appends the latest price to the historical data.
    * Handles potential API errors and rate limiting.
* **Metric Calculation**:
    * Calculates trailing returns for various periods (1d, 1w, 1m, 3m, 6m, 1y, 2y).
    * Computes the Relative Strength Index (RSI, 14-day).
    * Calculates Simple Moving Averages (50, 100, 200 days).
    * Determines the High-Water Mark price/date and current drawdown over a 2-year lookback.
    * Calculates the correlation matrix of returns for the specified tickers.
* **Data Visualization**:
    * Generates scatter plots (e.g., RSI vs. Analyst Target Return, PEGRatio vs. ForwardPE, ROE vs. Profit Margin) colored by sector.
    * Creates heatmaps for recent trailing returns and stock correlation.
    * Plots are embedded directly into the HTML email report.
* **HTML Report Generation**:
    * Formats numerical data appropriately (percentages, currency, large numbers).
    * Creates structured HTML tables for:
        * Major Index Performance
        * Top Weekly Gainers & Losers
        * Valuation Metrics
        * Efficiency Metrics
        * Return & Profitability Metrics
        * Drawdown Analysis
    * Includes basic CSS styling and a simple Table of Contents.
* **Email Automation**:
    * Sends the complete HTML report with embedded images via Gmail.

## Requirements

* Python 3.x
* Required Python Packages:
    * `requests`
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `seaborn`
    * `python-dateutil`
* Alpha Vantage API Key
* Gmail Account Credentials (Email address and App Password for SMTP)

## Configuration

Credentials and API keys can be configured in two ways:

1.  **Local `config.py` file (Recommended for local development)**:
    * Create a `config.py` file in the same directory as the script.
    * Define the following variables within `config.py`:
        ```python
        VANTAGE_API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'
        SENDER_EMAIL = 'your_email@gmail.com'
        EMAIL_APP_PASSWORD = 'your_gmail_app_password'
        RECIPIENT_EMAIL = 'recipient_email@example.com'
        ```
2.  **Environment Variables (Recommended for deployment/CI/CD)**:
    * Set the following environment variables:
        * `VANTAGE_API_KEY`
        * `SENDER_EMAIL`
        * `EMAIL_APP_PASSWORD`
        * `RECIPIENT_EMAIL`
    * The script will fall back to environment variables if `config.py` is not found.

## Usage

1.  **Install Dependencies**:
    ```bash
    pip install requests numpy pandas matplotlib seaborn python-dateutil
    ```
2.  **Configure Credentials**: Set up either `config.py` or environment variables as described above.
3.  **Customize Ticker Lists (Optional)**: Modify the `major_indices`, `coverage_set`, and `watchlist_set` lists within the `if __name__ == "__main__":` block to include the desired stock symbols and ETFs.
4.  **Run the Script**:
    ```bash
    python daily_ticker_update.txt
    ```
    The script will execute, fetch data, perform calculations, generate the report, and send two separate emails (one for the "Coverage" set and one for the "Watchlist" set) to the configured recipient address. The email subject lines will be "Daily Coverage Update - YYYY-MM-DD" and "Daily Watchlist Update - YYYY-MM-DD" respectively.

## Output

The primary output is an HTML email report sent to the specified recipient. The report contains:

* A header section with major index performance.
* Tables showing top/bottom weekly performers.
* Various plots visualizing technical and fundamental data.
* Tables summarizing valuation, efficiency, profitability, and risk metrics.

Logging information (INFO level and above) is printed to the console during execution.

## TODOs / Future Enhancements

* Add sector ETFs to the data pull and use them to calculate market-adjusted returns for individual stocks.
* Implement functionality to input portfolio holdings (ticker and share count) to calculate overall portfolio return.
* Enhance filtering logic (currently placeholder/basic) for selecting tickers for summary tables/plots.

## Disclaimer

Financial data is sourced from Alpha Vantage and provided for informational purposes only. This script does not constitute financial advice.