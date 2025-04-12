# Daily Market Report Generator

This project generates a daily financial report for specified ETFs and tickers and sends it via email. It's designed to be run automatically using GitHub Actions.

## Features

- Fetches daily price data for SPY, QQQ, SPW, and IWM using Alpha Vantage API
- Analyzes 10 years of historical data 
- Generates the following visualizations:
  - Price chart with 50, 100, and 200-day moving averages
  - Wilder's RSI with 70/30 lines
  - Rolling volatility (10, 30, 60, 90 day)
  - Fast RSI indicators (9, 14, 30 day)
  - Rolling drawdown (percentage and volatility-adjusted)
- Calculates performance metrics:
  - Last price
  - YTD, 1w, 1m, 3m, 6m, and 1y returns
  - Historical percentiles for each return period
- Sends a formatted HTML email report with embedded visualizations

## Setup

### Prerequisites

- Python 3.8 or higher
- Alpha Vantage API key
- Google account with app password configured

### Local Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/financial-report.git
   cd financial-report
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create environment variables or update the script with your API keys and email credentials:
   ```
   export VANTAGE_API_KEY='your_api_key'
   export SENDER_EMAIL='your_email@gmail.com'
   export EMAIL_APP_PASSWORD='your_app_password'
   export RECIPIENT_EMAIL='recipient_email@example.com'
   ```

4. Run the script:
   ```
   python daily_market_update.py
   ```

### GitHub Actions Setup

To run this automatically using GitHub Actions:

1. Fork/push this repository to your GitHub account

2. Set up the following secrets in your GitHub repository:
   - `VANTAGE_API_KEY`: Your Alpha Vantage API key
   - `SENDER_EMAIL`: Your Gmail address
   - `EMAIL_APP_PASSWORD`: Your Google account app password
   - `RECIPIENT_EMAIL`: Email address to receive reports

3. The workflow is configured to run daily at 6:00 AM EST and 5:30pm EST. You can adjust the timing in the `.github/workflows/daily-report.yml` file.

## Extending the Project

### Adding More Symbols

Edit the `etfs` list in the `main()` function to add more symbols:

```python
symbols = ['SPY', 'QQQ', 'SPW', 'IWM', 'ANOTHER_SYMBOL']
```

### Adding PDF Report Generation

Future enhancements can include generating a PDF report using libraries like `reportlab` or `weasyprint`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.