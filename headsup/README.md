# HeadsUp

## Overview

HeadsUp is a project designed for quantitative stock analysis. It automates the workflow of fetching daily data, analyzing and screening stocks, and generating comprehensive reports with insights, which are emailed to the user.

---

## Features

- **Data Fetching**:
  - Pulls market and financial data using Alpha Vantage API.
  - Fetches historical prices, financial statements, and ETF constituents.
- **Analysis**:
  - Calculates key financial metrics and stock characteristics.
  - Screens stocks for better-than-market characteristics.
- **Plotting**:
  - Plots returns, growth & valuation comps, and financial metrics.
  - Returns PDF with one detail page for each stock screened.
- **Report Generation**:
  - Creates an HTML report summarizing market trends and screened stocks.
- **Visualization**:
  - Prepares data for stock performance plots.
- **Email Notifications**:
  - Sends the generated report to a predefined recipient.

---

## Project Structure

### Files and Modules

- **`requirements.txt`**: Specifies project dependencies.
- **`universe.py`**: Defines the stock and ETF universe to analyze.
- **`settings.py`**: Loads and manages project settings, including API keys and email credentials.
- **`data_fetcher.py`**: Fetches market and financial data via the Alpha Vantage API.
- **`analyzer.py`**: Processes and analyzes data to calculate financial metrics and screen stocks.
- **`plotter.py`**: Prepares data for visualizations.
- **`report_generator.py`**: Generates an HTML report summarizing analysis results.
- **`email_sender.py`**: Sends the generated report via email.

---

## Requirements

- **Python**: >= 3.9
- **Dependencies**: Listed in `requirements.txt`
  - `yfinance`
  - `pandas`
  - `numpy`
  - `python-dotenv`

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Usage

1. **Setup Environment Variables**:
   - Create a `.env` file with the following keys:
     ```env
     VANTAGE_API_KEY=your_api_key_here
     VANTAGE_RPM=5
     SENDER_EMAIL=your_email@example.com
     EMAIL_APP_PASSWORD=your_app_password
     RECIPIENT_EMAIL=recipient_email@example.com
     ```
2. **Run the Project**:
   - Execute the `main.py` file to fetch data, analyze stocks, and generate/send the report.

---

## Workflow

1. Fetch data using the `DataFetcher` class:
   - ETF constituents, historical prices, financial statements.
2. Analyze stocks with the `StockAnalyzer` class:
   - Key metrics and screening logic.
3. Generate an HTML report with `ReportGenerator`.
4. Send the report via email using `EmailSender`.

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## Future Improvements

- Enhance speed of data pull via API calls. Work with Alpha Vantage on improved methods.
- Add ChatGPT API call -- add text to PDF, brief summary of each company screened.
- Add unit tests for key modules.
- Improve error handling in data fetching and API calls.
- Include support for additional data sources.

---

## Contact

For queries, reach out to the maintainers at: [your_email@example.com].
