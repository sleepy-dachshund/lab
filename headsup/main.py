from config.universe import UNIVERSE
from src.data_fetcher import DataFetcher
from src.analyzer import StockAnalyzer
from src.plotter import StockPlotter
from src.report_generator import ReportGenerator
from src.email_sender import EmailSender
from datetime import datetime
import pandas as pd
import numpy as np


def main():
    # Params
    pull_fresh_data = True
    etf_list = UNIVERSE['major_indices'] + UNIVERSE['sector_etfs']
    select_symbols = UNIVERSE['stocks']

    # Fetch data from AV API
    fetcher = DataFetcher(additional_symbols=select_symbols)
    overviews, prices, financials, data, etf_dict = fetcher.fetch_all_data(etf_list=etf_list,
                                                                           read_cache=(not pull_fresh_data))

    # Analyze data
    analyzer = StockAnalyzer(overviews=overviews, data=data, prices=prices, financials=financials, etf_dict=etf_dict,
                             selected_symbols=select_symbols, add_top_sector_mcaps=True, screen_count=50)
    index_characteristics, index_returns = analyzer.calculate_index_stats()
    analyzer.grab_select_characteristics()
    select_returns = analyzer.grab_select_returns()
    screener, screener_desc, select_characteristics = analyzer.screen_stocks()

    # # Plot data
    # plotter = StockPlotter(data, prices)

    # Generate report
    report_gen = ReportGenerator(ic=index_characteristics,
                                 ir=index_returns,
                                 sc=select_characteristics,
                                 sr=select_returns,
                                 screener=screener,
                                 screener_desc=screener_desc)
    report_html = report_gen.generate_html_report()

    # Send email
    email_sender = EmailSender()
    email_sender.send_report(report_html)


if __name__ == "__main__":
    main()
