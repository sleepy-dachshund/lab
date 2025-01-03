import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PyPDF2 import PdfMerger
import io


class StockPlotter:
    def __init__(self,
                 ticker_list: list,
                 data: pd.DataFrame,
                 prices: pd.DataFrame,
                 financials: pd.DataFrame,
                 overviews: pd.DataFrame,
                 screener: pd.DataFrame,
                 sectors: pd.DataFrame,
                 etf_dict: dict,
                 universe: dict,
                 start_date: str = '2019-01-01'):

        self.start_date = start_date
        self.ticker_list = ticker_list

        self.data = data.copy()
        self.prices = prices.copy()
        self.financials = financials.copy()
        self.overviews = overviews.copy()
        self.screener = screener.copy()
        self.sectors = sectors.copy()
        self.etf_dict = etf_dict.copy()
        self.UNIVERSE = universe

        self.ticker_screen = pd.DataFrame()
        self.ticker_overview = pd.DataFrame()
        self.ticker_data = pd.DataFrame()
        self.ticker_financials = pd.DataFrame()
        self.df_prices = pd.DataFrame()

        self.ticker_start = None
        self.comp_etf = None

        self.all_figs = []

    def prep_for_single_ticker(self, ticker):
        self.ticker_screen = self.screener[self.screener['symbol'] == ticker].copy()
        self.ticker_overview = self.overviews.loc[self.overviews.symbol == ticker].copy()
        self.ticker_data = self.data.loc[self.data.symbol == ticker].copy()
        self.ticker_financials = self.financials.loc[self.financials.symbol == ticker].copy()

        # ======= Prep: Find which Sector ETF the stock is in ======= #
        sector_etfs = self.UNIVERSE['sector_etfs']
        for sector_etf in sector_etfs:
            if ticker in self.etf_dict[sector_etf].symbol.values:
                self.comp_etf = sector_etf
                break
        if self.comp_etf is None:
            self.comp_etf = 'IWM'

        self.df_prices = self.prices.loc[self.prices.symbol.isin([ticker, 'SPY', self.comp_etf])].copy().reset_index(drop=True)
        self.df_prices['date'] = pd.to_datetime(self.df_prices['date'])
        self.df_prices = self.df_prices.loc[self.df_prices['date'] >= self.start_date]
        self.df_prices.dropna(subset=['price'], inplace=True)
        self.ticker_start = self.df_prices.date.min().strftime('%Y-%m-%d')
        self.df_prices = self.df_prices.sort_values('date')

    def price_chart(self):
        # ======= Figure 1: Plot, Price Chart w/ SPY & Sector ETF ======= #
        assert 'SPY' in self.prices.symbol.values, 'SPY not in prices'
        assert self.comp_etf in self.prices.symbol.values, f'{self.comp_etf} not in prices'
        df = self.df_prices.copy().reset_index(drop=True)

        df['symbol'] = df['symbol'].replace({'SPY': 'S&P 500', self.comp_etf: self.comp_etf})
        df = pd.pivot_table(df, values='price', index='date', columns='symbol')
        df = (df / df.iloc[0]) * 100

        self.fig1, ax = plt.subplots(figsize=(10, 6))
        df.plot(ax=ax)
        ax.grid(axis='y')
        ax.set_ylabel(f'Price (Normalized, {self.ticker_start} = 100)')
        ax.set_xlabel('Date')
        plt.tight_layout()

    def technical_chart(self, ticker):
        # ======= Figure 2: Stock Price Chart w/ Moving Averages, RSI, & BB ======= #
        df = self.df_prices.loc[self.df_prices.symbol == ticker].copy().reset_index(drop=True)

        # moving averages
        df['MA50'] = df['price'].rolling(window=50).mean()
        df['MA200'] = df['price'].rolling(window=200).mean()

        # bollinger bands (20-day)
        df['MA20'] = df['price'].rolling(window=20).mean()
        df['STD20'] = df['price'].rolling(window=20).std()
        df['UpperBB'] = df['MA20'] + 2 * df['STD20']
        df['LowerBB'] = df['MA20'] - 2 * df['STD20']

        # RSI (14-day)
        window = 14
        delta = df['price'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window).mean()
        avg_loss = pd.Series(loss).rolling(window).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Plot
        self.fig2, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))

        # Top: price + MAs + Bollinger
        ax1.plot(df['date'], df['price'], label='Price')
        ax1.plot(df['date'], df['MA50'], label='50DMA')
        ax1.plot(df['date'], df['MA200'], label='200DMA')
        ax1.fill_between(df['date'], df['UpperBB'], df['LowerBB'], alpha=0.2, label='Bollinger')
        ax1.set_ylabel('Price')
        ax1.grid(axis='y')
        ax1.legend(loc='upper left')

        # Bottom: RSI
        ax2.plot(df['date'], df['RSI'], color='orange', label='RSI')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')

        plt.tight_layout()

    def valuation_tables(self, ticker):
        # ======= Figure 3: Valuation Ratios Table ======= #
        df = self.screener.copy()
        df = pd.concat([df, self.sectors.reset_index(drop=False).rename(columns={'ETF': 'symbol'})], axis=0)
        df = df.loc[
            (df.symbol == ticker) | (df.symbol.str.contains('INDEX')) | (df.symbol.str.contains(self.comp_etf))].copy()

        df = df.loc[:, ['symbol', 'score', 'score_cash_flow', 'score_valuation', 'analyst_score', 'hit_rate_percentile',
                          'pe_trailing', 'pe_forward', 'ps_trailing', 'pb_trailing',
                          'revenue_growth', 'earnings_growth', 'fcf_yield', 'roa', 'roe', 'roic', 'roic_fi',
                          'gross_margin', 'operating_margin', 'profit_margin', 'fcf_margin']].copy()
        dfa = df.loc[df.symbol == ticker, ['symbol', 'score', 'score_cash_flow', 'score_valuation', 'analyst_score', 'hit_rate_percentile']].copy()
        dfb = df.loc[:, ['symbol', 'pe_trailing', 'pe_forward', 'ps_trailing', 'pb_trailing']].set_index('symbol').T.copy()
        dfc = df.loc[:, ['symbol', 'revenue_growth', 'earnings_growth', 'fcf_yield', 'roa', 'roe', 'roic', 'roic_fi']].set_index('symbol').T.copy()
        dfd = df.loc[:, ['symbol', 'gross_margin', 'operating_margin', 'profit_margin', 'fcf_margin']].set_index('symbol').T.copy()

        dfa = dfa.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x if isinstance(x, str) else np.nan)

        self.fig3, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 12))

        axs[0].axis('off')
        axs[0].table(
            cellText=dfa.values,
            colLabels=dfa.columns,
            cellLoc='center',
            loc='center',
            colLoc='center'
        )
        # table.scale(0.8, 1.2)

        x_labels = ['Valuations', 'Returns', 'Margins']
        for i, df in enumerate([dfb, dfc, dfd], start=1):
            df.plot(kind='bar', ax=axs[i])
            axs[i].set_title('Index Comparison')
            axs[i].set_xlabel(x_labels[i - 1])
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right')
            axs[i].grid(axis='y')
            axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

    def peer_comparison_chart(self):
        # ======= Figure 4: Highlights of Key Metrics vs. Peers ======= #
        df = self.data.loc[self.data.date == self.data.date.max()].copy()
        df = df.merge(self.overviews, on='symbol', how='left', validate='1:1', suffixes=('', 'overview')).copy()
        df = df.loc[df.sector == self.ticker_overview.sector.values[0]].copy()
        df['mcap_diff'] = (df['market_cap'] - self.ticker_overview['market_cap'].values[0]).abs()
        df = df.sort_values('mcap_diff').head(6).copy()
        df = df.loc[:, ['symbol',
                          'pe_trailing', 'pe_forward', 'ps_trailing', 'pb_trailing',
                          'revenue_growth', 'earnings_growth', 'fcf_yield', 'roa', 'roe', 'roic', 'roic_fi',
                          'gross_margin', 'operating_margin', 'profit_margin', 'fcf_margin']].copy()
        dfa = df.loc[:, ['symbol', 'pe_trailing', 'pe_forward', 'ps_trailing', 'pb_trailing', ]].set_index(
            'symbol').T.copy()
        dfb = df.loc[:, ['symbol', 'revenue_growth', 'earnings_growth', 'fcf_yield', 'roa', 'roe', 'roic', 'roic_fi']].set_index('symbol').T.copy()
        dfc = df.loc[:, ['symbol', 'gross_margin', 'operating_margin', 'profit_margin', 'fcf_margin']].set_index(
            'symbol').T.copy()

        self.fig4, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

        x_labels = ['Valuations', 'Returns', 'Margins']
        for i, df in enumerate([dfa, dfb, dfc], start=1):
            df.plot(kind='bar', ax=axs[i - 1])
            axs[i - 1].set_xlabel(x_labels[i - 1])
            axs[i - 1].set_title('Peer Comparison')
            axs[i - 1].set_xticklabels(axs[i - 1].get_xticklabels(), rotation=45, ha='right')
            axs[i - 1].grid(axis='y')
            axs[i - 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    def profitability_growth_chart(self, ticker):
        # ======= Figure 5: Profitability & Growth Chart / Table ======= #
        df = self.ticker_financials.copy()
        df = df.loc[df.date >= self.start_date].copy()
        df = df.loc[:, ['date', 'symbol', 'revenue', 'gross_profit', 'operating_income', 'net_income', 'fcf']].copy()
        df = df.set_index('date').copy()

        # plot
        self.fig5, ax = plt.subplots(figsize=(10, 6))
        assert df.symbol.nunique() == 1, 'Multiple symbols in profitability_growth_chart df'
        assert df.symbol.unique()[0] == ticker, 'Symbol in profitability_growth_chart df does not match ticker'
        df = df.drop(columns='symbol')
        df.plot(ax=ax)
        ax.set_ylabel('USD (TTM, $B)')
        ax.set_xlabel('Date')
        plt.tight_layout()

    def financial_health_chart(self, ticker):
        # ======= Figure 6: Financial Health Indicators ======= #
        df = self.ticker_financials.copy()
        df = df.loc[df.date >= self.start_date].copy()
        df = df.loc[:, ['date', 'symbol', 'assets', 'liabilities', 'cash']].copy()
        df = df.set_index('date').copy()

        # plot
        self.fig6, ax = plt.subplots(figsize=(10, 6))
        assert df.symbol.nunique() == 1, 'Multiple symbols in financial_health_chart df'
        assert df.symbol.unique()[0] == ticker, 'Symbol in financial_health_chart df does not match ticker'
        df = df.drop(columns='symbol')
        df.plot(ax=ax)
        ax.set_ylabel('USD ($B)')
        ax.grid(axis='y')
        ax.set_xlabel('Date')
        plt.tight_layout()

    def combine_figures_for_ticker(self, figs, nrows=3, ncols=2, figsize=(16, 24)):
        """
        Combines 6 figures into a single figure for one ticker.

        Args:
            figs: list of matplotlib figure objects for a single ticker
            nrows, ncols: layout of the subplots
            figsize: size of the combined figure

        Returns:
            Combined matplotlib figure object
        """
        combined_fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()

        for i, fig_obj in enumerate(figs):
            # Convert each figure to an image
            buf = io.BytesIO()
            fig_obj.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)

            # Display the image in the appropriate subplot
            img = plt.imread(buf)
            axs[i].imshow(img)
            axs[i].axis('off')

            # Clean up
            buf.close()
            plt.close(fig_obj)

        plt.tight_layout()
        return combined_fig

    def plot_all(self):
        """
        Creates plots for all tickers and saves each ticker's combined figure to PDF.
        """
        # Create a temporary PDF for each ticker
        temp_pdfs = []

        for ticker in self.ticker_list:
            # Create figures for current ticker
            ticker_figs = []
            self.prep_for_single_ticker(ticker)

            # Generate all plots for current ticker
            self.price_chart()
            self.technical_chart(ticker)
            self.valuation_tables(ticker)
            self.profitability_growth_chart(ticker)
            self.financial_health_chart(ticker)
            self.peer_comparison_chart()

            # Collect all figures
            ticker_figs.extend([
                self.fig1, self.fig2, self.fig3,
                self.fig4, self.fig5, self.fig6
            ])

            # Combine figures into one page
            combined_fig = self.combine_figures_for_ticker(ticker_figs)

            # Create a temporary PDF for this ticker
            temp_pdf = io.BytesIO()
            with PdfPages(temp_pdf) as pdf:
                pdf.savefig(combined_fig)
            temp_pdf.seek(0)
            temp_pdfs.append((ticker, temp_pdf))

            # Clean up the combined figure
            plt.close(combined_fig)

        # Combine all PDFs into final output
        self.create_final_pdf(temp_pdfs)

    def create_final_pdf(self, temp_pdfs, output_filename='screener_details.pdf'):
        """
        Combines all temporary PDFs into a single output PDF.

        Args:
            temp_pdfs: list of tuples (ticker, BytesIO object) containing temporary PDFs
            output_filename: name of the final PDF file
        """

        merger = PdfMerger()

        try:
            for ticker, temp_pdf in temp_pdfs:
                merger.append(fileobj=temp_pdf)
                merger.add_outline_item(title=ticker, page_number=len(merger.pages) - 1)
            merger.write(output_filename)

        finally:
            merger.close()
            for _, temp_pdf in temp_pdfs:
                temp_pdf.close()

    def save_plots(self):
        self.plot_all()

