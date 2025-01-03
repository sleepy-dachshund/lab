import pandas as pd
import numpy as np
from config.universe import UNIVERSE


class StockAnalyzer:
    """
    Analyze market, sector, and individual stock data, generating:
      • Index stats
      • Selected stock characteristics
      • Select returns
      • Screened ideas (score-based)
    """

    def __init__(
        self,
        overviews: pd.DataFrame,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        financials: pd.DataFrame,
        etf_dict: dict,
        selected_symbols: list = None,
        add_top_sector_mcaps: bool = True,
        screen_count: int = 20
    ):
        # Basic data
        self.data = data.sort_values(['date', 'market_cap'], ascending=[False, False]).copy()
        self.data_last = self.data[self.data['date'] == self.data['date'].max()].copy()
        self.prices = prices.copy()
        self.financials = financials.copy()
        self.overviews = overviews.copy()
        self.etf_dict = etf_dict.copy()

        self.screen_count = screen_count
        self.days_list = [5, 21, 63, 126, 252, 252 * 3]
        self.summary_cols = [
            'pe_trailing', 'pe_forward', 'fcf_yield', 'pb_trailing', 'ps_trailing',
            'revenue_growth', 'earnings_growth', 'roa', 'roe', 'roic', 'roic_fi',
            'gross_margin', 'operating_margin', 'profit_margin', 'fcf_margin'
        ]

        # Select symbols
        selected_symbols = self._init_symbols(selected_symbols, add_top_sector_mcaps)
        self.selected_symbols = (
            self.data_last.loc[self.data_last['symbol'].isin(selected_symbols)]
            .sort_values('market_cap', ascending=False)['symbol']
            .tolist()
        )

        # Outputs
        self.index_characteristics = pd.DataFrame()
        self.index_returns = pd.DataFrame()
        self.sector_characteristics = pd.DataFrame()
        self.select_characteristics = pd.DataFrame()
        self.select_returns = pd.DataFrame()
        self.screener = pd.DataFrame()
        self.screener_desc = pd.DataFrame()

    def _init_symbols(self, selected_symbols, add_top_sector_mcaps):
        """Set or expand selected symbols with top sector names."""
        if selected_symbols is None:
            # Top 5 from each sector
            selected_symbols = []
            for sector in self.data['sector'].unique():
                top5 = (
                    self.data_last[self.data_last['sector'] == sector]
                    .nlargest(5, 'market_cap')['symbol']
                    .tolist()
                )
                selected_symbols += top5
        elif add_top_sector_mcaps:
            # Supplement with top 3 if not already included
            for sector in self.data['sector'].unique():
                top3 = (
                    self.data_last[self.data_last['sector'] == sector]
                    .nlargest(3, 'market_cap')['symbol']
                    .tolist()
                )
                selected_symbols += top3
        return list(set(selected_symbols))

    def calculate_index_stats(self):
        """Calculate index characteristics and returns for SPY and SPY ex-top 10."""
        def calc_index_characteristics(etf_df: pd.DataFrame, name: str) -> pd.DataFrame:
            df = etf_df.copy()

            for col in self.summary_cols:
                # Clip outliers of pre-defined columns
                df[col] = df[col].clip(
                    lower=df[col].quantile(0.05),
                    upper=df[col].quantile(0.95)
                )

            # Initiate row
            row = pd.DataFrame(index=[name])
            row['Names'] = df.shape[0]

            # Calculate the aggregate index sums
            total_mcap = df['market_cap'].sum() if df['market_cap'].max() < 1e9 else df['market_cap'].sum() / 1e9
            total_ni = df['net_income'].sum()  # trailing earnings
            total_rev = df['revenue'].sum()
            total_equity = df['shareholder_equity'].sum()
            total_fcf = df['fcf'].sum()

            row['pe_trailing'] = (
                total_mcap / total_ni if total_ni > 0 else np.nan
            )
            row['fcf_yield'] = (
                total_fcf / total_mcap if total_mcap > 0 else np.nan
            )
            row['pb_trailing'] = (
                total_mcap / total_equity if total_equity > 0 else np.nan
            )
            row['ps_trailing'] = (
                total_mcap / total_rev if total_rev > 0 else np.nan
            )

            # Calculate weighted sums
            row['pe_forward'] = (
                    df['pe_forward'].fillna(99) * df['weight']
            ).sum()
            row['revenue_growth'] = (
                    df['revenue_growth'].fillna(0) * df['weight']
            ).sum()
            row['earnings_growth'] = (
                    df['earnings_growth'].fillna(0) * df['weight']
            ).sum()

            for col in ['roa', 'roe', 'roic', 'roic_fi',
                        'gross_margin', 'operating_margin', 'profit_margin', 'fcf_margin']:
                row[col] = (df[col].fillna(0) * df['weight']).sum()

            row['over50dma'] = (df['price'] > df['50dma']).mean()
            row['over200dma'] = (df['price'] > df['200dma']).mean()
            return row

        # Merge and set weights
        spy = self.etf_dict['SPY'].merge(self.overviews, on='symbol', how='left', suffixes=('', '_ovr'))
        spy = spy.merge(self.data_last, on='symbol', how='left', suffixes=('', '_data'))
        spy['weight'] /= spy['weight'].sum()

        spyx10 = spy.sort_values('market_cap', ascending=False).iloc[10:].copy()
        spyx10['weight'] /= spyx10['weight'].sum()

        # S&P Summaries
        spy_stats = calc_index_characteristics(spy, 'SPY')
        spyx10_stats = calc_index_characteristics(spyx10, 'SPYx10')
        self.index_characteristics = pd.concat([spy_stats, spyx10_stats], axis=0).T

        # Sector ETF Summaries
        sector_maps = {'XLC': 'Communication Services',
                       'XLY': 'Consumer Discretionary',
                       'XLP': 'Consumer Staples',
                       'XLE': 'Energy',
                       'XLF': 'Financials',
                       'XLV': 'Health Care',
                       'XLI': 'Industrials',
                       'XLB': 'Materials',
                       'XLRE': 'Real Estate',
                       'XLK': 'Technology',
                       'XLU': 'Utilities'}

        self.sector_characteristics = pd.DataFrame()
        for sector_etf in UNIVERSE['sector_etfs']:
            sector_etf_df = self.etf_dict[sector_etf].merge(self.overviews, on='symbol', how='left', suffixes=('', '_ovr')).copy()
            sector_etf_df = sector_etf_df.merge(self.data_last, on='symbol', how='left', suffixes=('', '_data'))
            sector_etf_df['weight'] /= sector_etf_df['weight'].sum()
            sector_etf_stats = calc_index_characteristics(sector_etf_df, sector_etf)
            sector_etf_stats['name'] = sector_maps[sector_etf]
            sector_etf_stats.reset_index(drop=False, inplace=True)
            sector_etf_stats.rename(columns={'index': 'symbol'}, inplace=True)
            self.sector_characteristics = pd.concat([self.sector_characteristics, sector_etf_stats], axis=0)
        self.sector_characteristics.sort_values('Names', ascending=False, inplace=True)
        self.sector_characteristics.reset_index(drop=True, inplace=True)

        # S&P Returns
        spy_rets = self.prices[self.prices['symbol'] == 'SPY'].copy()
        spy_rets = spy_rets.sort_values('date', ascending=False).reset_index(drop=True)
        last_price = spy_rets['price'].iloc[0]

        idx_ret = pd.DataFrame(index=['SPY'])
        for d in self.days_list:
            idx_ret[f'ret_{d}d'] = last_price / spy_rets['price'].iloc[d + 1] - 1
        idx_ret['vol_252d'] = spy_rets['return'].iloc[:253].std() * np.sqrt(252)
        idx_ret['z_21d'] = idx_ret['ret_21d'] / (idx_ret['vol_252d'] / np.sqrt(21))
        idx_ret['z_63d'] = idx_ret['ret_63d'] / (idx_ret['vol_252d'] / np.sqrt(63))
        self.index_returns = idx_ret.T

        return self.index_characteristics, self.index_returns, self.sector_characteristics

    def grab_select_characteristics(self):
        """Retrieve key characteristics for selected stocks."""
        sel = self.data_last[self.data_last['symbol'].isin(self.selected_symbols)].copy()
        sel = sel.merge(self.overviews, on='symbol', how='left', suffixes=('', '_ovr'))
        cols = ['symbol', 'sector', 'price', 'market_cap', '50dma', '200dma'] + self.summary_cols
        self.select_characteristics = sel[cols].copy()

    def grab_select_returns(self):
        """Collect returns for selected symbols vs SPY."""
        rets_list = []
        for sym in self.selected_symbols + ['SPY']:
            df = self.prices[self.prices['symbol'] == sym].copy()
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            last_price = df['price'].iloc[0]

            row = pd.DataFrame(index=[sym])
            for d in self.days_list:
                row[f'ret_{d}d'] = last_price / df['price'].iloc[d + 1] - 1
            row['vol_252d'] = df['return'].iloc[:253].std() * np.sqrt(252)
            row['z_21d'] = row['ret_21d'] / (row['vol_252d'] / np.sqrt(21))
            row['z_63d'] = row['ret_63d'] / (row['vol_252d'] / np.sqrt(63))
            rets_list.append(row)

        self.select_returns = pd.concat(rets_list, axis=0).T
        self.select_returns = self.select_returns[['SPY'] + self.selected_symbols].T
        return self.select_returns

    def screen_stocks(self):
        """Screen for better-than-market characteristics."""
        bogey_tracker = self.data_last.merge(
            self.overviews, on='symbol', how='left', suffixes=('_data', '')
        )

        # High-volume hit rate
        def calc_high_volume_hit_rate(df: pd.DataFrame, trailing_yrs: int = 2, top_n: int = 20):
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            cutoff = df['date'].max() - pd.DateOffset(years=trailing_yrs)
            df = df[df['date'] >= cutoff]
            topvol = df.groupby('symbol', group_keys=False).apply(lambda x: x.nlargest(top_n, 'volume'))
            hits = (topvol.assign(over_zero=(topvol['return'] > 0).astype(int))
                    .groupby('symbol')['over_zero'].mean().reset_index())
            hits.rename(columns={'over_zero': 'high_volume_hit_rate'}, inplace=True)
            hits['hit_rate_percentile'] = hits['high_volume_hit_rate'].rank(pct=True)
            return hits.drop(columns='high_volume_hit_rate')

        hvr = calc_high_volume_hit_rate(self.data)
        bogey_tracker = bogey_tracker.merge(hvr, on='symbol', how='left')

        # Score columns
        cash_flow = [
            'earnings_growth', 'revenue_growth', 'roa', 'roe', 'roic',
            'roic_fi', 'fcf_yield', 'gross_margin', 'operating_margin',
            'profit_margin', 'fcf_margin'
        ]
        valuation = ['pe_trailing', 'pe_forward', 'pb_trailing', 'ps_trailing']
        other = ['50over200dma', 'analyst_score', 'hit_rate_percentile']

        # Bogeys from index stats
        bx = self.index_characteristics.T.copy()
        bogeys = {k: bx[k].min() for k in valuation}
        bogeys.update({k: bx[k].max() for k in cash_flow})
        bogeys.update({'50over200dma': True,
                       'analyst_score': 0.5,
                       'hit_rate_percentile': 0.667})

        # Hits vs bogeys
        for col, bgy in bogeys.items():
            if col in cash_flow:
                bogey_tracker[f'{col}_bogey'] = bogey_tracker[col] > bgy
            elif col in valuation:
                bogey_tracker[f'{col}_bogey'] = bogey_tracker[col] < bgy
            elif col == '50over200dma':
                bogey_tracker['50over200dma_bogey'] = bogey_tracker['50dma'] > bogey_tracker['200dma']
            elif col == 'hit_rate_percentile':
                bogey_tracker['hit_rate_percentile_bogey'] = bogey_tracker['hit_rate_percentile'] > bgy

        # Analyst score
        bogey_tracker['analyst_score'] = (
            bogey_tracker['analyst_strong_buy'].fillna(0)
            + bogey_tracker['analyst_buy'].fillna(0)
            - bogey_tracker['analyst_strong_sell'].fillna(0)
            - bogey_tracker['analyst_sell'].fillna(0)
        )
        denom = [
            'analyst_strong_buy', 'analyst_buy', 'analyst_hold',
            'analyst_strong_sell', 'analyst_sell'
        ]
        bogey_tracker['analyst_score'] /= bogey_tracker[denom].sum(axis=1).replace(0, np.nan)
        bogey_tracker['analyst_score_bogey'] = bogey_tracker['analyst_score'] > bogeys['analyst_score']

        # Aggregate scores
        bogey_cols = [c for c in bogey_tracker.columns if c.endswith('_bogey')]
        bogey_tracker['score'] = bogey_tracker[bogey_cols].mean(axis=1)
        cash_cols = [c for c in bogey_cols if c.replace('_bogey', '') in cash_flow]
        val_cols = [c for c in bogey_cols if c.replace('_bogey', '') in valuation]
        oth_cols = [c for c in bogey_cols if c.replace('_bogey', '') in other]

        bogey_tracker['score_cash_flow'] = bogey_tracker[cash_cols].mean(axis=1)
        bogey_tracker['score_valuation'] = bogey_tracker[val_cols].mean(axis=1)
        bogey_tracker['score_other'] = bogey_tracker[oth_cols].mean(axis=1)
        bogey_tracker['market_cap'] /= 1e9

        # Merge scores
        merge_cols = ['symbol', 'score', 'score_cash_flow', 'score_valuation', 'score_other', 'hit_rate_percentile']
        self.data_last = self.data_last.merge(bogey_tracker[merge_cols], on='symbol', how='left')
        self.select_characteristics = self.select_characteristics.merge(bogey_tracker[merge_cols], on='symbol', how='left')

        # Final screener
        bogey_tracker = bogey_tracker.sort_values(['score', 'market_cap'], ascending=[False, False]).head(self.screen_count)
        keep_cols = [
            'symbol', 'name', 'sector', 'score', 'score_cash_flow', 'score_valuation', 'score_other',
            'price', 'market_cap', 'analyst_score', 'hit_rate_percentile', 'description'
        ] + self.summary_cols
        bogey_tracker = bogey_tracker[keep_cols]

        self.screener_desc = bogey_tracker[['symbol', 'sector', 'market_cap', 'score', 'description']]
        self.screener = bogey_tracker.drop(columns=['description'])

        # Index comps
        temp = self.index_characteristics[['SPY', 'SPYx10']].T.copy()
        temp.rename(columns={'roic_operating': 'roic', 'roic_financing': 'roic_fi'}, inplace=True)
        temp.drop(columns=[col for col in temp.columns if col not in self.screener.columns], inplace=True)
        self.screener = pd.concat([self.screener, temp], axis=0, ignore_index=False)
        self.screener.loc['SPYx10', 'symbol'] = 'INDEX SPYx10'
        self.screener.loc['SPY', 'symbol'] = 'INDEX SPY'
        self.screener.reset_index(drop=True, inplace=True)

        return self.screener, self.screener_desc, self.select_characteristics
