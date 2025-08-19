"""
    Author: Bruce McNair

    A script to model the risk of a portfolio using general factor model approach.

    inputs needed:
    1. Portfolio Holdings
    2. Factor Loadings
    3. Factor Covariance Matrix
    4. Alpha Volatilities

    can use gen_random_data() to test
"""

import pandas as pd
import numpy as np


class PortfolioRiskModel:
    def __init__(self,
                 df_port: pd.DataFrame,
                 df_factor_loadings: pd.DataFrame,
                 df_factor_covar: pd.DataFrame,
                 df_alpha_vols: pd.DataFrame,
                 factor_model_id: str = None,
                 date: str = None):

        # Validate Inputs, Set Basic Parameters, Initialize Data
        self._validate_inputs(df_port, df_factor_loadings, df_factor_covar, df_alpha_vols)
        self._set_basic_params(df_port, date, factor_model_id, df_factor_loadings)
        self._initialize_data(df_port, df_factor_loadings, df_factor_covar, df_alpha_vols)

        # ====================== factor checks ====================== #
        assert self.factors == df_factor_covar.index.tolist(), "Factors and covariance matrix don't match"
        assert self.factors == df_factor_covar.columns.tolist(), "Factors and covariance matrix don't match"

        # ====================== calculations ====================== #
        # output df
        self.port_risk_model_df = pd.DataFrame()

        # basics
        self.num_names = len(df_port.loc[df_port.mv != 0, 'ticker'].drop_duplicates())
        self.port_gmv = df_port['mv'].abs().sum()
        self.port_nmv = df_port['mv'].sum()
        self.port_vol_total_dollar = None
        self.port_vol_total_pct = None

        # alpha
        self.port_enp = None
        self.port_vol_alpha_dollar = None
        self.port_risk_contribution_alpha = None
        self.ticker_alpha_risk_contribution = pd.Series()
        self.port_alpha_risk_contribution_long = None
        self.port_alpha_risk_contribution_short = None

        # factor
        self.ticker_factor_exp_dollar = pd.DataFrame()
        self.ticker_factor_exp_pctgmv = pd.DataFrame()
        self.ticker_factor_vol_dollar = pd.DataFrame()
        self.ticker_factor_vol_pvtgmv = pd.DataFrame()

        self.factor_exp_dollar = pd.Series()
        self.factor_exp_pctgmv = pd.Series()
        self.factor_vol_dollar = pd.Series()
        self.factor_vol_pctgmv = pd.Series()

        self.port_vol_factor_dollar = None
        self.port_risk_contribution_factor = None

        self.port_factor_risk_contribution = pd.Series()
        self.port_factor_risk_contribution_df = pd.DataFrame()

    def _validate_inputs(self, df_port, df_factor_loadings, df_factor_covar, df_alpha_vols):
        required_cols_port = ['date', 'ticker', 'mv']
        for col in required_cols_port:
            assert col in df_port.columns, f"Missing column in df_port: {col}"

        required_cols_loadings = ['date', 'ticker', 'factor_model', 'factor', 'loading']
        for col in required_cols_loadings:
            assert col in df_factor_loadings.columns, f"Missing column in df_factor_loadings: {col}"

        required_cols_alpha = ['date', 'ticker', 'factor_model', 'alpha_vol']
        for col in required_cols_alpha:
            assert col in df_alpha_vols.columns, f"Missing column in df_alpha_vols: {col}"

        assert np.all(
            df_factor_covar.index == df_factor_covar.columns), "Covar matrix must have factors as index and columns"

    def _set_basic_params(self, df_port, date, factor_model_id, df_factor_loadings):
        if date is not None:
            assert date in df_port['date'].unique()
            if isinstance(date, str):
                self.portfolio_date = pd.to_datetime(date)
            else:
                self.portfolio_date = date
        else:
            self.portfolio_date = df_port['date'].max()

        if factor_model_id is not None:
            assert factor_model_id in df_factor_loadings[
                'factor_model'].unique(), "Factor model not in df_factor_loadings"
            self.factor_model_id = factor_model_id
        else:
            self.factor_model_id = df_factor_loadings['factor_model'].unique()[0]

        self.tickers = df_port['ticker'].drop_duplicates().tolist()

    def _initialize_data(self, df_port, df_factor_loadings, df_factor_covar, df_alpha_vols):
        self.holdings = df_port.loc[df_port['date'] == self.portfolio_date]
        self.ticker_loadings = df_factor_loadings.loc[(df_factor_loadings['date'] == self.portfolio_date) &
                                                      (df_factor_loadings['factor_model'] == self.factor_model_id)]
        self.factors = self.ticker_loadings['factor'].drop_duplicates().tolist()
        self.covar_df = df_factor_covar.loc[self.factors, self.factors]
        self.covar_array = df_factor_covar.loc[self.factors, self.factors].values
        self.alpha_vols = df_alpha_vols.loc[(df_alpha_vols['date'] == self.portfolio_date) &
                                            (df_alpha_vols['factor_model'] == self.factor_model_id)]
        self.factor_vols = np.sqrt(self.covar_array.diagonal())

    def _gen_basics(self):
        df = self.holdings.set_index('ticker').copy()
        df['weight'] = df.mv / self.port_gmv
        df['gmv'] = df.mv.abs()
        df['stock_alpha_vol'] = self.alpha_vols.set_index('ticker')['alpha_vol']
        df['position_alpha_vol'] = df['stock_alpha_vol'] * df.mv
        df['position_alpha_variance'] = df['position_alpha_vol'] ** 2
        df['alpha_risk_contribution'] = df['position_alpha_variance'] / df['position_alpha_variance'].sum()
        self.port_risk_model_df = df.copy()

    def _gen_alpha_risk(self):
        self.port_enp = 1 / (self.port_risk_model_df['alpha_risk_contribution'] ** 2).sum()
        self.port_vol_alpha_dollar = np.sqrt(self.port_risk_model_df['position_alpha_variance'].sum())
        self.ticker_alpha_risk_contribution = self.port_risk_model_df['alpha_risk_contribution'].sort_values(ascending=False)
        self.port_alpha_risk_contribution_long = self.port_risk_model_df.loc[
            self.port_risk_model_df.mv > 0, 'alpha_risk_contribution'].sum()
        self.port_alpha_risk_contribution_short = self.port_risk_model_df.loc[
            self.port_risk_model_df.mv < 0, 'alpha_risk_contribution'].sum()

        self.port_risk_model_df['side'] = np.where(self.port_risk_model_df['mv'] > 0, 'long', 'short')
        self.port_risk_model_df['alpha_risk_contribution'] = self.port_risk_model_df['alpha_risk_contribution']
        self.port_risk_model_df.sort_values('alpha_risk_contribution', ascending=False, inplace=True)

    def _gen_factor_risk(self):

        self.holdings.set_index('ticker', inplace=True)
        loadings = pd.pivot_table(self.ticker_loadings,
                                  index='ticker',
                                  columns='factor',
                                  values='loading')[self.factors]
        self.holdings = self.holdings.join(loadings, how='left')

        # ====================== ticker level ====================== #
        # exposures -- dollar & pct gmv
        self.ticker_factor_exp_dollar = self.holdings.copy()
        self.ticker_factor_exp_dollar[self.factors] = self.ticker_factor_exp_dollar[self.factors].multiply(self.ticker_factor_exp_dollar['mv'], axis=0)
        self.ticker_factor_exp_dollar.drop(columns=['date', 'mv'], inplace=True)
        self.ticker_factor_exp_pctgmv = self.ticker_factor_exp_dollar.copy()
        self.ticker_factor_exp_pctgmv[self.factors] = self.ticker_factor_exp_pctgmv[self.factors].divide(self.port_gmv, axis=0)

        # vols -- dollar & pct gmv
        self.ticker_factor_vol_dollar = self.ticker_factor_exp_dollar * self.factor_vols.T
        self.ticker_factor_vol_pvtgmv = self.ticker_factor_exp_pctgmv * self.factor_vols.T

        # ====================== factor-level ====================== #
        # exposures -- dollar & pct gmv
        self.factor_exp_dollar = self.ticker_factor_exp_dollar.sum()
        self.factor_exp_pctgmv = self.ticker_factor_exp_pctgmv.sum()

        # vols -- dollar & pct gmv
        self.factor_vol_dollar = self.factor_exp_dollar * self.factor_vols
        self.factor_vol_pctgmv = self.factor_exp_pctgmv * self.factor_vols

    def _port_level_risk(self):
        # factor dollar vol
        self.port_vol_factor_dollar = np.sqrt(
            np.dot(
                np.dot(self.factor_exp_dollar.values,
                       self.covar_array),
                self.factor_exp_dollar.values
            )
        )

        # alpha dollar vol
        self.port_vol_alpha_dollar = np.sqrt(self.port_risk_model_df['position_alpha_variance'].sum())

        # total dollar vol
        self.port_vol_total_dollar = np.sqrt(self.port_vol_factor_dollar ** 2 + self.port_vol_alpha_dollar ** 2)

        # factor risk contribution
        self.port_risk_contribution_factor = (self.port_vol_factor_dollar ** 2) / (self.port_vol_total_dollar ** 2)

        # alpha risk contribution
        self.port_risk_contribution_alpha = (self.port_vol_alpha_dollar ** 2) / (self.port_vol_total_dollar ** 2)

        # total vol as pct of gmv
        self.port_vol_total_pct = self.port_vol_total_dollar / self.port_gmv

    def _gen_factor_risk_contribution(self):
        def calculate_individual_factor_risk_contributions(factor_exposures: pd.Series,
                                                           covar_matrix: np.ndarray,
                                                           total_vol: float) -> pd.Series:
            risk_contributions = np.zeros(len(factor_exposures))

            # Calculate each factor's risk contribution
            for i, factor in enumerate(factor_exposures.index):
                masked_exposures = np.zeros_like(factor_exposures.values)
                masked_exposures[i] = factor_exposures.values[i]

                risk_contrib = np.dot(
                    np.dot(masked_exposures, covar_matrix),
                    factor_exposures.values
                ) / (total_vol ** 2)

                risk_contributions[i] = risk_contrib
            return pd.Series(risk_contributions, index=factor_exposures.index)

        self.port_factor_risk_contribution = calculate_individual_factor_risk_contributions(
            factor_exposures=self.factor_exp_dollar,
            covar_matrix=self.covar_array,
            total_vol=self.port_vol_total_dollar
        )
        self.port_factor_risk_contribution_df = self.port_factor_risk_contribution.to_frame(name='risk_cont_pct').copy()
        self.port_factor_risk_contribution_df['side'] = np.where(np.sign(self.factor_exp_pctgmv) == 1, 'long', 'short')
        self.port_factor_risk_contribution_df['factor_vol_bps'] = self.factor_vol_pctgmv * 10000

    def model_risk(self):
        self._gen_basics()
        self._gen_alpha_risk()
        self._gen_factor_risk()
        self._port_level_risk()
        self._gen_factor_risk_contribution()


def gen_random_data(num_names=1500, random_seed=252):
    np.random.seed(random_seed)

    # 1. Portfolio Holdings
    tickers = [f'ticker{i:04}' for i in range(num_names)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=2, freq='B')
    df_port = pd.DataFrame({
        'date': dates[0],
        'ticker': tickers,
        'mv': np.random.uniform(-1e6, 1e6, num_names)
    })
    # make sure all mv values greater than abs(50,000)
    df_port['mv'] = np.where(np.abs(df_port['mv']) < 50000, np.sign(df_port['mv']) * 50000, df_port['mv'])

    # 2. Factor Loadings

    # select factors
    factors_style = ['size', 'value', 'profitability', 'momentum']
    factors_market = ['market_intercept']
    factors_industry = ['tech', 'industrials', 'healthcare', 'consumer', 'energy', 'financials']

    factors = factors_style + factors_market + factors_industry

    # initialize factor loadings dataframe
    df_factor_loadings = pd.DataFrame({
        'date': dates[0],
        'ticker': np.repeat(tickers, len(factors)),
        'factor_model': 'jimmy-us-mh',
        'factor': factors * len(tickers),
    })

    # add factor_group column
    df_factor_loadings['factor_group'] = np.where(df_factor_loadings['factor'].isin(factors_style), 'style',
                                                    np.where(df_factor_loadings['factor'].isin(factors_market), 'market',
                                                             'industry'))
    # all market loadings 1
    if 'market_intercept' in df_factor_loadings['factor'].unique():
        df_factor_loadings.loc[df_factor_loadings['factor'] == 'market_intercept', 'loading'] = 1

    # all industry loadings 0 or 1, one industry per ticker
    for industry in factors_industry:
        df_factor_loadings.loc[df_factor_loadings['factor'] == industry, 'loading'] = 0
    for ticker in tickers:
        industry = np.random.choice(factors_industry)
        df_factor_loadings.loc[(df_factor_loadings['ticker'] == ticker) & (df_factor_loadings['factor'] == industry), 'loading'] = 1

    # all style loadings rand norm 0,1
    for style in factors_style:
        df_factor_loadings.loc[df_factor_loadings['factor'] == style, 'loading'] = np.random.normal(0, 1, len(df_factor_loadings[df_factor_loadings['factor'] == style]))

    # 3. Factor Covariance Matrix
    # add covariances
    factor_covar_values = np.random.uniform(-0.0010, 0.0010, (len(factors), len(factors)))
    factor_covar_values = (factor_covar_values + factor_covar_values.T) / 2
    assert np.all(factor_covar_values == factor_covar_values.T)

    # add variances
    np.fill_diagonal(factor_covar_values, np.random.uniform(0.0005, 0.03, len(factors)))

    df_factor_covar = pd.DataFrame(factor_covar_values, index=factors, columns=factors)

    # update daily variance of market intercept to something realistic
    if 'market_intercept' in df_factor_covar.index:
        df_factor_covar.loc['market_intercept', 'market_intercept'] = 0.23 ** 2  # 23% annualized vol

    # 4. Alpha Volatilities
    df_alpha_vols = pd.DataFrame({
        'date': dates[0],
        'ticker': tickers,
        'factor_model': 'jimmy-us-mh',
        'alpha_vol': np.random.uniform(0.10, 0.35, len(tickers))
    })
    return df_port, df_factor_loadings, df_factor_covar, df_alpha_vols


if __name__ == '__main__':
    # generate random example data
    df_port, df_factor_loadings, df_factor_covar, df_alpha_vols = gen_random_data(random_seed=832)

    # ================================================
    #               Portfolio Risk Model
    # ================================================
    prm = PortfolioRiskModel(df_port, df_factor_loadings, df_factor_covar, df_alpha_vols)
    prm.model_risk()

    print('\nPortfolio Risk Notes')
    print(f'Portfolio Date: {prm.portfolio_date}')
    print(f'Factor Model ID: {prm.factor_model_id}')

    # ====== Portfolio Basics ====== #
    print(f'\nPortfolio Number of Names: {prm.num_names}')
    print(f'Portfolio ENP: {prm.port_enp:.2f}')

    print(f'\nPortfolio GMV ($mm): {prm.port_gmv / 1e6:.2f}')
    print(f'Portfolio NMV ($mm): {prm.port_nmv / 1e6:.2f}')

    print(f'\nPortfolio Total Volatility ($mm): {prm.port_vol_total_dollar / 1e6:.2f}')
    print(f'Portfolio Total Volatility (% GMV): {prm.port_vol_total_pct * 100:.2f}')

    # ====== Alpha Analysis ====== #
    print(f'\nPortfolio Alpha Risk Contribution: {prm.port_risk_contribution_alpha * 100:.2f}%')
    print(f'Alpha Risk (Long): {prm.port_alpha_risk_contribution_long * 100:.2f}%')
    print(f'Alpha Risk (Short): {prm.port_alpha_risk_contribution_short * 100:.2f}%')
    print(f'Top 10 Alpha Risk Contributors (%): '
          f"\n{prm.port_risk_model_df[['side', 'mv', 'weight', 'alpha_risk_contribution']].head(10)}")

    # ====== Factor Analysis ====== #
    print(f'\nPortfolio Factor Risk Contribution: {prm.port_risk_contribution_factor * 100:.2f}%')
    print(f'Top 5 Contributors to Factor Risk: '
          f'\n{prm.port_factor_risk_contribution_df.sort_values("risk_cont_pct", ascending=False).head()}')

    df_factor_covar.to_csv('factor_covar.csv', index=True)