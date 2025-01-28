

'''

    Author: Bruce McNair

    Instructions:
        - You can just run this script in isolation as long as you have an internet connection.
        - If you want, you can adjust parameters (lookback period, weighting scheme, etc.) in the main block at the bottom.
        - The script will cache a .csv file of prices and a .png image of the momentum factor returns in the local directory.
        - Only very basic packages required: pandas, numpy, yfinance, matplotlib.

    Notes & Improvements:
        - Find access to a more comprehensive estimation universe, sp500 alone isn't enough.
        - Make sure you have clean and accurate price & return data, yfinance isn't bad though.
        - Confirm weighting schema. the systematic factor (what we have here) is one thing, 
            but calculating a risk factor that we want to hedge to is another.
            Think about characteristics: drift, volatility, toxicity, turnover, correlation to original systematic factor, intuitive loadings, etc.
            This is a big area of research.
        - Don't have to calc factor spread by top_quantile - bottom_quantile, could do some other grouping.
            - the originals, Jegadeesh & Titman (1993), did do top decile - bottom decile though.
        - Should not be taking simple average of returns within quantiles to calculate top/bottom portfolio returns.
            - Typically weight by mcap or sqrt(mcap).
        - Quantile is also "rebalanced" here daily, which is not realistic for trading but still informative for hedging.
        - Note max DD calcs are not perfect, particularly the xVol units -- that's a more sophisticated calc but outside the scope here.
        - Correlation calc is simple, but time series relationship can be more sophisticated, e.g., calc rolling beta to factor.

        Lastly:
        - You could set up a loop to test different params, and compare the resulting return .pngs and summary stats.
        - You could also deviate further from the simple, systematic momentum factor, and/or combine many characteristics into a single factor, and trade that as a "momentum alpha." But that is way outside the scope of this.

'''

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


''' ===============================================================================================================
        1. Pull estimation universe
=============================================================================================================== '''


def pull_estimation_universe(index: str = 'sp500'):
    """
    scrape S&P 500 tickers from wikipedia by default, or use a csv file.
    """
    if index.lower() == 'sp500':
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    else:
        return pd.read_csv(f'{index}.csv')['Ticker'].tolist()


''' ===============================================================================================================
        2. Pull daily prices and calculate daily returns
=============================================================================================================== '''


def pull_daily_prices_of_etsu(tickers, start='2020-01-01', end=None, try_cache=True):
    """
    pull daily adj. close prices from yfinance. check local cache first by default.
    """

    def pull_fresh_prices(tickers, start, end):
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        if len(tickers) == 1:
            data = data.to_frame()
        data.to_csv('prices.csv')
        return data

    if not tickers:
        return pd.DataFrame()

    if try_cache:
        try:
            data = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
            data = data[tickers]
            if end:
                data = data.loc[start:end]
            else:
                data = data.loc[start:]
            return data
        except FileNotFoundError:
            data = pull_fresh_prices(tickers, start, end)
            data.to_csv('prices.csv')
            return data
    else:
        data = pull_fresh_prices(tickers, start, end)
        data.to_csv('prices.csv')
        return data


def calc_daily_returns_of_etsu(price_df):
    """
    simple daily pct_change returns
    """
    return price_df.pct_change().fillna(0)


''' ===============================================================================================================
        3. Calculate momentum score (weighted sum of past returns)
=============================================================================================================== '''


def calc_momentum_score(returns_df,
                        lookback=252,
                        weighting_scheme='trapezoidal',
                        lag=21,
                        custom_weights=None):
    """
    compute a weighted sum of past returns for every ticker every day.
    - 'uniform': equally weight last X days
    - 'trapezoidal': linearly increasing weights to plateau then linearly decreasing (Axioma)
    - 'exp': exponential decay weighting (BARRA)
    - 'custom': can pass your own array
    lag: number of days on front end left out of score calc, given 0% weight
    """
    momentum_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    weights = np.zeros(lookback)

    for i, date in enumerate(returns_df.index):
        if i < lookback:
            continue  # not enough lookback data, could use some other estimation or fill value here
        window = returns_df.iloc[i - lookback:i]

        if weighting_scheme == 'uniform':
            weights = np.ones(lookback) / lookback

        elif weighting_scheme == 'trapezoidal':
            # set ramp periods for increasing and decreasing weights
            ramp_up = 10
            ramp_down = 10

            weights = np.zeros(lookback)
            for j in range(lookback):
                if j < lag:
                    weights[j] = 0
                elif j < ramp_up + lag:
                    weights[j] = j / ramp_up
                elif j >= lookback - ramp_down:
                    weights[j] = (lookback - j) / ramp_down
                else:
                    weights[j] = 1
        elif weighting_scheme == 'exp':
            # an example of an exponential decay weighting schema
            # preferred route for up-weighting recent returns, also resulting in lower turnover
            half_life = lookback / 2
            ramp_up = 10
            ramp_down = 10

            weights = np.exp(-np.log(2) * np.arange(lookback) / half_life)

            # now clean up the basic exp series
            max_weight = weights.max()
            min_weight = weights.min()

            weights[:lag] = 0

            for j in range(lookback):
                if j < lag:
                    weights[j] = 0
                elif j < ramp_up + lag:
                    weights[j] = weights[j - 1] + (max_weight / ramp_up)
                elif j >= lookback - ramp_down:
                    weights[j] = weights[j - 1] - (min_weight / ramp_down)
                else:
                    continue
        else:
            # could provide custom array here...
            weights = np.ones(lookback) / lookback  # fallback

        # make sure all weights are > 0 and sum to 1
        weights = weights - weights.min()
        weights = weights / weights.sum()

        # ensure we zero out the lag on front end, regardless of above logic...
        if lag > 0:
            weights[:lag] = 0

        # weighted sum for each ticker
        momentum_df.loc[date] = (window.values * weights[:, None]).sum(axis=0)

    return momentum_df.astype(float), weights


''' ===============================================================================================================
        4. Normalize factor loadings by day (z-score each date)
=============================================================================================================== '''


def normalize_factor_loadings_by_day(momentum_df):
    """
    mean = 0, std = 1 for each day across all tickers.
    """
    daily_mean = momentum_df.mean(axis=1)
    daily_std = momentum_df.std(axis=1)

    # handle 0-value std dev
    daily_std[daily_std == 0] = np.nan

    # subtract daily mean, divide by daily std
    normed = (momentum_df.T - daily_mean).T
    normed = (normed.T / daily_std).T
    return normed


''' ===============================================================================================================
        5. Group stocks by quantile, compute factor returns
=============================================================================================================== '''


def calc_quartile_returns(returns_df, factor_loadings_df, quantiles=10):
    """
    for each day, sort stocks by factor loadings into quantiles,
    then compute the daily return of each quantile.
    """

    # align indices
    common_idx = returns_df.index.intersection(factor_loadings_df.index)
    returns_df = returns_df.loc[common_idx]
    factor_loadings_df = factor_loadings_df.loc[common_idx]

    quantile_returns = []
    for date in common_idx:
        loadings = factor_loadings_df.loc[date]
        rets = returns_df.loc[date]

        # drop tickers with NaN value
        daily_data = pd.concat([loadings, rets], axis=1).dropna()
        daily_data.columns = ['loadings', 'returns']

        if daily_data.empty:
            quantile_returns.append((date, np.nan, np.nan))
            continue

        loadings_sorted = daily_data.sort_values(by='loadings', ascending=True)

        bin_size = int(len(loadings_sorted) / quantiles)
        bottom_group = loadings_sorted.iloc[0:bin_size]
        top_group = loadings_sorted.iloc[-bin_size:]

        # just taking the average here, but should definitely do a different weighted average scheme like sqrt(mcap)
        bottom_ret = bottom_group['returns'].mean()
        top_ret = top_group['returns'].mean()

        quantile_returns.append((date, top_ret, bottom_ret))

    # df with top and bottom quantile returns
    quantile_returns_df = pd.DataFrame(quantile_returns, columns=['Date', 'TopQuantileRet', 'BottomQuantileRet'])
    quantile_returns_df.set_index('Date', inplace=True)
    return quantile_returns_df


def calc_momentum_factor_returns(quantile_returns_df):
    """
    Long top quantile, short bottom quantile each day.
    """
    quantile_returns_df['FactorRet'] = quantile_returns_df['TopQuantileRet'] - quantile_returns_df['BottomQuantileRet']
    return quantile_returns_df


''' ===============================================================================================================
        6. Summarize factor returns
=============================================================================================================== '''


def summarize_factor_returns(factor_rets_df, freq=252):
    """
    produce some summary stats on the factor returns
    """
    factor_ret = factor_rets_df['FactorRet'].dropna()

    # ann. return
    avg_daily = factor_ret.mean()
    ann_return = (1 + avg_daily) ** freq - 1

    # ann. vol
    ann_vol = factor_ret.std() * np.sqrt(freq)

    # Sharpe
    sharpe = 0 if ann_vol == 0 else ann_return / ann_vol

    # max drawdown -- just summing the returns here
    cum = factor_ret.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd_percent = drawdown.min()

    # max drawdown in vol units
    # -- this is not the best way to do this, should calc trailing vol on each day for denominator
    # -- could be different date than max percent dd
    dd_in_vol_units = max_dd_percent / (factor_ret.std() * np.sqrt(freq))

    # worst 2-week performance
    rolling_2w = factor_ret.rolling(10).sum()
    worst_2w = rolling_2w.min()

    summary = {
        'Annualized Return (%)': round(ann_return * 100, 2),
        'Annualized Vol': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 2),
        'Max Drawdown (%)': round(max_dd_percent * 100, 2),
        'Max DD (Ann. Vol Units)': round(dd_in_vol_units, 1),
        'Worst 2-week Perf (%)': round(worst_2w * 100, 2),
        'Date of Worst 2-week Perf': rolling_2w.idxmin().strftime('%Y-%m-%d')
    }
    return summary


def plot_factor_cumulative_returns(
    factor_rets_df,
    lookback=252,
    weighting_scheme='exp',
    etsu='sp500',
    quantiles=10
):
    """
    plots cumulative return of the factor.
    """

    # determine which label to use
    if lookback > 125:
        momentum_label = "Medium-Term Momentum Factor"
    elif lookback < 65:
        momentum_label = "Short-Term Momentum Factor"
    else:
        momentum_label = "Quarterly Momentum Factor"

    # get cumulative returns of the factor
    df = factor_rets_df.copy()
    df['Cumulative'] = df['FactorRet'].fillna(0).cumsum()

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df['Cumulative'].plot(ax=ax, label=momentum_label)

    # titles, labels, legend
    title = momentum_label
    subtitle = f"ETSU='{etsu}', lookback={lookback}, weighting='{weighting_scheme}', quantiles={quantiles}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc='upper left')

    # show plot
    plt.show()

    # save plot
    file_name_title = momentum_label.replace(' ', '_').lower()
    file_name_subtitle = f"etsu_{etsu}_lookback_{lookback}_weighting_{weighting_scheme}_quantiles_{quantiles}"
    plt.savefig(f"{file_name_title}_{file_name_subtitle}.png")


''' ===============================================================================================================
        7. Correlation of Factor to any other return series
=============================================================================================================== '''


def calc_factor_correlation(factor_rets_df, other_rets_df, lookback=None):
    """
    calculate correlation of FactorRet with another series.
    if lookback is provided, use only that period.
    """
    merged = pd.concat([factor_rets_df['FactorRet'], other_rets_df], axis=1, join='inner').dropna()
    if lookback:
        merged = merged.iloc[:lookback]
    return merged.corr().iloc[0, 1]


''' ===============================================================================================================
        Example Usage
=============================================================================================================== '''


if __name__ == '__main__':

    # 0. Params
    etsu = 'sp500'
    lookback = 252
    weighting_scheme = 'exp'
    quantiles = 10

    # 1. Universe
    tickers = pull_estimation_universe('sp500')

    # 2. Pull prices and returns
    prices = pull_daily_prices_of_etsu(tickers, start='2016-01-01', try_cache=True)
    rets = calc_daily_returns_of_etsu(prices)

    # 3. Momentum scores
    mom_scores, weights = calc_momentum_score(rets, lookback=lookback, weighting_scheme=weighting_scheme)

    # 4. Normalize
    norm_mom_scores = normalize_factor_loadings_by_day(mom_scores)

    # 5. Factor returns
    quantile_ret = calc_quartile_returns(rets, norm_mom_scores, quantiles=quantiles)
    factor_rets = calc_momentum_factor_returns(quantile_ret)

    # 6. Summaries
    stats = summarize_factor_returns(factor_rets)
    plot_factor_cumulative_returns(factor_rets.dropna(), lookback=lookback, weighting_scheme=weighting_scheme, etsu=etsu, quantiles=quantiles)

    print(f"\nETSU='{etsu}', lookback={lookback}, weighting='{weighting_scheme}', quantiles={quantiles}")
    for k, v in stats.items():
        print(f'{k}: {v}')

    # 7. Correlation with another series (e.g., NVDA returns)
    corr_ticker = 'NVDA'
    other_prices = pull_daily_prices_of_etsu([corr_ticker], start='2020-01-01')
    other_returns = calc_daily_returns_of_etsu(other_prices)
    corr = calc_factor_correlation(factor_rets, other_returns[corr_ticker], lookback=252)
    print(f'\nTest Ticker Correlation with {corr_ticker} (last year): {round(corr, 2)}')
    print(f'Average momentum loading for {corr_ticker}: {round(norm_mom_scores[corr_ticker].mean(), 2)}')
