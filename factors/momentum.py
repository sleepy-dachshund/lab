

'''

    Author: Bruce McNair

    Instructions:
        - You can just run this script in isolation as long as you have an internet connection.
        - If you want, you can adjust parameters (lookback period, weighting scheme, etc.) in the main block at the bottom.
        - The script will cache a .csv file of prices and a .png image of the momentum factor returns in the DATA_DIRECTORY provided.
        - Only very basic packages required: pandas, numpy, yfinance, matplotlib.

    Caveat:
        - This is not a complete momentum factor model, just an example.
        - Assume any choices made are just there to illustrate that the choice exists and therefore must be studied.

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

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

DATA_DIRECTORY = 'data_dump/'


''' ===============================================================================================================
        1. Pull estimation universe
=============================================================================================================== '''


def pull_estimation_universe(index: str = 'sp500'):
    """
    scrape S&P 500 tickers from wikipedia by default, or use a csv file.
    """
    # todo: need a better estimation universe
    if index.lower() == 'sp500':
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    else:
        # read a custom .csv file from data directory, just needs a 'Ticker' column
        return pd.read_csv(f'{DATA_DIRECTORY}{index}.csv')['Ticker'].tolist()


''' ===============================================================================================================
        2. Pull daily prices and calculate daily returns
        
        -- not ethis is messy due to yfinance API limits...
=============================================================================================================== '''


def load_price_cache(cache_filename=DATA_DIRECTORY+'price_cache.csv'):
    """load cached price data if available."""
    if os.path.exists(cache_filename):
        try:
            return pd.read_csv(cache_filename, index_col=0)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return pd.DataFrame()


def save_price_cache(data, cache_filename=DATA_DIRECTORY+'price_cache.csv'):
    """save the price data to a cache CSV."""
    data.to_csv(cache_filename)


def download_batch(batch, start, end, delay, max_retries):
    """download a batch of tickers with retry logic and validate the data."""
    retries = 0
    while retries < max_retries:
        try:
            batch_data = yf.download(batch, start=start, end=end, auto_adjust=False)['Adj Close']
            if isinstance(batch_data, pd.Series):
                batch_data = batch_data.to_frame()

            # validate that at least 75% of columns are not entirely null...
            valid_cols = batch_data.columns[~batch_data.isnull().all()]
            if len(valid_cols) < 0.75 * batch_data.shape[1]:
                raise ValueError(f"Only {len(valid_cols)}/{batch_data.shape[1]} columns are valid.")

            return batch_data
        except Exception as e:
            retries += 1
            print(f"Error downloading batch {batch}: Retry {retries}/{max_retries} - {e}")
            time.sleep(delay * retries)  # exponential backoff
    return None


def pull_fresh_prices(tickers, start, end, batch_size=100, delay=10, max_retries=3,
                      cache_filename=DATA_DIRECTORY+'price_cache.csv', output_file_name='prices.csv'):
    """
    Pull fresh price data for the given tickers.
    - Only downloads tickers that aren't already cached.
    - Updates the cache CSV as each batch is successfully downloaded.
    """
    # load existing cache.
    cache_df = load_price_cache(cache_filename)
    cached_tickers = set(cache_df.columns) if not cache_df.empty else set()

    # filter out tickers already in the cache.
    tickers_to_download = [t for t in tickers if t not in cached_tickers]
    print(
        f"Total tickers: {len(tickers)}. Already cached: {len(cached_tickers)}. To download: {len(tickers_to_download)}")

    # download missing tickers in batches.
    for i in range(0, len(tickers_to_download), batch_size):
        batch = tickers_to_download[i:i + batch_size]
        batch_data = download_batch(batch, start, end, delay, max_retries)
        if batch_data is not None:
            # update cache by merging new data with existing data.
            if cache_df.empty:
                cache_df = batch_data.copy()
            else:
                # align date cols of new and existing data
                cache_df.index = pd.to_datetime(cache_df.index)
                batch_data.index = pd.to_datetime(batch_data.index)
                cache_df = pd.concat([cache_df, batch_data], axis=1)
                cache_df = cache_df.loc[:, ~cache_df.columns.duplicated()]
            save_price_cache(cache_df, cache_filename)
        else:
            print(f"Batch {batch} failed after {max_retries} retries. Skipping.")
        time.sleep(delay)  # Delay between batches

    # save the full data to another file.
    cache_df = cache_df.loc[:, ~cache_df.columns.duplicated()]
    cache_df.to_csv(output_file_name)
    return cache_df


def pull_daily_prices_of_estu(tickers, start='2020-01-01', end=None,
                              cache_dir: str = DATA_DIRECTORY,
                              cache_path: str = 'prices_sp500.csv',
                              try_cache=True):
    """
    pull daily adj. close prices from yfinance. check local cache first by default.
    """
    cache_csv = cache_dir + cache_path

    if not tickers:
        return pd.DataFrame()

    if try_cache:
        try:
            data = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
            data = data[tickers].dropna(how='all', axis=0)
            if end:
                data = data.loc[start:end]
            else:
                data = data.loc[start:]
            return data
        except FileNotFoundError:
            data = pull_fresh_prices(tickers, start, end, output_file_name=cache_csv)
            data.dropna(how='all', axis=0, inplace=True)
            data.to_csv(cache_csv)
            return data
    else:
        data = pull_fresh_prices(tickers, start, end, output_file_name=cache_csv)
        data.dropna(how='all', axis=0, inplace=True)
        data.to_csv(cache_csv)
        return data


def calc_daily_returns_of_estu(price_df):
    """
    simple daily pct_change returns
    """
    return price_df.pct_change().fillna(0)


''' ===============================================================================================================
        3. Calculate momentum score (weighted sum of past returns)
=============================================================================================================== '''


def calc_momentum_score(returns_df,
                        lookback=252,
                        weighting_scheme='exp',
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
            ramp_up = int(lookback * 10 / 252)
            ramp_down = int(lookback * 10 / 252)

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
            ramp_up = int(lookback * 10 / 252)
            ramp_down = int(lookback * 10 / 252)

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


def calc_quantile_returns(returns_df, factor_loadings_df, quantiles=10):
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

        # todo: just taking the average here, but should definitely do a different weighted average scheme like sqrt(mcap)
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
        6. Calculate the Turnover of the Factor
=============================================================================================================== '''


def calc_quantile_name_turnover(factor_loadings_df, quantiles=10):
    """
    returns a df of day-to-day overlap and turnover
    for top and bottom quantiles.

    note: this calcs daily turnover of names in the top/bottom quantile
    -- not in % GMV, and not of the loadings themselves, which is what you'd actually manage to

    # todo: should slow this down from daily to monthly, include a buffer zone
    """
    dates = factor_loadings_df.index
    top_overlap = []
    bottom_overlap = []

    # calc daily top/bottom sets
    daily_top = []
    daily_bottom = []
    for dt in dates:
        data = factor_loadings_df.loc[dt].dropna()
        if data.empty:
            daily_top.append(set())
            daily_bottom.append(set())
            continue
        data_sorted = data.sort_values()
        bin_size = int(len(data_sorted) / quantiles)
        bottom_set = set(data_sorted.index[:bin_size])
        top_set = set(data_sorted.index[-bin_size:])

        # append daily top/bottom sets to list
        daily_bottom.append(bottom_set)
        daily_top.append(top_set)

    # calc overlaps
    for i in range(1, len(dates)):
        prev_top = daily_top[i-1]
        curr_top = daily_top[i]
        prev_bottom = daily_bottom[i-1]
        curr_bottom = daily_bottom[i]

        # fraction that remain in top
        top_in_both = len(prev_top.intersection(curr_top))
        top_overlap.append(top_in_both / (len(prev_top) if len(prev_top) > 0 else np.nan))

        # fraction that remain in bottom
        bottom_in_both = len(prev_bottom.intersection(curr_bottom))
        bottom_overlap.append(bottom_in_both / (len(prev_bottom) if len(prev_bottom) > 0 else np.nan))

    turnover_df = pd.DataFrame({
        'TopOverlap': [np.nan] + top_overlap,
        'TopTurnover': [np.nan] + [1 - x for x in top_overlap],
        'BottomOverlap': [np.nan] + bottom_overlap,
        'BottomTurnover': [np.nan] + [1 - x for x in bottom_overlap]
    }, index=dates)
    return turnover_df


''' ===============================================================================================================
        7. Summarize factor returns
=============================================================================================================== '''


def summarize_factor_returns(factor_rets_df, turnover, freq=252, side=None):
    """
    produce some summary stats on the factor returns
    """
    if side is None:
        factor_ret = factor_rets_df['FactorRet'].dropna().copy()
        ann_turnover = turnover[['TopTurnover', 'BottomTurnover']].mean().mean() * 252
    elif side == 'long':
        factor_ret = factor_rets_df['TopQuantileRet'].dropna().copy()
        ann_turnover = turnover['TopTurnover'].mean() * 252
    elif side == 'short':
        factor_ret = factor_rets_df['BottomQuantileRet'].dropna().copy()
        ann_turnover = turnover['BottomTurnover'].mean() * 252
    else:
        raise ValueError('Invalid side parameter')

    # ann. return
    avg_daily = factor_ret.mean()
    ann_return = avg_daily * freq

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
    # -- could be also be different date than max percent dd
    dd_in_vol_units = max_dd_percent / (factor_ret.std() * np.sqrt(freq))

    # worst 2-week performance
    rolling_2w = factor_ret.rolling(10).sum()
    worst_2w = rolling_2w.min()

    # select z-score
    z_dates = ['2020-11-09', '2022-11-10']
    z_scores = ((factor_ret - factor_ret.mean()) / factor_ret.std())[z_dates]

    summary = {
        'Ann. Return (%)': round(ann_return * 100, 2),
        'Ann. Vol (%)': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 2),
        'Ann. Return (Long, %)': round(factor_rets_df['TopQuantileRet'].mean() * 100 * freq, 2),
        'Ann. Returns (Short, %)': round(factor_rets_df['BottomQuantileRet'].mean() * 100 * freq, 2),
        'Ann. Turnover (% Names)': round(ann_turnover * 100, 2),
        'Ann. Turnover (Loadings)': '',  # todo: this would be more valuable -- targets used to hedge
        'Max Drawdown (%)': round(max_dd_percent * 100, 2),
        'Max DD (Ann. Vol Units)': round(dd_in_vol_units, 1),
        'Worst 2-week Perf (%)': round(worst_2w * 100, 2),
        'Date of Worst 2-week Perf': rolling_2w.idxmin().strftime('%Y-%m-%d'),
        f'Z Score on Vaccine News {z_dates[0]}': round(z_scores[z_dates[0]], 2),
        f'Z Score on Cool Inflation {z_dates[1]}': round(z_scores[z_dates[1]], 2)
    }
    return summary


def plot_factor_cumulative_returns(
    factor_rets_df: pd.DataFrame,
    factor_stretch: pd.Series,
    quantile_turnover_names: pd.DataFrame,
    lookback: int,
    weighting_scheme: str,
    estu: str,
    quantiles: int,
    cache_dir: str = DATA_DIRECTORY,
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

    # get daily turnover
    turnover = quantile_turnover_names[['TopTurnover', 'BottomTurnover']].mean(axis=1).loc[df.index]

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(12, 6),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})

    # df['Cumulative'].plot(ax=ax, label=momentum_label)
    ax1.plot(df.index, df['Cumulative'], label=momentum_label)
    ax2.plot(turnover.index, turnover, label='Name Turnover')
    ax3.plot(df.index, factor_stretch, label='Factor Stretch')

    # titles, labels, legend
    title = momentum_label
    subtitle = f"ESTU='{estu}', lookback={lookback}, weighting='{weighting_scheme}', quantiles={quantiles}"

    ax1.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax1.set_ylabel("Cumulative Factor Return")
    ax1.grid()
    ax1.legend(loc='upper left')

    ax2.set_ylabel("Turnover")
    ax2.axhline(turnover.mean(), color='black', linestyle='--', label=f'Avg. = {round(turnover.mean(), 3)}')
    ax2.legend(loc='upper left')

    ax3.set_ylabel("Factor Stretch")
    ax3.set_xlabel("Date")
    ax3.axhline(factor_stretch.mean(), color='black', linestyle='--', label=f'Avg. = {round(factor_stretch.mean(), 4)}')
    ax3.legend(loc='upper left')

    # save plot
    file_name_title = momentum_label.replace(' ', '_').lower()
    file_name_subtitle = f"estu_{estu}_lookback_{lookback}_weighting_{weighting_scheme}_quantiles_{quantiles}"
    plt.savefig(f"{cache_dir}{file_name_title}_{file_name_subtitle}.png")

    # show plot
    plt.show()


''' ===============================================================================================================
        8. Correlation of Factor to any other return series
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
        9. Grid Search
=============================================================================================================== '''


def run_momentum_grid_search(
        returns_df,
        horizons=[5, 10, 21, 63, 126, 252],
        weighting_schemes=['exp'], # ['uniform', 'trapezoidal', 'exp']
        quantiles_list=[4, 6, 10],
        lag=None
):
    """
    grid search over lookback horizons, weighting schemes, and quantiles.
    for each combination:
      1) calc momentum scores
      2) normalize them
      3) calc factor returns
      4) summarizeresults
    return a df of results sorted by Sharpe descending.
    """
    results = []

    # run a loop through params
    for lookback in horizons:
        for weighting_scheme in weighting_schemes:
            for q in quantiles_list:

                # 1. calc momentum scores

                # wonky dynamic lag
                if lookback < 30:
                    lag = 0
                elif lookback < 70:
                    lag = 10
                else:
                    lag = 21


                mom_scores, _ = calc_momentum_score(
                    returns_df,
                    lookback=lookback,
                    weighting_scheme=weighting_scheme,
                    lag=lag
                )

                # 2. norm
                norm_mom_scores = normalize_factor_loadings_by_day(mom_scores)

                # 3. turnover
                quantile_turnover_names = calc_quantile_name_turnover(norm_mom_scores, quantiles=q)

                # 4. factor rets
                quantile_ret = calc_quantile_returns(returns_df, norm_mom_scores, quantiles=q)
                factor_rets = calc_momentum_factor_returns(quantile_ret)

                # 5. summary
                stats = summarize_factor_returns(factor_rets, quantile_turnover_names)

                # organize results
                row = {
                    'Lookback': lookback,
                    'Weighting': weighting_scheme,
                    'Quantiles': q,
                    'Lag': lag,
                }
                row.update(stats)
                results.append(row)

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='Sharpe', ascending=False).reset_index(drop=True)


''' ===============================================================================================================
        Example Usage
=============================================================================================================== '''


if __name__ == '__main__':

    '''
        Factor
    '''

    # 0. Params
    estu = 'IWV'  # 'sp500' or 'IWV'
    lookback = 252
    weighting_scheme = 'exp'
    quantiles = 4

    # 1. Universe
    tickers = pull_estimation_universe(estu)

    # 2. Pull prices and returns
    prices = pull_daily_prices_of_estu(tickers, start='2016-01-01', try_cache=True, cache_path=f'prices_{estu}.csv')
    rets = calc_daily_returns_of_estu(prices)

    # 3. Momentum scores
    mom_scores, weights = calc_momentum_score(rets, lookback=lookback, weighting_scheme=weighting_scheme)
    factor_stretch = mom_scores.quantile(0.9, axis=1) - mom_scores.quantile(0.1, axis=1)

    # 4. Normalize
    norm_mom_scores = normalize_factor_loadings_by_day(mom_scores)

    # 5. Factor returns
    quantile_ret = calc_quantile_returns(rets, norm_mom_scores, quantiles=quantiles)
    factor_rets = calc_momentum_factor_returns(quantile_ret)

    # 6. Turnover
    quantile_turnover_names = calc_quantile_name_turnover(norm_mom_scores, quantiles=quantiles)

    # 7. Summaries
    stats = summarize_factor_returns(factor_rets, quantile_turnover_names)
    stats_long = summarize_factor_returns(factor_rets, quantile_turnover_names, side='long')
    stats_short = summarize_factor_returns(factor_rets, quantile_turnover_names, side='short')
    plot_factor_cumulative_returns(factor_rets.dropna(), factor_stretch.dropna(), quantile_turnover_names,
                                   lookback=lookback, weighting_scheme=weighting_scheme, estu=estu, quantiles=quantiles)

    print(f"\nESTU='{estu}', lookback={lookback}, weighting='{weighting_scheme}', quantiles={quantiles}")
    for k, v in stats.items():
        print(f'{k}: {v}')

    # 8. Correlation with another series (e.g., NVDA returns)
    corr_ticker = 'NVDA'
    other_prices = pull_daily_prices_of_estu([corr_ticker], start='2020-01-01')
    other_returns = calc_daily_returns_of_estu(other_prices)
    corr = calc_factor_correlation(factor_rets, other_returns[corr_ticker], lookback=252)
    print(f'\nTest Ticker Correlation with {corr_ticker} (last year): {round(corr, 2)}')
    print(f'Average momentum loading for {corr_ticker}: {round(norm_mom_scores[corr_ticker].mean(), 2)}')

    # 9. Momentum Exposure of a Single Stock Over Time

    # 10. Momentum Exposure of a L/S Book Over Time

    '''
        Factory
    '''
    # run across various lookbacks, weighting schemes, quantiles -- compare stats of various factors
    results_df = run_momentum_grid_search(rets)
    results_df.to_excel(f'{DATA_DIRECTORY}momentum_grid_search_results_{estu}.xlsx')
