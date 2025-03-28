import numpy as np
import pandas as pd
import cvxpy as cp


def optimize_portfolio(df_port, df_factor_loadings, df_alpha_vols, factor_covar_df,
                       momentum_factor='momentum',
                       factor_exposure_bound=0.15,
                       trade_penalty=0.01,
                       side_preserve: bool = True):
    """
    Very simple demonstration of:
      max  x^T (diag(alpha_vol^2)) x  -  trade_penalty * ||x - w0||_1
      s.t. sum(x) = NMV, sum(|x|) <= GMV, factor_exp_momentum <= bound, etc.
    """

    # --------------------- Prepare inputs ---------------------#
    # Current portfolio dollar weights w0
    df_port = df_port.copy()
    w0 = df_port['mv'].values  # shape: (n,)

    alpha_vols = df_alpha_vols.set_index('ticker')['alpha_vol'].reindex(df_port['ticker']).values
    alpha_var = alpha_vols ** 2  # shape: (n,)
    Sigma = factor_covar_df.values  # must be PSD

    # Factor loadings for momentum
    df_mom = df_factor_loadings[df_factor_loadings['factor'] == momentum_factor].copy()
    df_mom = df_mom.set_index('ticker')['loading'].reindex(df_port['ticker']).fillna(0.0)
    momentum_loadings = df_mom.values  # shape: (n,)

    # Basic stats
    nmv = w0.sum()  # net exposure
    gmv = np.abs(w0).sum()  # gross exposure
    mom_limit = factor_exposure_bound * gmv  # e.g. ±15% of GMV => ±0.15 * GMV

    n = len(w0)

    # --------------------- Define optimization vars ---------------------#
    x = cp.Variable(n, name="new_weights")  # new dollar weights

    # --------------------- Objective ---------------------#
    # minimize factor variance minus linear alpha term
    B = df_factor_loadings.pivot_table(index='ticker', columns='factor', values='loading').reindex(df_port['ticker']).fillna(0.0).values
    Sigma_asset = B @ Sigma @ B.T
    obj = cp.Minimize(0.5 * cp.quad_form(x, Sigma_asset) - alpha_var @ x)

    # --------------------- Constraints ---------------------#
    constraints = []

    # 1) limit on net exposure
    constraints.append(cp.abs(cp.sum(x)) / gmv <= 0.03)

    # 2) limit gross exposure
    constraints.append(cp.sum(cp.abs(x)) <= gmv)

    # 3) factor exposure constraint for momentum
    mom_exp = momentum_loadings @ x  # ~ sum_i (loading_i * x_i)
    constraints.append(mom_exp <= mom_limit)
    constraints.append(mom_exp >= -mom_limit)

    # 4) Same side constraints
    if side_preserve:
        for i in range(n):
            if w0[i] != 0:
                constraints.append(x[i] * w0[i] >= 0)

    # # 5) Position size limits
    # for i in range(n):
    #     if np.abs(w0[i]) > 0:
    #         constraints.append(cp.abs(x[i]) >= (gmv / (n * 10)))

    # --------------------- Solve ---------------------#
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True)

    # --------------------- Return solution ---------------------#
    df_port['mv_opt'] = x.value
    df_port['trade'] = df_port['mv_opt'] - df_port['mv']
    return df_port


import numpy as np
import pandas as pd
import cvxpy as cp


def optimize_portfolio(df_port, df_factor_loadings, df_factor_covar, df_alpha_vols,
                       factor_limit=0.15, lambda_trade=1.0):
    """
    Example convex optimization using cvxpy to:
      minimize [ w^T B Σ B^T w  -  alpha^T w  +  λ * ||w - w_old||_1 ]
      subject to net-zero, factor exposures, no flipping sides, etc.
    """
    # ----- Prep Data -----
    # We'll treat current weights as w_old (scaled by gross notional for convenience)
    gmv = df_port['mv'].abs().sum()
    w_old = df_port['mv'] / gmv  # old weights
    tickers = df_port['ticker'].values

    # For simplicity, define alpha as some linear alpha for each stock (e.g. from alpha_vol or separate alpha signal)
    # Here we just use alpha_vol * 1.0 for demonstration
    alpha_map = df_alpha_vols.set_index('ticker')['alpha_vol']
    alpha_vec = np.array([alpha_map.get(t, 0.0) for t in tickers])

    # Build factor exposure matrix B (N x K)
    # pivot factor_loadings -> row: ticker, cols: factor
    pivoted = df_factor_loadings.pivot(index='ticker', columns='factor', values='loading').fillna(0.0)
    B = pivoted.loc[tickers].values  # matches order of df_port

    # Factor covariance Σ (K x K) -> pre-compute Q = B Σ B^T (N x N)
    Sigma = df_factor_covar.values
    Q = B @ Sigma @ B.T

    # ----- Define CVX Variables & Objective -----
    N = len(tickers)
    w = cp.Variable(N)

    # factor risk: w^T Q w
    factor_risk = cp.quad_form(w, Q)

    # alpha reward: - alpha^T w  (we *minimize* negative alpha)
    alpha_term = - alpha_vec @ w

    # L1 trading cost: sum |w - w_old|
    trade_cost = cp.norm1(w - w_old)

    # final objective
    obj = cp.Minimize(factor_risk + alpha_term + lambda_trade * trade_cost)

    # ----- Constraints -----
    constraints = []

    # (a) Net exposure = 0  (dollar neutral or sum of weights = 0)
    constraints.append(cp.sum(w) == 0)

    # (b) No flipping sides: sign(w_i) == sign(w_old_i) unless w_old_i == 0
    #    A simple approach: if w_old_i > 0, force w_i >= 0; if w_old_i < 0, force w_i <= 0.
    for i in range(N):
        if w_old[i] > 0:
            constraints.append(w[i] >= 0)
        elif w_old[i] < 0:
            constraints.append(w[i] <= 0)
        # if w_old[i] == 0, we might allow either sign or do something else

    # (c) Factor exposure constraints, e.g. |-exposure to momentum| <= factor_limit
    #    Suppose we limit each factor’s exposure (in weight space) to +/- factor_limit
    #    B^T w is a K-dimensional vector of factor exposures
    factor_exposures = B.T @ w
    for k in range(factor_exposures.shape[0]):
        constraints.append(factor_exposures[k] <= factor_limit)
        constraints.append(factor_exposures[k] >= -factor_limit)

    # ----- Solve -----
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)  # or another QP-capable solver

    # ----- Gather Results -----
    w_opt = w.value
    trade_recs = w_opt - w_old.values

    # Package into a DataFrame
    df_opt = pd.DataFrame({
        'ticker': tickers,
        'w_old': w_old.values,
        'w_opt': w_opt,
        'trade': trade_recs
    })
    return df_opt, prob.value


if __name__ == '__main__':
    # 1) Generate random data
    from portfolio_risk_model import gen_random_data, PortfolioRiskModel

    df_p, df_fl, df_fc, df_av = gen_random_data(num_names=30, random_seed=832)  # smaller for demo

    # 2) Original portfolio risk
    prm_before = PortfolioRiskModel(df_p, df_fl, df_fc, df_av)
    prm_before.model_risk()

    print("=== Original Portfolio ===")
    print(f"Names: {prm_before.num_names}")
    print(f"GMV: {prm_before.port_gmv:.1f}, NMV: {prm_before.port_nmv:.1f}")
    print(f"Idio Risk Contribution: {prm_before.port_risk_contribution_alpha * 100:.2f}%")
    print(f"ENP: {prm_before.port_enp:.2f}")
    print("\nBeginning Factor Exposures (% of GMV):")
    print((prm_before.factor_exp_pctgmv * 100).round(2))

    # 3) Run the optimization
    df_opt, obj_val = optimize_portfolio(df_p, df_fl, df_fc, df_av,
                                         factor_limit=0.15, lambda_trade=1.0)

    # 4) Measure risk after
    df_new_port = df_opt[['ticker', 'w_opt']].rename(columns={'w_opt': 'mv'})
    df_new_port['date'] = df_p['date']
    prm_after = PortfolioRiskModel(df_new_port, df_fl, df_fc, df_av)
    prm_after.model_risk()

    print("\n=== Optimized Portfolio ===")
    print(f"GMV: {prm_after.port_gmv:.1f}, NMV: {prm_after.port_nmv:.1f}")
    print(f"Idio Risk Contribution: {prm_after.port_risk_contribution_alpha * 100:.2f}%")
    print(f"ENP: {prm_after.port_enp:.2f}")

    # 5) Print factor exposures, trades, etc.
    print("\nFinal Factor Exposures (% of GMV):")
    print((prm_after.factor_exp_pctgmv * 100).round(2))

    print("\nTop Trades:")
    df_opt['abs_trade'] = df_opt['trade'].abs()
    print(df_opt[['ticker', 'w_old', 'w_opt', 'trade', 'abs_trade']].nlargest(10, 'abs_trade'))
