import pandas as pd

COLS_HOLD = ['industry', 'side', 'gmv', 'mv', 'weight',
             'alpha_risk_contribution', 'position_alpha_vol', 'stock_alpha_vol']
COLS_FACTOR = ['factor_group', 'side', 'factor_vol_bps', 'risk_cont_pct']

def style_df(df):
    """Return a Styler with sensible numeric formatting."""
    COLS_PERCENTAGE = ['Breadth', 'Total Vol (% GMV)',
                       'Alpha Risk Contribution', 'Alpha Risk (Long)',
                       'Alpha Risk (Short)', 'Factor Risk Contribution',
                       'weight', 'alpha_risk_contribution',
                       'stock_alpha_vol', 'risk_cont_pct', 'pct_gmv']

    COLS_INT = ['Names']

    COLS_ROUND_ONE = ['ENP', 'GMV ($mm)', 'NMV ($mm)',
                      'Total Vol ($mm)', 'gmv', 'nmv',
                      'position_alpha_vol', 'factor_vol_bps']

    fmt = {}

    # integers
    fmt.update({c: '{:,.0f}'.format for c in COLS_INT if c in df.columns})

    # 1-decimal floats
    fmt.update({c: '{:,.1f}'.format for c in COLS_ROUND_ONE if c in df.columns})

    # percentages (assumed 0-1)
    fmt.update({c: '{:.1%}'.format for c in COLS_PERCENTAGE if c in df.columns})

    return df.style.format(fmt)


# ---------- builders ----------
def portfolio_overview(prm) -> pd.DataFrame:
    return pd.DataFrame([{
        'Names':                     prm.num_names,
        'ENP':                       prm.port_enp,
        'Breadth':                   prm.port_enp / prm.num_names,
        'GMV ($mm)':                 prm.port_gmv / 1e6,
        'NMV ($mm)':                 prm.port_nmv / 1e6,
        'Total Vol ($mm)':           prm.port_vol_total_dollar / 1e6,
        'Total Vol (% GMV)':         prm.port_vol_total_pct,
        'Alpha Risk Contribution':   prm.port_risk_contribution_alpha,
        'Alpha Risk (Long)':         prm.port_alpha_risk_contribution_long,
        'Alpha Risk (Short)':        prm.port_alpha_risk_contribution_short,
        'Factor Risk Contribution':  prm.port_risk_contribution_factor
    }])

def portfolio_holdings(prm, df_factor_loadings) -> pd.DataFrame:
    df_ind = (df_factor_loadings
              .loc[(df_factor_loadings.factor_group == 'industry') &
                   (df_factor_loadings.loading == 1),
                   ['ticker', 'factor']]
              .drop_duplicates()
              .rename(columns={'factor': 'industry'}))
    return (prm.port_risk_model_df
            .merge(df_ind, on='ticker', how='left', validate='1:1')
            .set_index('ticker')[COLS_HOLD])

def factor_risk(prm, df_factor_loadings) -> pd.DataFrame:
    base = prm.port_factor_risk_contribution_df.copy()
    base = base.merge(df_factor_loadings[['factor', 'factor_group']].drop_duplicates(),
                      left_index=True, right_on='factor', how='left', validate='1:1')
    return (base.set_index('factor')[COLS_FACTOR]
                .sort_values(['factor_group', 'risk_cont_pct'],
                             ascending=[False, False]))

def alpha_breakdown(port_hold) -> pd.DataFrame:
    port_hold['pct_gmv'] = port_hold['gmv'] / port_hold['gmv'].sum()
    side = (port_hold.groupby('side')[['pct_gmv', 'alpha_risk_contribution']]
                    .sum().rename_axis('subset').reset_index())
    side['group'] = 'Side'

    ind = (port_hold.groupby('industry')[['pct_gmv', 'alpha_risk_contribution']]
                   .sum().rename_axis('subset').reset_index())
    ind['group'] = 'Industry'

    out = (pd.concat([side, ind], ignore_index=True)
             .set_index(['group', 'subset'])
             .sort_values(['group', 'alpha_risk_contribution', 'pct_gmv'],
                          ascending=[False, False, False]))
    return out
