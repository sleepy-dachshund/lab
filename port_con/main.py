import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from portfolio_risk_model import gen_random_data, PortfolioRiskModel

# # Set page configuration
st.set_page_config(page_title="Portfolio Risk Monitor", layout="wide")

# generate random example data
df_port, df_factor_loadings, df_factor_covar, df_alpha_vols = gen_random_data(random_seed=832)

# initialize portfolio risk model
prm = PortfolioRiskModel(df_port, df_factor_loadings, df_factor_covar, df_alpha_vols)
prm.model_risk()

st.write(f"# Portfolio Risk Model")
st.write("---")
st.write(f"### Portfolio Date: {prm.portfolio_date.strftime('%Y-%m-%d')}")
st.write(F"### Risk Model: {prm.factor_model_id}")
st.write("---")

left, middle, right1, right2 = st.columns([2, 5, 3, 2])

with left:
    st.subheader('Portfolio Overview')
    port_ovr = pd.DataFrame(
        {   'Names': [prm.num_names],
            'ENP': [prm.port_enp],
            'Breadth': [prm.port_enp / prm.num_names],
            'GMV ($mm)': [prm.port_gmv / 1e6],
            'NMV ($mm)': [prm.port_nmv / 1e6],
            'Total Vol ($mm)': [prm.port_vol_total_dollar / 1e6],
            'Total Vol (% GMV)': [prm.port_vol_total_pct * 100],
            'Alpha Risk Contribution': [prm.port_risk_contribution_alpha * 100],
            'Alpha Risk (Long)': [prm.port_alpha_risk_contribution_long * 100],
            'Alpha Risk (Short)': [prm.port_alpha_risk_contribution_short * 100],
            'Factor Risk Contribution': [prm.port_risk_contribution_factor * 100]}
    ).T
    port_ovr.columns = ['Value']
    st.dataframe(port_ovr, use_container_width=True, height=750)

with middle:
    st.subheader('Portfolio Holdings')
    cols = ['industry', 'side', 'gmv', 'mv', 'weight', 'alpha_risk_contribution', 'position_alpha_vol', 'stock_alpha_vol']
    port_hold = prm.port_risk_model_df.copy()
    df_industry = df_factor_loadings.loc[(df_factor_loadings['factor_group'] == 'industry') & (df_factor_loadings['loading'] == 1), ['ticker', 'factor']].drop_duplicates().rename(columns={'factor': 'industry'})
    port_hold = port_hold.merge(df_industry, left_on='ticker', right_on='ticker', how='left', validate='1:1').set_index('ticker')[cols]
    st.dataframe(port_hold, use_container_width=True, height=750)

with right1:
    st.subheader('Factor Risk')
    cols = ['side', 'factor_vol_bps', 'risk_cont_pct']
    port_fact = prm.port_factor_risk_contribution_df.copy()
    port_fact = port_fact.merge(df_factor_loadings[['factor', 'factor_group']].drop_duplicates(), left_index=True, right_on='factor', how='left', validate='1:1')
    cols = ['factor_group'] + cols
    port_fact = port_fact.set_index('factor')[cols].sort_values(['factor_group', 'risk_cont_pct'], ascending=[False, False])
    st.dataframe(port_fact, use_container_width=True, height=750)

with right2:
    st.subheader('Alpha Breakdown')
    alpha_side = port_hold.groupby('side')['alpha_risk_contribution'].sum().reset_index().rename(columns={'side': 'subset'})
    alpha_side['group'] = 'Side'
    alpha_industry = port_hold.groupby('industry')['alpha_risk_contribution'].sum().reset_index().rename(columns={'industry': 'subset'})
    alpha_industry['group'] = 'Industry'

    # concat the above dfs vertically
    alpha_combined = pd.concat([alpha_side, alpha_industry], ignore_index=True)
    cols = ['group', 'subset', 'alpha_risk_contribution']
    alpha_combined = alpha_combined[cols].sort_values(['group', 'alpha_risk_contribution'], ascending=[False, False]).set_index(['group', 'subset'])
    st.dataframe(alpha_combined, use_container_width=True, height=750)

st.write("---")