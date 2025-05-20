import streamlit as st
from portfolio_risk_model import gen_random_data, PortfolioRiskModel
from app_utils import (portfolio_overview, portfolio_holdings,
                       factor_risk, alpha_breakdown, style_df)

st.set_page_config(page_title="Portfolio Risk Monitor", layout="wide")

# --- data / model ---
df_port, df_factor_loadings, df_factor_covar, df_alpha_vols = gen_random_data(random_seed=832)
prm = PortfolioRiskModel(df_port, df_factor_loadings, df_factor_covar, df_alpha_vols)
prm.model_risk()

# --- header ---
st.title("Portfolio Risk Model")
st.write(f"**Portfolio Date:** {prm.portfolio_date:%Y-%m-%d}")
st.write(f"**Risk Model:** {prm.factor_model_id}")
st.markdown("---")

# --- layout ---
left, middle, right = st.columns([5, 3, 2])

st.subheader("Portfolio Overview")
st.dataframe(style_df(portfolio_overview(prm)))

with left:
    st.subheader("Portfolio Holdings")
    ph = portfolio_holdings(prm, df_factor_loadings)
    st.dataframe(style_df(ph))

with middle:
    st.subheader("Factor Risk")
    st.dataframe(style_df(factor_risk(prm, df_factor_loadings)))

with right:
    st.subheader("Alpha Breakdown")
    st.dataframe(style_df(alpha_breakdown(ph)))

st.markdown("---")
