import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Define helper functions
def portfolio_variance(w, sigma):
    return w.T @ sigma @ w

def neg_sharpe_ratio(w, mu, sigma, rf_annual):
    return -(w.T @ mu - rf_annual) / np.sqrt(w.T @ sigma @ w)

def compute_var_es(returns, alpha=0.95):
    cutoff = np.percentile(returns, 100 * (1 - alpha))
    es = returns[returns <= cutoff].mean()
    return cutoff, es


# Title
st.set_page_config(layout="wide")
col_i, col_t, col_z = st.columns([3.5, 0.5, 1.5])
with col_i:
    st.header('Portfolio optimization tool')
with col_t:
    st.image("report.jpg", width=80)
with col_z:
    st.markdown("""Created by 
    <a href="https://www.linkedin.com/company/starting-finance-club-bocconi/posts/?feedView=all" target="_blank">
        <button style="background-color: #262730; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">
            Starting Finance Club Bocconi
        </button>
    </a>
    """, unsafe_allow_html=True)

# Sidebar: User Inputs
st.sidebar.header("Portfolio Optimization Settings")
uploaded_file = st.sidebar.file_uploader("Upload Returns Data", type=["xlsx", "xls"])
frequency = st.sidebar.selectbox("Frequency", ["Daily", "Monthly"])
rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", min_value=0.0, value=2.00, max_value=100.0, step=0.01, format="%.2f") / 100
long_only = st.sidebar.selectbox("Long Only?", ["No", "Yes"])
vol_cap_enabled = st.sidebar.selectbox("Volatility Cap?", ["No", "Yes"])

# Use only for annualizing returns & vol; do NOT scale the risk-free rate
if frequency == "Daily":
    scale_factor = 252
elif frequency == "Monthly":
    scale_factor = 12

volatility_cap = None
if vol_cap_enabled == "Yes":
    volatility_cap = st.sidebar.number_input("Target Volatility (%)", min_value=0.1, max_value=100.0, value = 20.00, step=0.1, format="%.1f") / 100

if uploaded_file is not None:
    # Read asset returns data
    df = pd.read_excel(uploaded_file)
    asset_names = df.columns
    asset_returns = df.to_numpy()  # raw returns (daily or monthly)
    
    # Annualize returns and volatility for summary purposes
    means = asset_returns.mean(axis=0) * scale_factor
    std_devs = asset_returns.std(axis=0, ddof=1) * np.sqrt(scale_factor)
    mins = asset_returns.min(axis=0)
    maxs = asset_returns.max(axis=0)
    # Sharpe ratio in summary uses annual risk-free rate
    sharpe_ratios = (means - rf_annual) / std_devs

    # Compute 95% VaR and ES for each asset (based on raw returns)
    var_95 = []
    es_95 = []
    for i in range(len(asset_names)):
        v, es = compute_var_es(asset_returns[:, i], alpha=0.95)
        var_95.append(v)
        es_95.append(es)
    
    # Create the asset summary DataFrame
    summary_df = pd.DataFrame({
        "Annualized Return": means,
        "   Volatility    ": std_devs,
        " Min Daily Return": mins,
        " Max Daily Return": maxs,
        "   Sharpe Ratio  ": sharpe_ratios,
        "      95% VaR    ": var_95,
        "      95% ES     ": es_95
    }, index=asset_names)

    st.subheader("Asset Summary Statistics")
    st.dataframe(
        summary_df.style
        .set_properties(**{'background-color': 'black', 'color': 'white'})
        .format({
            "Annualized Return": "{:.2%}",
            "   Volatility    ": "{:.2%}",
            " Min Daily Return": "{:.2%}",
            " Max Daily Return": "{:.2%}",
            "      95% VaR    ": "{:.2%}",
            "      95% ES     ": "{:.2%}",
            "   Sharpe Ratio  ": "{:.2f}"
        })
    )

    # Annualized covariance matrix
    sigma_matrix = np.cov(asset_returns, rowvar=False) * scale_factor



    mu = means
    x0 = np.ones(len(asset_names)) / len(asset_names)

    # Constraints
    constraint1 = dict(type='eq', fun=lambda x: x.T @ np.ones(x.shape) - 1)
    constraint2 = dict(type='ineq', fun=lambda x: x)
    constraint3 = dict(type='eq', fun=lambda x: np.sqrt(portfolio_variance(x, sigma_matrix)) - volatility_cap)

    # Optimization for max Sharpe (Optimal/Tangency Portfolio)
    if long_only == "Yes" and volatility_cap is not None:
        opt_result = minimize(neg_sharpe_ratio, x0, args=(mu, sigma_matrix, rf_annual), constraints=[constraint1, constraint2, constraint3])
    elif long_only == "Yes":
        opt_result = minimize(neg_sharpe_ratio, x0, args=(mu, sigma_matrix, rf_annual), constraints=[constraint1, constraint2])
    elif volatility_cap is not None:
        opt_result = minimize(neg_sharpe_ratio, x0, args=(mu, sigma_matrix, rf_annual), constraints=[constraint1, constraint3])
    else:
        opt_result = minimize(neg_sharpe_ratio, x0, args=(mu, sigma_matrix, rf_annual), constraints=[constraint1])

    optimal_weights = opt_result.x
    portfolio_return = np.dot(optimal_weights, mu)  # annualized return
    portfolio_std = np.sqrt(portfolio_variance(optimal_weights, sigma_matrix))  # annualized volatility
    portfolio_sharpe = (portfolio_return - rf_annual) / portfolio_std

    # Optimization for Minimum Variance Portfolio
    if long_only == "Yes":
        min_var_result = minimize(portfolio_variance, x0, args=(sigma_matrix,), constraints=[constraint1, constraint2])
    else:
        min_var_result = minimize(portfolio_variance, x0, args=(sigma_matrix,), constraints=[constraint1])

    min_var_weights = min_var_result.x
    min_var_return = np.dot(min_var_weights, mu)
    min_var_std = np.sqrt(portfolio_variance(min_var_weights, sigma_matrix))
    min_var_sharpe = (min_var_return - rf_annual) / min_var_std

    # --- Generate Random Portfolios for the Efficient Frontier ---
    num_portfolios = 1000
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)

    for i in range(num_portfolios):
        weights = np.random.randn(len(asset_names))
        weights /= np.sum(weights)
        ret_arr[i] = np.dot(weights, mu)
        vol_arr[i] = np.sqrt(portfolio_variance(weights, sigma_matrix))
        sharpe_arr[i] = (ret_arr[i] - rf_annual) / vol_arr[i]

    # (Optional) Filter out extreme random portfolios for clarity
    mask = (vol_arr <= 0.8) & (ret_arr <= 1.5)
    ret_arr = ret_arr[mask]
    vol_arr = vol_arr[mask]
    sharpe_arr = sharpe_arr[mask]

    # --- Compute Portfolio VaR and ES ---
    # Here we compute VaR/ES based on the raw returns for each portfolio
    optimal_portfolio_returns = asset_returns @ optimal_weights
    min_var_portfolio_returns = asset_returns @ min_var_weights

    optimal_var_95, optimal_es_95 = compute_var_es(optimal_portfolio_returns, alpha=0.95)
    min_var_var_95, min_var_es_95 = compute_var_es(min_var_portfolio_returns, alpha=0.95)

    # --- Display Results ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Portfolio Weights")
        portfolio_weights_df = pd.DataFrame({
            "Optimal Portfolio": optimal_weights.round(4),
            "Minimum Variance Portfolio": min_var_weights.round(4)
        }, index=asset_names)
        st.dataframe(portfolio_weights_df)

    with col2:
        portfolio_summary = pd.DataFrame({
            "Optimal Portfolio": [portfolio_return, portfolio_std, portfolio_sharpe, optimal_var_95, optimal_es_95],
            "Minimum Variance Portfolio": [min_var_return, min_var_std, min_var_sharpe, min_var_var_95, min_var_es_95]
        }, index=["Return", "Volatility", "Sharpe Ratio", "95% VaR", "95% ES"])
        st.subheader("Portfolio Summary Statistics")
        st.dataframe(
            portfolio_summary.style
            .format("{:.2%}", subset=pd.IndexSlice[["Return", "Volatility", "95% VaR", "95% ES"], :])
            .format("{:.2f}", subset=pd.IndexSlice[["Sharpe Ratio"], :])
            )


    # --- Plot ---
    fig = go.Figure()

    # Random Portfolios
    fig.add_trace(go.Scatter(
        x=vol_arr,
        y=ret_arr,
        mode='markers',
        marker=dict(
            color=sharpe_arr,
            colorscale='Viridis',
            size=5,
            opacity=0.5,
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        showlegend=False,
        name = 'Random Portfolios'
        
    ))

    # Individual Assets
    fig.add_trace(go.Scatter(
        x=std_devs,
        y=means,
        mode='markers',
        marker=dict(color='#ADD8E6', size=5),
        showlegend=False,
        name = 'Individual Asset'
    ))

    # Optimal Portfolio
    fig.add_trace(go.Scatter(
        x=[portfolio_std],
        y=[portfolio_return],
        mode='markers',
        marker=dict(color='orange', size=12),
        name="Optimal Portfolio",
        showlegend=False
    ))

    # Minimum Variance Portfolio
    fig.add_trace(go.Scatter(
        x=[min_var_std],
        y=[min_var_return],
        mode='markers',
        marker=dict(color='blue', size=12),
        name="Minimum Variance",
        showlegend=False
    ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Expected Return"
    )

    st.subheader("Efficient Frontier")
    st.plotly_chart(fig)
