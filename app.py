from datetime import date
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.figure_factory as ff
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from scipy.stats import norm

from data_loader import fetch_data, fetch_live_data
from ml_models import (prepare_ml_data, run_xgboost_classification,
                       run_xgboost_regression, run_random_forest_classification, time_series_split)
from models import (calculate_returns, fit_auto_arimax, fit_garch,
                    get_acf_pacf, prepare_exog, run_adf_test, run_ks_test,
                    run_ljung_box)
from risk_models import calculate_evt_var, calculate_historical_var

# Set page configuration
st.set_page_config(
    page_title="Financial Intelligence Analysis and Forecasting", layout="wide", page_icon="📈"
)

import base64

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_backgrounds():
    sidebar_bg = r"C:\Users\Shivam\.gemini\antigravity-ide\brain\7aafba30-0824-4f46-aeba-52c04aab38e8\sidebar_bg_1784198232120.png"
    main_bg = r"C:\Users\Shivam\.gemini\antigravity-ide\brain\7aafba30-0824-4f46-aeba-52c04aab38e8\main_bg_1784198241342.png"
    
    try:
        sidebar_base64 = get_base64_of_bin_file(sidebar_bg)
        main_base64 = get_base64_of_bin_file(main_bg)
        
        custom_css = f'''
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{main_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{
            background-image: url("data:image/png;base64,{sidebar_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        /* Add a subtle dark overlay to the main content area for better text readability */
        .block-container {{
            background-color: rgba(14, 17, 23, 0.85);
            border-radius: 15px;
            padding: 2rem !important;
            margin-top: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        </style>
        '''
        st.markdown(custom_css, unsafe_allow_html=True)
    except Exception as e:
        pass

set_backgrounds()

st.title("📈 Financial Intelligence Analysis and Forecasting")

# Live Market Tracker
st.markdown("### Live Market Tracker")
with st.spinner("Fetching live market data..."):
    live_data = fetch_live_data()

if live_data:
    items = list(live_data.items())
    # Display 4 metrics per row to avoid squishing
    for i in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (name, data) in zip(cols, items[i:i+4]):
            col.metric(
                label=name.replace("_", " "),
                value=f"{data['price']:,.2f}",
                delta=f"{data['change']:.2f} ({data['pct_change']:.2f}%)"
            )
st.markdown("---")

st.markdown("### Part 1: Statistical Modeling (ARIMAX & GARCH)")

# Sidebar for configuration
st.sidebar.header("Data Configuration")
start_date = st.sidebar.date_input("Start Date", date(2024, 1, 1), min_value=date(2008, 1, 1), max_value=date.today())
end_date = st.sidebar.date_input("End Date", date.today(), min_value=date(2008, 1, 1), max_value=date.today())

st.sidebar.header("Risk Configuration (Part 3)")

nifty_weight = st.sidebar.slider("NIFTY 50 Portfolio Weight (%)", 0, 100, 50) / 100.0
sensex_weight = 1.0 - nifty_weight
portfolio_value = st.sidebar.number_input(
    "Portfolio Value (INR)", min_value=1000, value=1000000, step=10000
)

st.sidebar.header("Statistical Model Parameters")
lag_days = st.sidebar.number_input(
    "Exogenous Lag (Days)",
    min_value=0,
    max_value=5,
    value=1,
    help="Lags global markets to predict tomorrow's NIFTY",
)
p_garch = st.sidebar.number_input("GARCH (p)", min_value=1, max_value=5, value=1)
q_garch = st.sidebar.number_input("ARCH (q)", min_value=1, max_value=5, value=1)

st.sidebar.header("ML Parameters")
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20) / 100.0

# Fetch Data
with st.spinner("Fetching data from Yahoo Finance..."):
    df_prices, df_open_prices = fetch_data(start_date, end_date)


def render_analysis_for_index(target_index, df_returns, df_open_returns, lag_days, p_garch, q_garch, test_size):
    st.header(f"1. Pre-Modeling Diagnostics (Log Returns) - {target_index}")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stationarity (ADF Test)")
        st.write("Testing Null Hypothesis: Series has a unit root (is non-stationary)")
        adf_res = run_adf_test(df_returns[target_index])
        st.write(f"**ADF Statistic:** {adf_res['ADF Statistic']:.4f}")
        st.write(f"**p-value:** {adf_res['p-value']:.4e}")
        if adf_res["p-value"] < 0.05:
            st.success(
                "Result: The {target_index} log returns are **Stationary** (Reject Null)."
            )
        else:
            st.error(
                "Result: The {target_index} log returns are **Non-Stationary** (Fail to Reject Null)."
            )

    with col2:
        st.subheader("Autocorrelation (ACF & PACF)")
        acf_vals, pacf_vals = get_acf_pacf(df_returns[target_index])
        fig_acf = go.Figure()
        fig_acf.add_trace(
            go.Bar(
                x=np.arange(len(acf_vals)), y=acf_vals, name="ACF", marker_color="blue"
            )
        )
        fig_acf.add_trace(
            go.Scatter(
                x=np.arange(len(pacf_vals)),
                y=pacf_vals,
                mode="markers",
                name="PACF",
                marker_color="red",
                marker_symbol="x",
            )
        )
        fig_acf.update_layout(
            height=300,
            template="plotly_dark",
            title="ACF and PACF of {target_index} Returns",
            barmode="group",
        )
        st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown("---")
    st.header("2. Statistical Modeling")

    # Define Endog and Exog with Lag
    endog, exog = df_returns[target_index], df_returns.drop(columns=[target_index])
    endog_open = df_open_returns[target_index]
    
    exog = prepare_exog(exog, lag=lag_days)
    
    # Ensure endog, endog_open, and exog are aligned after exog shift
    common_idx = endog.index.intersection(exog.index).intersection(endog_open.index)
    endog = endog.loc[common_idx]
    endog_open = endog_open.loc[common_idx]
    exog = exog.loc[common_idx]

    st.subheader("Auto-ARIMAX Mean Model")
    st.write(
        f"Automatically finding optimal (p,d,q) for {target_index} using {lag_days}-day lagged global regressors."
    )

    with st.spinner("Running Auto-ARIMA..."):
        arimax_model = fit_auto_arimax(endog, exog)

    if arimax_model is not None:
        with st.expander("View Auto-ARIMAX Model Summary"):
            st.text(arimax_model.summary().as_text())

        st.success(f"Best Model Selected: **ARIMA{arimax_model.order}**")

        # Plot Actual vs Fitted
        fig_arimax = go.Figure()
        fig_arimax.add_trace(
            go.Scatter(
                x=endog.index,
                y=endog,
                mode="lines",
                name="Actual Returns",
                line=dict(color="cyan", width=1),
            )
        )

        fitted_vals = arimax_model.predict_in_sample(X=exog)
        fig_arimax.add_trace(
            go.Scatter(
                x=fitted_vals.index,
                y=fitted_vals,
                mode="lines",
                name="Fitted Returns",
                line=dict(color="orange", width=1),
            )
        )
        fig_arimax.update_layout(
            title="{target_index} Returns vs Auto-ARIMAX Fitted Values",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig_arimax, use_container_width=True)

        # Residual Diagnostics
        st.markdown("---")
        st.header("3. Post-Modeling Residual Diagnostics")

        residuals = pd.Series(arimax_model.resid(), index=endog.index)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Normality (Kolmogorov-Smirnov Test)")
            ks_res = run_ks_test(residuals)
            st.write(f"**KS Statistic:** {ks_res['Statistic']:.4f}")
            st.write(f"**p-value:** {ks_res['p-value']:.4e}")
            if ks_res["p-value"] < 0.05:
                st.warning(
                    "Result: Residuals are **NOT normally distributed**. (Expected for stock returns - fat tails)."
                )
            else:
                st.success("Result: Residuals appear normally distributed.")

        with col4:
            st.subheader("Homoskedasticity (Ljung-Box on Squared Residuals)")
            lb_res = run_ljung_box(residuals**2)
            st.write(f"**LB Statistic:** {lb_res['Statistic']:.4f}")
            st.write(f"**p-value:** {lb_res['p-value']:.4e}")
            if lb_res["p-value"] < 0.05:
                st.warning(
                    "Result: Evidence of **Conditional Heteroskedasticity** (Volatility clustering exists). Proceeding to GARCH."
                )
            else:
                st.success(
                    "Result: Residuals are homoskedastic. GARCH may not be necessary."
                )

        # Fit GARCH on Residuals
        st.markdown("---")
        st.header("4. Volatility Modeling (GARCH)")
        st.write(
            f"Fitting GARCH({p_garch}, {q_garch}) with Student's t-distribution to capture fat tails and volatility clustering."
        )

        with st.spinner("Fitting GARCH Model..."):
            garch_model = fit_garch(residuals, p_garch, q_garch)

        if garch_model is not None:
            with st.expander("View GARCH Model Summary"):
                st.text(garch_model.summary().as_text())

            cond_vol = garch_model.conditional_volatility / 100  # Rescale back
            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Scatter(
                    x=cond_vol.index,
                    y=cond_vol,
                    mode="lines",
                    name="Conditional Volatility",
                    line=dict(color="red"),
                )
            )
            fig_vol.update_layout(
                title="Estimated Conditional Volatility (Risk)",
                height=400,
                template="plotly_dark",
                yaxis_title="Volatility",
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            # We will use this cond_vol for Part 3 GARCH VaR.
            garch_vol_series = cond_vol
        else:
            st.error("Failed to fit GARCH model.")
            garch_vol_series = None
    else:
        st.error("Failed to fit Auto-ARIMAX model.")

    # ---------------------------------------------------------
    # PART 2: MACHINE LEARNING (XGBOOST)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("---")
    st.title("🤖 Part 2: Machine Learning (Multicollinearity Resilient)")
    st.write(
        "Using XGBoost to natively handle multicollinearity amongst global indices. We use a strict chronological 80/20 train-test split."
    )


    # Prepare data for ML
    X_ml, y_reg, y_clf_close, y_clf_open = prepare_ml_data(endog, endog_open, exog)

    # Split
    X_train, X_test, y_train_reg, y_test_reg = time_series_split(
        X_ml, y_reg, test_size=test_size
    )
    _, _, y_train_clf_close, y_test_clf_close = time_series_split(X_ml, y_clf_close, test_size=test_size)
    _, _, y_train_clf_open, y_test_clf_open = time_series_split(X_ml, y_clf_open, test_size=test_size)

    col_ml1, col_ml2, col_ml3 = st.columns(3)

    # XGBoost Regressor
    with col_ml1:
        st.header("A. Regression (Predict Exact Return)")
        with st.spinner("Training XGBRegressor..."):
            model_reg, preds_reg, metrics_reg = run_xgboost_regression(
                X_train, X_test, y_train_reg, y_test_reg
            )

        with st.expander("View Model Specifications & Insights"):
            st.write("**Model Type:** XGBoost Regressor")
            st.json(model_reg.get_params())

            st.write("**Feature Importances (Model Internal Logic)**")
            importances_reg = model_reg.feature_importances_
            fig_fi_reg = go.Figure(go.Bar(x=importances_reg, y=X_train.columns, orientation='h'))
            fig_fi_reg.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark", title="What is driving the predictions?")
            st.plotly_chart(fig_fi_reg, use_container_width=True)

        st.write("**Evaluation on Test Set (Unseen Future Data)**")
        st.write(f"- **MSE:** {metrics_reg['MSE']:.6f}")
        st.write(f"- **RMSE:** {metrics_reg['RMSE']:.6f}")
        st.write(f"- **MAE:** {metrics_reg['MAE']:.6f}")

        # Plot Regressor Predictions
        fig_ml_reg = go.Figure()
        fig_ml_reg.add_trace(
            go.Scatter(
                x=y_test_reg.index,
                y=y_test_reg,
                mode="lines",
                name="Actual Return",
                line=dict(color="cyan"),
            )
        )
        fig_ml_reg.add_trace(
            go.Scatter(
                x=y_test_reg.index,
                y=preds_reg,
                mode="lines",
                name="Predicted Return",
                line=dict(color="orange"),
            )
        )
        fig_ml_reg.update_layout(
            title="XGBoost Regression Predictions", height=350, template="plotly_dark"
        )
        st.plotly_chart(fig_ml_reg, use_container_width=True)

    # XGBoost Classifier
    with col_ml2:
        st.header("B. Classification (XGBoost)")
        with st.spinner("Training XGBClassifier..."):
            model_clf_close, preds_clf_close, metrics_clf_close = run_xgboost_classification(
                X_train, X_test, y_train_clf_close, y_test_clf_close
            )
            model_clf_open, preds_clf_open, metrics_clf_open = run_xgboost_classification(
                X_train, X_test, y_train_clf_open, y_test_clf_open
            )

        with st.expander("View Model Specifications & Insights"):
            st.write("**Model Type:** XGBoost Classifier")
            st.json(model_clf_close.get_params())
            
        st.write("**Evaluation on Test Set (Unseen Future Data)**")
        cx1, cx2 = st.columns(2)
        with cx1:
            st.write("**Morning Gap (Open)**")
            st.write(f"- Acc: {metrics_clf_open['Accuracy'] * 100:.2f}%")
            st.write(f"- F1: {metrics_clf_open['F1-Score'] * 100:.2f}%")
        with cx2:
            st.write("**Closing Trend (Close)**")
            st.write(f"- Acc: {metrics_clf_close['Accuracy'] * 100:.2f}%")
            st.write(f"- F1: {metrics_clf_close['F1-Score'] * 100:.2f}%")

        # Confusion Matrix Heatmap (Close)
        cm = metrics_clf_close["ConfusionMatrix"]
        z = cm[::-1]
        x = ["Pred: Down (0)", "Pred: Up (1)"]
        y = ["Actual: Up (1)", "Actual: Down (0)"]

        fig_cm = ff.create_annotated_heatmap(
            z, x=x, y=y, colorscale="Viridis", showscale=True
        )
        fig_cm.update_layout(
            title="Confusion Matrix (Close)", height=350, template="plotly_dark"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Random Forest Classifier
    with col_ml3:
        st.header("C. Classification (Random Forest)")
        with st.spinner("Training Random Forest..."):
            rf_model_close, preds_rf_close, probs_rf_close, metrics_rf_close = run_random_forest_classification(
                X_train, X_test, y_train_clf_close, y_test_clf_close
            )
            rf_model_open, preds_rf_open, probs_rf_open, metrics_rf_open = run_random_forest_classification(
                X_train, X_test, y_train_clf_open, y_test_clf_open
            )

        with st.expander("View Model Specifications & Insights"):
            st.write("**Model Type:** Random Forest Classifier")
            st.json(rf_model_close.get_params())
            
        st.write("**Evaluation on Test Set (Unseen Future Data)**")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Morning Gap (Open)**")
            st.write(f"- Acc: {metrics_rf_open['Accuracy'] * 100:.2f}%")
            st.write(f"- F1: {metrics_rf_open['F1-Score'] * 100:.2f}%")
        with c2:
            st.write("**Closing Trend (Close)**")
            st.write(f"- Acc: {metrics_rf_close['Accuracy'] * 100:.2f}%")
            st.write(f"- F1: {metrics_rf_close['F1-Score'] * 100:.2f}%")

        # Confusion Matrix Heatmap
        cm_rf = metrics_rf_close["ConfusionMatrix"]
        z_rf = cm_rf[::-1]
        x_rf = ["Pred: Down (0)", "Pred: Up (1)"]
        y_rf = ["Actual: Up (1)", "Actual: Down (0)"]

        fig_cm_rf = ff.create_annotated_heatmap(
            z_rf, x=x_rf, y=y_rf, colorscale="Cividis", showscale=True
        )
        fig_cm_rf.update_layout(
            title="Confusion Matrix (RF - Close)", height=350, template="plotly_dark"
        )
        st.plotly_chart(fig_cm_rf, use_container_width=True)



    # Return results for Part 3 and Part 4
    return {
        "returns": endog.dropna(),
        "garch_vol_series": garch_vol_series if "garch_vol_series" in locals() else None,
        "X_ml": X_ml,
        "rf_model_close": rf_model_close if "rf_model_close" in locals() else None,
        "rf_model_open": rf_model_open if "rf_model_open" in locals() else None,
        "model_reg": model_reg if "model_reg" in locals() else None
    }

if df_prices.empty or 'NIFTY_50' not in df_prices.columns or 'SENSEX' not in df_prices.columns:

    st.error("Error loading data. Please check the date range or internet connection.")
else:
    st.subheader("Historical Normalized Prices (Base 100)")
    normalized_prices = (df_prices / df_prices.iloc[0]) * 100
    fig_prices = go.Figure()
    for col in normalized_prices.columns:
        fig_prices.add_trace(
            go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[col],
                mode="lines",
                name=col,
            )
        )
    fig_prices.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        template="plotly_dark",
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    # Calculate Returns
    df_returns = calculate_returns(df_prices)
    import numpy as np
    df_open_returns = np.log(df_open_prices / df_prices.shift(1)).dropna()

    st.markdown("---")

    tab1, tab2 = st.tabs(["NIFTY 50 Analysis", "BSE SENSEX Analysis"])
    results = {}
    
    with tab1:
        results["NIFTY_50"] = render_analysis_for_index("NIFTY_50", df_returns, df_open_returns, lag_days, p_garch, q_garch, test_size)
        
    with tab2:
        results["SENSEX"] = render_analysis_for_index("SENSEX", df_returns, df_open_returns, lag_days, p_garch, q_garch, test_size)

    # --- Combined Ensemble Accuracy ---
    n_rf_close = results["NIFTY_50"]["rf_model_close"]
    s_rf_close = results["SENSEX"]["rf_model_close"]
    n_rf_open = results["NIFTY_50"]["rf_model_open"]
    s_rf_open = results["SENSEX"]["rf_model_open"]
    
    if n_rf_close and s_rf_close:
        X_ml_n = results["NIFTY_50"]["X_ml"]
        X_ml_s = results["SENSEX"]["X_ml"]
        
        split_idx = int(len(X_ml_n) * (1 - test_size))
        
        X_test_n = X_ml_n.iloc[split_idx:]
        X_test_s = X_ml_s.iloc[split_idx:]
        
        # Extract Probability of Class 1 (Up) for the entire test set
        n_probs = n_rf_close.predict_proba(X_test_n)[:, 1]
        s_probs = s_rf_close.predict_proba(X_test_s)[:, 1]
        
        # Weight the probabilities
        combined_probs = (n_probs * nifty_weight) + (s_probs * sensex_weight)
        combined_preds = (combined_probs > 0.5).astype(int)
        
        # Get Actual Direction of the Combined Portfolio
        combined_returns_eval = (results["NIFTY_50"]["returns"] * nifty_weight) + (results["SENSEX"]["returns"] * sensex_weight)
        combined_returns_test = combined_returns_eval.loc[X_test_n.index]
        actual_direction = (combined_returns_test > 0).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        ens_acc = accuracy_score(actual_direction, combined_preds)
        ens_prec = precision_score(actual_direction, combined_preds, zero_division=0)
        ens_rec = recall_score(actual_direction, combined_preds, zero_division=0)
        ens_f1 = f1_score(actual_direction, combined_preds, zero_division=0)
        
        st.markdown("---")
        st.subheader("🎯 Ensemble Strategy Performance (Test Set)")
        st.write(f"This represents the accuracy of trading your specific **{int(nifty_weight*100)}% NIFTY / {int(sensex_weight*100)}% SENSEX** blended portfolio using the combined predictive power of both Random Forest models.")
        
        col_ens1, col_ens2, col_ens3, col_ens4 = st.columns(4)
        col_ens1.metric("Ensemble Accuracy", f"{ens_acc * 100:.2f}%")
        col_ens2.metric("Ensemble Precision", f"{ens_prec * 100:.2f}%")
        col_ens3.metric("Ensemble Recall", f"{ens_rec * 100:.2f}%")
        col_ens4.metric("Ensemble F1-Score", f"{ens_f1 * 100:.2f}%")


    # ---------------------------------------------------------
    # PART 3: QUANTITATIVE RISK MANAGEMENT (VaR & EVT)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("---")
    st.title("🛡️ Part 3: Quantitative Risk Management (Combined Portfolio)")
    st.write(
        f"Calculating 99% Value at Risk (VaR) and Expected Shortfall (ES) for a Portfolio of **₹{portfolio_value:,.2f}**"
    )

    combined_returns = (results["NIFTY_50"]["returns"] * nifty_weight) + (results["SENSEX"]["returns"] * sensex_weight)
    combined_returns = combined_returns.dropna()
    alpha = 0.01

    # 1. Historical
    hist_var, hist_es = calculate_historical_var(combined_returns, alpha)

    # 2. EVT
    evt_var, evt_es = calculate_evt_var(combined_returns, alpha)

    # 3. Parametric GARCH (Combined approach approximation)
    garch_vol_nifty = results["NIFTY_50"]["garch_vol_series"]
    garch_vol_sensex = results["SENSEX"]["garch_vol_series"]
    
    if garch_vol_nifty is not None and garch_vol_sensex is not None:
        # Assuming correlation of 1 for conservative portfolio variance proxy or just weighted sum of vols
        # A true multivariate GARCH is better, but this approximates the combined risk
        combined_vol = (garch_vol_nifty * nifty_weight) + (garch_vol_sensex * sensex_weight)
        latest_vol = combined_vol.iloc[-1]
        z_score = norm.ppf(alpha)
        garch_var_current = z_score * latest_vol
        garch_es_current = -latest_vol * norm.pdf(z_score) / alpha
        garch_vol_series = combined_vol
    else:
        garch_var_current = np.nan
        garch_es_current = np.nan
        garch_vol_series = None
    # Comparative Table

    risk_data = {
        "Methodology": [
            "Historical (Non-Parametric)",
            "Extreme Value Theory (EVT/POT)",
            "Dynamic GARCH (Parametric)",
        ],
        "99% VaR (%)": [hist_var * 100, evt_var * 100, garch_var_current * 100],
        "99% ES (CVaR) (%)": [hist_es * 100, evt_es * 100, garch_es_current * 100],
        "Value at Risk (INR)": [
            hist_var * portfolio_value,
            evt_var * portfolio_value,
            garch_var_current * portfolio_value,
        ],
        "Expected Shortfall (INR)": [
            hist_es * portfolio_value,
            evt_es * portfolio_value,
            garch_es_current * portfolio_value,
        ],
    }

    df_risk = pd.DataFrame(risk_data)

    # Format the table for display
    def format_money(val):
        if pd.isna(val):
            return "N/A"
        return f"₹{val:,.2f}"

    def format_pct(val):
        if pd.isna(val):
            return "N/A"
        return f"{val:.2f}%"

    df_risk_display = df_risk.copy()
    df_risk_display["99% VaR (%)"] = df_risk_display["99% VaR (%)"].apply(format_pct)
    df_risk_display["99% ES (CVaR) (%)"] = df_risk_display["99% ES (CVaR) (%)"].apply(
        format_pct
    )
    df_risk_display["Value at Risk (INR)"] = df_risk_display[
        "Value at Risk (INR)"
    ].apply(format_money)
    df_risk_display["Expected Shortfall (INR)"] = df_risk_display[
        "Expected Shortfall (INR)"
    ].apply(format_money)

    st.table(df_risk_display.set_index("Methodology"))

    # Visualizations
    col_risk1, col_risk2 = st.columns(2)

    with col_risk1:
        st.subheader("Return Distribution & Tail Risk")
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=combined_returns,
                nbinsx=100,
                name="Returns",
                marker_color="gray",
                opacity=0.7,
            )
        )

        # Add VaR lines
        fig_hist.add_vline(
            x=hist_var,
            line_dash="dash",
            line_color="yellow",
            annotation_text="Hist VaR",
        )
        fig_hist.add_vline(
            x=evt_var, line_dash="dash", line_color="red", annotation_text="EVT VaR"
        )
        if not np.isnan(garch_var_current):
            fig_hist.add_vline(
                x=garch_var_current,
                line_dash="dash",
                line_color="blue",
                annotation_text="GARCH VaR",
            )

        fig_hist.update_layout(
            height=400,
            template="plotly_dark",
            title="Histogram of Combined Portfolio Returns with 99% VaR Thresholds",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_risk2:
        st.subheader("Dynamic VaR Backtest (GARCH)")
        if "garch_vol_series" in locals() and garch_vol_series is not None:
            # Reconstruct historical dynamic VaR over time
            dynamic_var_history = norm.ppf(alpha) * garch_vol_series

            fig_backtest = go.Figure()
            fig_backtest.add_trace(
                go.Scatter(
                    x=combined_returns.index,
                    y=combined_returns,
                    mode="lines",
                    name="Actual Returns",
                    line=dict(color="cyan", width=1),
                )
            )
            fig_backtest.add_trace(
                go.Scatter(
                    x=dynamic_var_history.index,
                    y=dynamic_var_history,
                    mode="lines",
                    name="99% Dynamic VaR",
                    line=dict(color="red", width=1),
                )
            )

            # Highlight breaches
            breaches = combined_returns[combined_returns < dynamic_var_history]
            fig_backtest.add_trace(
                go.Scatter(
                    x=breaches.index,
                    y=breaches,
                    mode="markers",
                    name="VaR Breaches",
                    marker=dict(color="yellow", size=6, symbol="x"),
                )
            )

            fig_backtest.update_layout(
                height=400,
                template="plotly_dark",
                title="Backtesting GARCH VaR vs Actual Returns",
            )
            st.plotly_chart(fig_backtest, use_container_width=True)
        else:
            st.warning("GARCH model failed; cannot render Dynamic VaR backtest.")


    # ---------------------------------------------------------
    # PART 4: FINAL OUTCOME - MARKET HEALTH (Combined)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("---")
    st.title("🚦 Part 4: Final Outcome - Market Health (Combined Portfolio)")
    st.write("Synthesizing the Machine Learning Trend Prediction with the Quantitative Risk (VaR) to provide a definitive market outlook.")

    # 1. Trend Prediction (Morning Gap vs Closing Trend)
    n_features = results["NIFTY_50"]["X_ml"].iloc[-1:]
    s_features = results["SENSEX"]["X_ml"].iloc[-1:]
    
    # Open Probs
    n_probs_open = n_rf_open.predict_proba(n_features)[0] if n_rf_open else [0.5, 0.5]
    s_probs_open = s_rf_open.predict_proba(s_features)[0] if s_rf_open else [0.5, 0.5]
    bull_prob_open = ((n_probs_open[1] * nifty_weight) + (s_probs_open[1] * sensex_weight)) * 100
    
    # Close Probs
    n_probs_close = n_rf_close.predict_proba(n_features)[0] if n_rf_close else [0.5, 0.5]
    s_probs_close = s_rf_close.predict_proba(s_features)[0] if s_rf_close else [0.5, 0.5]
    bull_prob_close = ((n_probs_close[1] * nifty_weight) + (s_probs_close[1] * sensex_weight)) * 100
    
    latest_pred_class = 1 if bull_prob_close > 50 else 0
    
    if bull_prob_open > 50:
        trend_status_open = f"Open Green ({bull_prob_open:.1f}%)"
        trend_color_open = "#00FF00"
    else:
        trend_status_open = f"Open Red ({100-bull_prob_open:.1f}%)"
        trend_color_open = "#FF0000"

    if bull_prob_close > 50:
        trend_status_close = f"Close Green ({bull_prob_close:.1f}%)"
        trend_color_close = "#00FF00"
    else:
        trend_status_close = f"Close Red ({100-bull_prob_close:.1f}%)"
        trend_color_close = "#FF0000"

    # 1b. Trend Drivers (Feature Importances - Averaged from Close model)
    if n_rf_close and s_rf_close:
        n_importances = n_rf_close.feature_importances_
        s_importances = s_rf_close.feature_importances_
        # Ensure they have the same columns
        df_n = pd.DataFrame({"Feature": results["NIFTY_50"]["X_ml"].columns, "N_Imp": n_importances})
        df_s = pd.DataFrame({"Feature": results["SENSEX"]["X_ml"].columns, "S_Imp": s_importances})
        
        drivers = pd.merge(df_n, df_s, on="Feature")
        drivers["Importance"] = (drivers["N_Imp"] * nifty_weight) + (drivers["S_Imp"] * sensex_weight)
        top_drivers = drivers.sort_values(by="Importance", ascending=False)
        driver_1 = top_drivers.iloc[0]
        driver_2 = top_drivers.iloc[1]
    else:
        driver_1 = {"Feature": "N/A", "Importance": 0}
        driver_2 = {"Feature": "N/A", "Importance": 0}

    # 2. Risk Assessment
    if "garch_var_current" in locals() and not pd.isna(garch_var_current):
        current_risk_pct = abs(garch_var_current) * 100
        if current_risk_pct > 2.5:
            risk_status = f"High Risk (VaR: {current_risk_pct:.2f}%)"
            risk_color = "#FF4500" # OrangeRed
        elif current_risk_pct > 1.5:
            risk_status = f"Moderate Risk (VaR: {current_risk_pct:.2f}%)"
            risk_color = "#FFA500" # Orange
        else:
            risk_status = f"Low Risk / Stable (VaR: {current_risk_pct:.2f}%)"
            risk_color = "#00FF00" # Green
    else:
        risk_status = "Unknown (Risk model failed)"
        risk_color = "gray"
        current_risk_pct = 0

    from pandas.tseries.offsets import BDay
    last_date = df_prices.index[-1]
    next_session_date = (last_date + BDay(1)).strftime('%Y-%m-%d')

    col_health1, col_health2, col_health3 = st.columns(3)
    with col_health1:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid {trend_color_open}; height: 100%;">
            <h3>🌅 Morning Momentum</h3>
            <p style="color: #bbb; font-size: 0.9em;">Will it Gap Up?</p>
            <h2 style="color: {trend_color_open};">{trend_status_open}</h2>
            <hr style="border-color: #444;">
            <p style="margin-bottom: 0;">Date: {next_session_date}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_health2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid {trend_color_close}; height: 100%;">
            <h3>🌇 Closing Trend</h3>
            <p style="color: #bbb; font-size: 0.9em;">Will it End Green?</p>
            <h2 style="color: {trend_color_close};">{trend_status_close}</h2>
            <hr style="border-color: #444;">
            <p style="margin-bottom: 0;"><b>#1 Driver:</b> {driver_1['Feature']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_health3:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid {risk_color}; height: 100%;">
            <h3>⚠️ Risk Management</h3>
            <p style="color: #bbb; font-size: 0.9em;">Current VaR</p>
            <h2 style="color: {risk_color};">{risk_status}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 3. Financial Impact Projection
    n_reg = results["NIFTY_50"]["model_reg"]
    s_reg = results["SENSEX"]["model_reg"]
    
    n_ret = n_reg.predict(n_features)[0] if n_reg else 0
    s_ret = s_reg.predict(s_features)[0] if s_reg else 0
    
    latest_predicted_return = (n_ret * nifty_weight) + (s_ret * sensex_weight)
    expected_pnl = latest_predicted_return * portfolio_value
    
    if expected_pnl >= 0:
        pnl_color = "#00FF00"
        pnl_text = f"+₹{expected_pnl:,.2f}"
    else:
        pnl_color = "#FF0000"
        pnl_text = f"-₹{abs(expected_pnl):,.2f}"
        
    if np.isnan(garch_var_current):
        var_text = "N/A"
    else:
        var_text = f"-₹{abs(garch_var_current * portfolio_value):,.2f}"

    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid #aaa; margin-top: 15px; margin-bottom: 25px;">
        <h3>Financial Impact Projection (Next Session)</h3>
        <p style="font-size: 1.2em;">Based on Portfolio Value: <b>₹{portfolio_value:,.2f}</b></p>
        <hr style="border-color: #444; width: 50%; margin: auto;">
        <div style="display: flex; justify-content: space-around; margin-top: 15px;">
            <div>
                <p style="margin-bottom: 0;">Predicted Portfolio Move</p>
                <h3 style="color: {pnl_color}; margin-top: 5px;">{latest_predicted_return * 100:.2f}%</h3>
            </div>
            <div>
                <p style="margin-bottom: 0;">Expected Gain/Loss</p>
                <h3 style="color: {pnl_color}; margin-top: 5px;">{pnl_text}</h3>
            </div>
            <div>
                <p style="margin-bottom: 0;">99% Max Downside (VaR)</p>
                <h3 style="color: {risk_color}; margin-top: 5px;">{var_text}</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Synthesized Strategy
    if latest_pred_class == 1 and current_risk_pct > 2.5:
        strategy = "⚠️ **Cautious Bullish:** The model expects an upward move, but market volatility is very high. Consider smaller position sizes and tight stop-losses."
    elif latest_pred_class == 1 and current_risk_pct <= 2.5:
        strategy = "✅ **Confident Bullish:** Favorable conditions for long positions with stable/moderate risk."
    elif latest_pred_class == 0 and current_risk_pct > 2.5:
        strategy = "🛑 **High Conviction Bearish:** The model expects a downward move amid high volatility. High risk of sudden drawdowns. Consider hedging or staying in cash."
    elif latest_pred_class == 0 and current_risk_pct <= 2.5:
        strategy = "🛡️ **Defensive / Bearish:** Downward move expected, but without extreme volatility. Consider reducing exposure."
    else:
        strategy = "Monitor the market closely."

    st.info(f"💡 **Synthesized Strategy:** {strategy}")
