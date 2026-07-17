import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import pmdarima as pm
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")


def calculate_returns(df):
    """
    Calculates log returns of the price dataframe to achieve stationarity.
    """
    returns = np.log(df / df.shift(1)).dropna()
    
    # Winsorization (Handling API Data Glitches)
    lower_bound = returns.quantile(0.01)
    upper_bound = returns.quantile(0.99)
    returns = returns.clip(lower=lower_bound, upper=upper_bound, axis=1)
    
    return returns


def prepare_exog(exog_df, lag=1):
    """
    Shifts the exogenous variables by the specified lag to account for timezone differences.
    E.g., if lag=1, today's response is modeled against yesterday's global market returns.
    """
    return exog_df.shift(lag).dropna()


def run_adf_test(series):
    """
    Runs the Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series.dropna())
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4],
    }


def get_acf_pacf(series, nlags=40):
    """
    Calculates ACF and PACF for plotting.
    """
    lag_acf = acf(series.dropna(), nlags=nlags)
    lag_pacf = pacf(series.dropna(), nlags=nlags)
    return lag_acf, lag_pacf


def run_ks_test(series):
    """
    Runs the Kolmogorov-Smirnov test for normality against a standard normal distribution.
    Since we are testing residuals, we standardize them first.
    """
    std_series = (series - np.mean(series)) / np.std(series)
    stat, p_value = stats.kstest(std_series.dropna(), "norm")
    return {"Statistic": stat, "p-value": p_value}


def run_ljung_box(squared_series, lags=10):
    """
    Runs the Ljung-Box test on squared residuals to detect conditional heteroskedasticity.
    """
    # Ljung-Box returns a dataframe in newer statsmodels versions
    res = acorr_ljungbox(squared_series.dropna(), lags=[lags], return_df=True)
    return {"Statistic": res.iloc[0]["lb_stat"], "p-value": res.iloc[0]["lb_pvalue"]}


def fit_auto_arimax(endog_returns, exog_returns):
    """
    Fits an ARIMAX model automatically finding the best p,d,q using AIC.
    """
    try:
        model = pm.auto_arima(
            y=endog_returns,
            X=exog_returns,
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            d=0,  # Log returns are generally stationary
            seasonal=False,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        return model
    except Exception as e:
        print(f"Error fitting Auto-ARIMAX: {e}")
        return None


def fit_garch(residuals, p_garch, q_garch):
    """
    Fits a GARCH(p,q) model to the residuals using a Student's t-distribution for fat tails.
    """
    try:
        # Rescale residuals by 100 for better optimization in arch
        scaled_resids = residuals * 100
        model = arch_model(
            scaled_resids,
            vol="Garch",
            p=p_garch,
            q=q_garch,
            dist="StudentsT",
            rescale=False,
        )
        fitted_model = model.fit(disp="off")
        return fitted_model
    except Exception as e:
        print(f"Error fitting GARCH: {e}")
        return None
