import numpy as np
from scipy.stats import genpareto


def calculate_historical_var(returns, alpha=0.01):
    """
    Calculates Historical VaR and Expected Shortfall (CVaR).
    alpha=0.01 means 99% confidence.
    """
    var = np.percentile(returns, alpha * 100)
    # Expected Shortfall: average of returns worse than VaR
    es = returns[returns <= var].mean()
    return var, es


def calculate_evt_var(returns, alpha=0.01):
    """
    Calculates VaR and ES using Extreme Value Theory (POT method).
    Fits a Generalized Pareto Distribution to the left tail.
    """
    # Isolate the left tail (losses). We invert the sign so losses are positive for GPD fitting.
    losses = -returns

    # Define a threshold (e.g., 90th percentile of losses)
    threshold = np.percentile(losses, 90)

    # Exceedances over threshold
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 10:
        # Not enough data for robust EVT, fallback to Historical
        var, es = calculate_historical_var(returns, alpha)
        return var, es

    # Fit Generalized Pareto Distribution (c=shape, loc, scale)
    # We fix loc=0 since exceedances are already shifted by threshold
    shape, loc, scale = genpareto.fit(exceedances, floc=0)

    # N_u = number of exceedances, n = total observations
    n = len(returns)
    n_u = len(exceedances)

    # Probability of exceeding the threshold
    prob_exceed = n_u / n

    # Calculate EVT VaR
    # Formula: VaR = u + (scale/shape) * [((alpha / prob_exceed) ^ -shape) - 1]
    # alpha here is the tail probability (0.01)

    try:
        if shape != 0:
            var_evt = threshold + (scale / shape) * (
                ((alpha / prob_exceed) ** -shape) - 1
            )
            # Expected shortfall for GPD
            es_evt = (var_evt + scale - shape * threshold) / (1 - shape)
        else:
            var_evt = threshold - scale * np.log(alpha / prob_exceed)
            es_evt = var_evt + scale

        # Re-invert sign to match return series (negative numbers)
        return -var_evt, -es_evt
    except Exception as e:
        print(f"EVT calculation failed: {e}")
        var, es = calculate_historical_var(returns, alpha)
        return var, es
