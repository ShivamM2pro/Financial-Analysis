import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def prepare_ml_data(endog, endog_open, exog):
    """
    Combines endog and exog, creates a binary classification target for both Open and Close.
    endog: Series of log returns based on Close prices
    endog_open: Series of log returns based on Open prices (Open_T / Close_T-1)
    """
    endog_renamed = endog.rename(f"{endog.name}_close")
    endog_open_renamed = endog_open.rename(f"{endog.name}_open")
    df = pd.concat([endog_renamed, endog_open_renamed, exog], axis=1).dropna()

    X = df.drop(columns=[endog_renamed.name, endog_open_renamed.name]).copy()
    
    # Feature Engineering
    for col in list(X.columns):
        # Rolling Volatility (Risk)
        X[f"{col}_Vol_5d"] = X[col].rolling(window=5).std()
        X[f"{col}_Vol_20d"] = X[col].rolling(window=20).std()
        
        # Momentum (Moving Averages)
        X[f"{col}_Mom_5d"] = X[col].rolling(window=5).mean()
        
    # Drop rows with NaNs caused by rolling windows
    X.dropna(inplace=True)
    
    y_reg = df.loc[X.index, endog_renamed.name]
    y_clf_close = (y_reg > 0).astype(int)  # 1 for UP, 0 for DOWN
    
    y_reg_open = df.loc[X.index, endog_open_renamed.name]
    y_clf_open = (y_reg_open > 0).astype(int)

    return X, y_reg, y_clf_close, y_clf_open


def time_series_split(X, y, test_size=0.2):
    """
    Strict chronological split. No random shuffling.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def run_xgboost_regression(X_train, X_test, y_train, y_test):
    """
    Trains XGBRegressor and evaluates predictions.
    """
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)

    return model, preds, {"MSE": mse, "RMSE": rmse, "MAE": mae}


def run_xgboost_classification(X_train, X_test, y_train, y_test):
    """
    Trains XGBClassifier and evaluates predictions.
    """
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    return (
        model,
        preds,
        {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ConfusionMatrix": cm,
        },
    )

def run_random_forest_classification(X_train, X_test, y_train, y_test):
    """
    Trains RandomForestClassifier and evaluates predictions, returning probabilities as well.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    return (
        model,
        preds,
        probs,
        {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ConfusionMatrix": cm,
        },
    )
