import yfinance as yf
import pandas as pd
import streamlit as st

# Tickers dictionary mapping names to yfinance ticker symbols
TICKERS = {
    "NIFTY_50": "^NSEI",  # Response Variable
    "SENSEX": "^BSESN",
    "NASDAQ": "^IXIC",
    "DOW_JONES": "^DJI",
    "SP_500": "^GSPC",
    "NIKKEI_225": "^N225",
    "FTSE_100": "^FTSE",
    "SHANGHAI": "000001.SS",
    "GOLD": "GC=F",
    "CRUDE_OIL": "CL=F",
}


from datetime import timedelta

@st.cache_data(ttl=3600)
def fetch_data(start_date, end_date):
    """
    Fetches historical data for all tickers and aligns them by date.
    Returns two dataframes: Adjusted Close prices and Open prices.
    """
    adjusted_end_date = end_date + timedelta(days=1)
    
    data_close = {}
    data_open = {}
    
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=adjusted_end_date, progress=False)
            if not df.empty:
                if "Close" in df.columns:
                    close_prices = df["Close"]
                    if hasattr(close_prices.index, "tz_localize"):
                        close_prices.index = close_prices.index.tz_localize(None)
                    if isinstance(close_prices, pd.DataFrame):
                        close_prices = close_prices.squeeze()
                    data_close[name] = close_prices
                    
                if "Open" in df.columns:
                    open_prices = df["Open"]
                    if hasattr(open_prices.index, "tz_localize"):
                        open_prices.index = open_prices.index.tz_localize(None)
                    if isinstance(open_prices, pd.DataFrame):
                        open_prices = open_prices.squeeze()
                    data_open[name] = open_prices
                    
        except Exception as e:
            print(f"Error fetching data for {name} ({ticker}): {e}")

    if not data_close or not data_open:
        return pd.DataFrame(), pd.DataFrame()

    df_close = pd.DataFrame(data_close)
    df_open = pd.DataFrame(data_open)

    def cleanse(df):
        if df.empty: return df
        df = df[~df.index.duplicated(keep='first')]
        # Exclude redundant weekend days (Saturday=5, Sunday=6)
        df = df[df.index.dayofweek < 5]
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.dropna(axis=0, inplace=True)
        return df

    df_close = cleanse(df_close)
    df_open = cleanse(df_open)
    
    # Ensure both dataframes have the exact same index after cleansing
    common_index = df_close.index.intersection(df_open.index)
    return df_close.loc[common_index], df_open.loc[common_index]


@st.cache_data(ttl=300)
def fetch_live_data():
    """
    Fetches the latest available price and percentage change for all tickers.
    """
    live_data = {}
    for name, ticker in TICKERS.items():
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d")
            if len(hist) >= 2:
                last_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_price - prev_close
                pct_change = (change / prev_close) * 100
                live_data[name] = {
                    "price": last_price,
                    "change": change,
                    "pct_change": pct_change
                }
        except Exception as e:
            print(f"Error fetching live data for {name}: {e}")
    return live_data
