# src/main.py
import time
import pandas as pd
import yfinance as yf
import joblib

from src.logger import logger;
from src.env_reader import get_env_value
from pathlib import Path
from src.features import make_features_for_latest
from src.trade_manager import execute_order_from_probs

def fetch_today_data(symbol)-> pd.DataFrame:
    try:
        df = yf.download(
            tickers=symbol,
            interval="1m",
            period="1d",
            progress=False,
            auto_adjust=False
        )
        df=df
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        df = df.reset_index()
        df.columns = [col[0] for col in df.columns]
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
        df.drop(columns=['Adj Close'], inplace=True)
        df = df.loc[~(df == 0).any(axis=1)]
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        return None
    

def load_model(model_path: Path):
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise

def sleep_until():
    time.sleep(int(get_env_value("NEXT_ORDER_EXECUTION_TIME")))


def run_loop(model_path: Path):
    model_obj=load_model("Model/lgb_stock_signal_with_calib.pkl")

    while True:
        try:
            latest=fetch_today_data("YESBANK.NS")
            latest_df = latest.tail(1)       # Last row as latest
            history_df = latest.iloc[:-1]  
            # print(latest_df)
            # print(history_df)
            single_row_features = make_features_for_latest(latest_df, history=history_df, feature_cols=model_obj['feature_cols'])
            
            probs = model_obj['calibrator'].predict_proba(single_row_features)
            print(probs)
            price = latest_df['Close'].iloc[-1]
            state, info = execute_order_from_probs(probs, price)
            print(info)
            
        except Exception as e:
            logger.error(f"Error in polling/prediction loop: {e}", exc_info=True)

        sleep_until()


if __name__ == "__main__":
    run_loop(get_env_value("MODEL_NAME"))
