import yfinance as yf
import pandas as pd
from yahoo_fin import stock_info

def fetch_stock_data(tickers):
    data = {}
    sector_map = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        info = stock.info
        sector_map[ticker] = info.get("sector", "Unknown")
        data[ticker] = {
            "history": hist,
            "current_price": info.get("currentPrice", 0),
            "volatility": hist['Close'].pct_change().std(),
            "sector": info.get("sector", "Unknown")
        }
    return data, sector_map
