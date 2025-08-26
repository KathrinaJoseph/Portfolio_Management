import yfinance as yf
import pandas as pd

def backtest_portfolio(portfolio, start_days=30):
    tickers = portfolio['Ticker'].str.upper().tolist()
    quantities = dict(portfolio[['Ticker', 'Quantity']].values)

    prices_df = pd.DataFrame()

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=f"{start_days}d", interval='1d', progress=False)
            if df.empty or 'Close' not in df.columns:
                continue
            df = df[['Close']].rename(columns={'Close': ticker})
            prices_df = pd.concat([prices_df, df], axis=1)
        except Exception:
            continue

    if prices_df.empty:
        return None, None

    # Fill missing values
    prices_df.fillna(method='ffill', inplace=True)

    # Calculate portfolio value over time
    portfolio_values = []
    for date, row in prices_df.iterrows():
        total_value = 0
        for ticker in tickers:
            close_price = row.get(ticker, 0)
            qty = quantities.get(ticker, 0)
            total_value += close_price * qty
        portfolio_values.append({'Date': date, 'Total Value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df.set_index('Date', inplace=True)

    # Calculate overall return
    start_value = portfolio_df['Total Value'].iloc[0]
    end_value = portfolio_df['Total Value'].iloc[-1]
    total_return_pct = ((end_value - start_value) / start_value) * 100

    return portfolio_df, round(total_return_pct, 2)
