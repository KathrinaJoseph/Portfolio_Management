import numpy as np
import yfinance as yf
from yahoo_fin import stock_info
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import streamlit as st
import random
import tensorflow as tf
from scipy import stats
import pandas as pd

# Fix seeds to make LSTM training deterministic
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Cache to prevent repeated LSTM retraining on the same data
forecast_cache = {}

def calculate_enhanced_risk_score(data, sentiment_scores, market_data=None):
    """
    Enhanced risk score calculation incorporating multiple risk factors:
    - Volatility (30-day and 90-day)
    - Value at Risk (VaR)
    - Maximum Drawdown
    - Beta (if market data available)
    - Sharpe Ratio
    - Sentiment Score
    """
    risk_scores = {}
    
    for ticker, metrics in data.items():
        try:
            history = metrics["history"]
            if len(history) < 30:
                risk_scores[ticker] = 0.5  # Default moderate risk
                continue
            
            # Calculate returns
            returns = history['Close'].pct_change().dropna()
            
            # 1. Multi-timeframe volatility
            vol_30 = returns.tail(30).std() * np.sqrt(252)  # Annualized 30-day vol
            vol_90 = returns.tail(90).std() * np.sqrt(252) if len(returns) >= 90 else vol_30
            volatility_score = (vol_30 * 0.7 + vol_90 * 0.3)
            
            # 2. Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5)
            var_score = abs(var_95)
            
            # 3. Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # 4. Beta calculation (if market data available)
            beta_score = 1.0  # Default market beta
            if market_data is not None:
                market_returns = market_data.pct_change().dropna()
                aligned_returns = returns.align(market_returns, join='inner')
                if len(aligned_returns[0]) > 20:
                    beta_score = np.cov(aligned_returns[0], aligned_returns[1])[0,1] / np.var(aligned_returns[1])
            
            # 5. Sharpe Ratio (risk-free rate assumed 0.02)
            excess_returns = returns.mean() * 252 - 0.02  # Annualized excess return
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            sharpe_score = max(0, 1 - sharpe_ratio) if sharpe_ratio > 0 else 1
            
            # 6. Sentiment adjustment
            sentiment = sentiment_scores.get(ticker, 0)
            sentiment_adjustment = (1 - sentiment) * 0.5  # Reduce impact of sentiment
            
            # Combine all risk factors with weights
            risk_score = (
                volatility_score * 0.25 +
                var_score * 0.20 +
                max_drawdown * 0.20 +
                abs(beta_score - 1) * 0.15 +  # Deviation from market
                sharpe_score * 0.10 +
                sentiment_adjustment * 0.10
            )
            
            # Normalize to 0-1 range
            risk_scores[ticker] = min(max(risk_score, 0), 1)
            
        except Exception as e:
            print(f"Error calculating risk for {ticker}: {e}")
            risk_scores[ticker] = 0.5  # Default moderate risk
    
    return risk_scores

def forecast_future_return_lstm(history, ticker, epochs=50, look_back=20):
    """
    Enhanced LSTM forecasting with improved architecture and training.
    """
    cache_key = f"{ticker}_{len(history)}"
    if cache_key in forecast_cache:
        return forecast_cache[cache_key]
    
    df = history[['Close', 'Volume', 'High', 'Low']].dropna()
    if len(df) <= look_back:
        return df['Close'].iloc[-1]
    
    # Use more recent data but ensure sufficient history
    df = df.tail(min(300, len(df)))
    
    # Feature engineering
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
    df = df.dropna()
    
    # Prepare features (Close price + technical indicators)
    features = ['Close', 'Volume', 'Price_Change', 'High_Low_Ratio']
    data = df[features].values
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i+look_back])
        y.append(scaled_data[i+look_back, 0])  # Predict Close price
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        return df['Close'].iloc[-1]
    
    # Enhanced LSTM model
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    # Train model
    model.fit(X, y, epochs=epochs, batch_size=32, shuffle=False, 
              verbose=0, callbacks=[early_stopping])
    
    # Make prediction
    last_sequence = scaled_data[-look_back:]
    last_sequence = np.reshape(last_sequence, (1, look_back, X.shape[2]))
    pred_scaled = model.predict(last_sequence, verbose=0)
    
    # Inverse transform prediction
    pred_full = np.zeros((1, len(features)))
    pred_full[0, 0] = pred_scaled[0, 0]
    predicted_price = scaler.inverse_transform(pred_full)[0, 0]
    
    forecast_cache[cache_key] = round(predicted_price, 2)
    return forecast_cache[cache_key]

def calculate_portfolio_metrics(current_allocations, data):
    """
    Calculate portfolio-level metrics for better rebalancing decisions.
    """
    portfolio_returns = []
    weights = list(current_allocations.values())
    
    # Align all stock returns
    all_returns = {}
    min_length = float('inf')
    
    for ticker in current_allocations:
        if ticker in data:
            returns = data[ticker]['history']['Close'].pct_change().dropna()
            all_returns[ticker] = returns
            min_length = min(min_length, len(returns))
    
    if not all_returns or min_length < 30:
        return None
    
    # Calculate portfolio returns
    portfolio_return_series = pd.Series(0, index=list(all_returns.values())[0].tail(min_length).index)
    
    for i, (ticker, weight) in enumerate(current_allocations.items()):
        if ticker in all_returns:
            aligned_returns = all_returns[ticker].tail(min_length)
            portfolio_return_series += weight * aligned_returns
    
    return {
        'returns': portfolio_return_series,
        'volatility': portfolio_return_series.std() * np.sqrt(252),
        'sharpe': (portfolio_return_series.mean() * 252) / (portfolio_return_series.std() * np.sqrt(252))
    }

def rebalance_portfolio_enhanced(current_allocations, risk_scores, data, target_return=0.12):
    """
    Enhanced portfolio rebalancing using Modern Portfolio Theory principles.
    """
    recommended = {}
    expected_returns = {}
    
    # Calculate expected returns and confidence scores
    for ticker in current_allocations:
        try:
            if ticker not in data:
                continue
                
            future_price = forecast_future_return_lstm(data[ticker]['history'], ticker)
            current_price = data[ticker]['current_price']
            
            if current_price <= 0:
                continue
            
            # Expected return
            expected_return = (future_price - current_price) / current_price
            expected_returns[ticker] = expected_return
            
            # Calculate prediction confidence based on historical accuracy
            history = data[ticker]['history']
            recent_volatility = history['Close'].pct_change().tail(30).std()
            confidence = 1 / (1 + recent_volatility * 10)  # Higher volatility = lower confidence
            
            # Risk-adjusted return with confidence weighting
            risk_adjusted_return = expected_return * confidence / (1 + risk_scores[ticker])
            recommended[ticker] = risk_adjusted_return
            
        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")
            continue
    
    if not recommended:
        return {ticker: 1/len(current_allocations) for ticker in current_allocations}
    
    # Apply portfolio constraints
    # 1. No single stock > 40% of portfolio
    # 2. Minimum allocation of 5% for diversification
    # 3. Prefer positive expected returns
    
    # Normalize to positive values
    min_val = min(recommended.values())
    if min_val < 0:
        for ticker in recommended:
            recommended[ticker] -= min_val
    
    # Apply constraints
    total_value = sum(recommended.values())
    if total_value > 0:
        for ticker in recommended:
            weight = recommended[ticker] / total_value
            # Apply max and min constraints
            weight = min(weight, 0.4)  # Max 40%
            weight = max(weight, 0.05)  # Min 5%
            recommended[ticker] = weight
    
    # Renormalize after constraints
    total = sum(recommended.values())
    if total > 0:
        for ticker in recommended:
            recommended[ticker] = round(recommended[ticker] / total, 4)
    else:
        # Equal weighting fallback
        equal_weight = 1 / len(recommended)
        for ticker in recommended:
            recommended[ticker] = round(equal_weight, 4)
    
    return recommended

def get_rebalancing_insights(current_allocations, recommended_allocations, risk_scores, data):
    """
    Provide insights and reasoning for rebalancing decisions.
    """
    insights = []
    
    for ticker in current_allocations:
        current_weight = current_allocations.get(ticker, 0)
        new_weight = recommended_allocations.get(ticker, 0)
        risk_score = risk_scores.get(ticker, 0)
        
        change = new_weight - current_weight
        
        if abs(change) > 0.05:  # Significant change threshold
            direction = "increase" if change > 0 else "decrease"
            risk_level = "high" if risk_score > 0.6 else "moderate" if risk_score > 0.3 else "low"
            
            insight = f"{ticker}: {direction.capitalize()} allocation by {abs(change):.1%} "
            insight += f"(Risk: {risk_level}, Score: {risk_score:.3f})"
            insights.append(insight)
    
    return insights