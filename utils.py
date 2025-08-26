import matplotlib.pyplot as plt
import pandas as pd

def plot_price_trends(data):
    fig, ax = plt.subplots()
    for ticker, info in data.items():
        info['history']['Close'].plot(ax=ax, label=ticker)
    ax.legend()
    ax.set_title("Stock Price Trends (Last 30 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    return fig
 


