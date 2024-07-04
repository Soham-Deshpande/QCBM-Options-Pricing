import yfinance as yf
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def black_scholes_merton(S, X, T, r, sigma, option_type='call'):
    d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close']

def plot_option_prices(stock_prices, strike_price, time_to_expiration, risk_free_rate, volatility):
    option_types = ['call', 'put']
    option_prices = {}

    for option_type in option_types:
        option_prices[option_type] = []

        for current_stock_price in stock_prices:
            option_price = black_scholes_merton(current_stock_price, strike_price, time_to_expiration,
                                                risk_free_rate, volatility, option_type)
            option_prices[option_type].append(option_price)

    plt.figure(figsize=(10, 6))

    for option_type, prices in option_prices.items():
        plt.plot(stock_prices, prices, label=f'{option_type.capitalize()} Option')

    plt.title('Black-Scholes-Merton Option Prices')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    ticker_symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    stock_prices = get_stock_data(ticker_symbol, start_date, end_date)

    # Assuming the latest closing price is the current stock price
    current_stock_price = stock_prices[-1]

    # Parameters for the Black-Scholes-Merton model
    strike_price = 150  # Example strike price
    time_to_expiration = 30 / 365  # Example time to expiration in years
    risk_free_rate = 0.02  # Example risk-free interest rate
    volatility = 0.2  # Example volatility

    plot_option_prices(stock_prices, strike_price, time_to_expiration, risk_free_rate, volatility)
