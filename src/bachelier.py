import yfinance as yf
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def bachelier_model(S, X, T, r, sigma, option_type='call'):
    d1 = (S - X) / (sigma * np.sqrt(T))

    if option_type == 'call':
        option_price = np.exp(-r * T) * (S - X) * norm.cdf(d1) + (sigma ** 2 * T / 2) * np.exp(-r * T) * norm.pdf(d1)
    elif option_type == 'put':
        option_price = np.exp(-r * T) * (X - S) * norm.cdf(-d1) + (sigma ** 2 * T / 2) * np.exp(-r * T) * norm.pdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

def get_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return stock_data

def plot_option_prices(stock_prices, strike_price, time_to_expiration, risk_free_rate, volatility):
    option_types = ['call', 'put']
    option_prices = {}

    for option_type in option_types:
        option_prices[option_type] = []

        for current_stock_price in stock_prices:
            option_price = bachelier_model(current_stock_price, strike_price, time_to_expiration,
                                           risk_free_rate, volatility, option_type)
            option_prices[option_type].append(option_price)

    plt.figure(figsize=(10, 6))

    for option_type, prices in option_prices.items():
        plt.plot(stock_prices, prices, label=f'{option_type.capitalize()} Option')

    plt.title("Bachelier's Model Option Prices for Multiple Stocks")
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def simulate_galton_board(stock_symbols, num_balls, num_slots):
    positions = {}

    for symbol in stock_symbols:
        positions[symbol] = np.zeros(num_balls)

        for ball in range(num_balls):
            position = 0
            for _ in range(num_slots):
                # Randomly move left or right
                move = np.random.choice([-1, 1])
                position += move

            positions[symbol][ball] = position

    return positions

def plot_galton_board_simulation(positions, num_slots):
    plt.figure(figsize=(10, 6))

    for symbol, positions_array in positions.items():
        plt.hist(positions_array, bins=np.arange(-num_slots, num_slots + 1), align='left',
                 edgecolor='black', linewidth=1.2, alpha=0.5, label=symbol)

    plt.title('Simulation of Galton Board for Multiple Stocks')
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    ticker_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX', 'TSLA', 'IBM', 'INTC', 'CSCO',
                   'AAP', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE',
                   'AEP', 'AES', 'AFL', 'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN',
                   'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'NVDA', 'AMGN',
                   'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD',
                   'CL=F', 'DX-Y.NYB', 'BTC-USD', 'EURUSD=X',
                   'NG=F', '^TNX', 'GBPUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'EURJPY=X',
                   '^TYX', 'EURGBP=X', 'CADUSD=X', 'CHFUSD=X',
                   'TBT', 'GLL']
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    stock_prices = get_stock_data(ticker_symbols, start_date, end_date)

    # Assuming the latest closing prices are the current stock prices
    current_stock_prices = stock_prices.iloc[-1]

    # Parameters for the Bachelier's model
    strike_price = 150  # Example strike price
    time_to_expiration = 30 / 365  # Example time to expiration in years
    risk_free_rate = 0.02  # Example risk-free interest rate
    volatility = 0.2  # Example volatility

    # Plot Bachelier's model option prices
    plot_option_prices(current_stock_prices, strike_price, time_to_expiration, risk_free_rate, volatility)

    # Simulate and plot the Galton board
    num_balls = 1000
    num_slots = 10
    simulated_positions = simulate_galton_board(ticker_symbols, num_balls, num_slots)
    plot_galton_board_simulation(simulated_positions, num_slots)
"C:/Users/sanka/Desktop/Projects/Quantum/bdd/bdd100k/images/100k/test/cabc9045-1b8282ba.jpg"