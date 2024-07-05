import yfinance as yf
import pandas as pd
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None

# Function to calculate daily and monthly returns
def calculate_returns(stock_data):
    try:
        stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
        stock_data['Monthly_Return'] = stock_data['Adj Close'].resample('M').ffill().pct_change()
        return stock_data
    except Exception as e:
        print(f"Error calculating returns: {e}")
        return None

# Function to create a matrix with daily and monthly returns
def create_returns_matrix(stock_data):
    if stock_data is not None:
        returns_matrix = stock_data[['Daily_Return', 'Monthly_Return']].dropna()
        return returns_matrix
    else:
        return None

# Estimate beta using tensor contraction
def estimate_beta(asset_returns, market_returns):
    if asset_returns is not None and market_returns is not None:
        # Convert returns to tensors and transpose to ensure compatibility
        asset_tensor = qtn.Tensor(asset_returns.values.reshape(1, -1), inds=['i', 'j'])
        market_tensor = qtn.Tensor(market_returns.values.reshape(1, -1), inds=['j', 'k'])

        # Perform tensor contraction
        cov = qtn.tensor_contract(asset_tensor, market_tensor, optimize='auto-hq')

        beta = cov / np.var(market_returns)
        return beta.data[0, 0]  # Extract the numerical value of beta
    else:
        return None
# Calculate expected return using CAPM
def capm(r_f, beta, r_m):
    if beta is not None:
        return r_f + beta * (r_m - r_f)
    else:
        return None

ticker_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX', 'TSLA', 'IBM', 'INTC', 'CSCO',
        'AAP', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE',
        'AEP', 'AES', 'AFL', 'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN',
        'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'NVDA', 'AMGN', 'AMP', 'AMT', 'ANET', 'ANSS',
        'AON', 'AOS', 'APA', 'APD', 'CL=F', 'DX-Y.NYB', 'BTC-USD', 'EURUSD=X',
        'NG=F', '^TNX', 'GBPUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'EURJPY=X',
        '^TYX', 'EURGBP=X', 'CADUSD=X', 'CHFUSD=X',
        'TBT', 'GLL',
        'ARE', 'ADM', 'ANF', 'ADI',
        'BABA', 'BAC', 'BA', 'BBY', 'BDX', 'BIIB', 'BK', 'BLK', 'BMY', 'BRK-B',
        'BSX', 'BUD', 'C', 'CAT', 'CB', 'CCI', 'CCL', 'CDNS', 'CF',
        'CHTR', 'CI', 'CL', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF',
        'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTVA',
        'CVS', 'CVX', 'DAL', 'DE', 'DFS', 'DG', 'DHI', 'DHR', 'DIS'
        , 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK',
        'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'EOG',
        'EQIX', 'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD',
        'EXPE', 'EXR', 'F', 'FAST', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS',
        'FITB', 'FLS', 'FMC', 'FOX', 'FOXA', 'FRT',  'TCS.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'INFY.NS', 'KOTAKBANK.NS',
        'BHARTIARTL.NS', 'ICICIBANK.NS', 'WIPRO.NS', 'LT.NS', 'HCLTECH.NS', 'AXISBANK.NS',
        'MARUTI.NS', 'ONGC.NS', 'COALINDIA.NS', 'NTPC.NS', 'POWERGRID.NS', 'SUNPHARMA.NS',
        'ITC.NS', 'M&M.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'BAJAJ-AUTO.NS', 'SBIN.NS', 'BAJFINANCE.NS',
        'JSWSTEEL.NS', 'GAIL.NS', 'HEROMOTOCO.NS', 'BPCL.NS', 'CIPLA.NS', 'INDUSINDBK.NS',
        'RECLTD.NS', 'DRREDDY.NS', 'ONGC.NS', 'IOC.NS', 'BAJAJFINSV.NS', 'SHREECEM.NS', 'WIPRO.NS', 'TECHM.NS',
        'GRASIM.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'VEDL.NS', 'LT.NS', 'INFY.NS',
        'KOTAKBANK.NS', 'HINDUNILVR.NS', 'TCS.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'ITC.NS', 'BAJFINANCE.NS', 'SBIN.NS',
        'ICICIBANK.NS', 'WIPRO.NS', 'SUNPHARMA.NS', 'M&M.NS', 'ONGC.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'AXISBANK.NS',
        'CIPLA.NS', 'SHREECEM.NS', 'BAJAJ-AUTO.NS', 'JSWSTEEL.NS', 'HEROMOTOCO.NS', 'GAIL.NS', 'DRREDDY.NS', 'IOC.NS', 'RECLTD.NS',
        'TECHM.NS', 'GRASIM.NS', 'VEDL.NS', 'TITAN.NS']


start_date = '2010-01-01'
end_date = '2024-02-25'

# Assuming we have market data available
market_ticker = '^GSPC'  # S&P 500 index ticker
market_data = get_stock_data(market_ticker, start_date, end_date)
market_returns = calculate_returns(market_data)['Daily_Return'].dropna()

# Create an empty DataFrame to store the results
results_data = []
for ticker_symbol in ticker_symbols:
    # Get historical stock data
    stock_data = get_stock_data(ticker_symbol, start_date, end_date)

    if stock_data is not None:
        # Calculate daily and monthly returns
        stock_data = calculate_returns(stock_data)

        # Create a matrix with daily and monthly returns
        returns_matrix = create_returns_matrix(stock_data)

        # Estimate beta using tensor contraction
        beta = estimate_beta(returns_matrix['Daily_Return'], market_returns)

        if beta is not None:
            # Assuming risk-free rate is 2%
            risk_free_rate = 0.02

            # Calculate expected market return
            expected_market_return = np.mean(market_returns)

            # Calculate expected return using CAPM
            expected_return = capm(risk_free_rate, beta, expected_market_return)

            if expected_return is not None:
                # Append results to the list
                results_data.append({
                    'Ticker': ticker_symbol,
                    'Beta': beta,
                    'Expected Return': expected_return * 100  # Convert to percentage
                })
            else:
                print(f"Error calculating expected return for {ticker_symbol}.")
        else:
            print(f"Error estimating beta for {ticker_symbol}.")
    else:
        print(f"Skipping {ticker_symbol} due to data retrieval error.")

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_data)

# Display the results table
print(results_df)
# Number of Monte Carlo simulations
num_simulations = 100

# Perform Monte Carlo simulation for market returns
monte_carlo_market_returns = np.random.normal(np.mean(market_returns), np.std(market_returns), (len(market_returns), num_simulations))

# Create an empty DataFrame to store the Monte Carlo simulation results
monte_carlo_results_data = []

for ticker_symbol in ticker_symbols:
    # Get historical stock data
    stock_data = get_stock_data(ticker_symbol, start_date, end_date)

    if stock_data is not None:
        # Calculate daily and monthly returns
        stock_data = calculate_returns(stock_data)

        # Create a matrix with daily and monthly returns
        returns_matrix = create_returns_matrix(stock_data)

        # Estimate beta using tensor contraction
        beta = estimate_beta(returns_matrix['Daily_Return'], market_returns)

        if beta is not None:
            # Assuming risk-free rate is 2%

            # Monte Carlo simulation for expected market return
            monte_carlo_expected_market_returns = np.random.normal(np.mean(market_returns), np.std(market_returns), num_simulations)

            # Monte Carlo simulation for expected return using CAPM
            monte_carlo_expected_returns = capm(risk_free_rate, beta, monte_carlo_expected_market_returns)

            # Append average results to the list
            monte_carlo_results_data.append({
                'Ticker': ticker_symbol,
                'Beta': beta,
                'Average Expected Return': np.mean(monte_carlo_expected_returns) * 100  # Convert to percentage
            })
        else:
            print(f"Error estimating beta for {ticker_symbol}.")
    else:
        print(f"Skipping {ticker_symbol} due to data retrieval error.")

# Create a DataFrame from the Monte Carlo simulation results list
monte_carlo_results_df = pd.DataFrame(monte_carlo_results_data)

# Display the Monte Carlo simulation results table
print(monte_carlo_results_df)

plt.figure(figsize=(12, 8))

# Scatter plot for Beta
plt.subplot(2, 1, 1)
plt.scatter(monte_carlo_results_df['Ticker'], monte_carlo_results_df['Beta'], c='r')  # Change color to red
plt.title('Beta for Each Ticker (Monte Carlo Simulation)')
plt.xlabel('Ticker')
plt.ylabel('Beta')

# Bar plot for Average Expected Return
plt.subplot(2, 1, 2)
plt.bar(monte_carlo_results_df['Ticker'], monte_carlo_results_df['Average Expected Return'])
plt.title('Average Expected Return for Each Ticker (Monte Carlo Simulation)')
plt.xlabel('Ticker')
plt.ylabel('Average Expected Return (%)')

plt.tight_layout()
plt.show()


