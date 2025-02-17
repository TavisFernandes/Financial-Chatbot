import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def load_data():
    stock_market_df = pd.read_csv("/content/Stock_Market.csv")
    real_estate_df = pd.read_csv("/content/Real_Estate.csv")
    crypto_df = pd.read_csv("/content/Cryptocurrencies.csv")
    return stock_market_df, real_estate_df, crypto_df

def preprocess_data(df):
    df = df.dropna()
    if df.select_dtypes(include=['object']).shape[1] > 0:
        df = pd.get_dummies(df, drop_first=True)
    return df

def get_best_stock(stock_market_df, amount, duration):
    best_stock = stock_market_df.loc[stock_market_df["1Y Return (%)"].idxmax()]
    expected_return = amount * ((1 + best_stock['1Y Return (%)'] / 100) ** duration)
    return (f"Best stock to invest in: {best_stock['Company']} with {best_stock['1Y Return (%)']}% return.\n"
            f"Expected value after {duration} years: ${expected_return:.2f}")

def get_best_real_estate(real_estate_df, amount, duration):
    best_location = real_estate_df.loc[real_estate_df["Rental Yield (%)"].idxmax()]
    best_appreciation = real_estate_df.loc[real_estate_df["Appreciation Rate (%)"].idxmax()]
    expected_appreciation = amount * ((1 + best_appreciation['Appreciation Rate (%)'] / 100) ** duration)
    return (f"Best location for rental yield: {best_location['Location']} with {best_location['Rental Yield (%)']}% yield.\n"
            f"Best location for appreciation: {best_appreciation['Location']} with {best_appreciation['Appreciation Rate (%)']}% appreciation.\n"
            f"Expected value after {duration} years: ${expected_appreciation:.2f}")

def get_best_crypto(crypto_df, amount, duration):
    best_crypto = crypto_df.loc[crypto_df["Market Cap (B)"].idxmax()]
    expected_return = amount * ((1 + best_crypto['Sentiment Score'] / 100) ** duration)
    return (f"Best cryptocurrency to invest in: {best_crypto['Company']} with {best_crypto['Market Cap (B)']}B market cap.\n"
            f"Expected value after {duration} years: ${expected_return:.2f}")

def recommend_investment():
    stock_market_df, real_estate_df, crypto_df = load_data()
    
    amount = float(input("Enter investment amount: "))
    duration = int(input("Enter investment duration (in years): "))
    sector = input("Choose sector (stocks, real estate, crypto): ").strip().lower()
    
    if sector == "stocks":
        return get_best_stock(stock_market_df, amount, duration)
    elif sector == "real estate":
        return get_best_real_estate(real_estate_df, amount, duration)
    elif sector == "crypto":
        return get_best_crypto(crypto_df, amount, duration)
    else:
        return "Sector not recognized. Please choose from 'stocks', 'real estate', or 'crypto'."

# Run the interactive investment recommendation
print(recommend_investment())
