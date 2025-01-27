import pandas as pd
import mysql.connector
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import faker

# Initialize Faker instance to generate fake data
fake = faker.Faker()

# DBMS - Database Configuration
DB_CONFIG = {
    'user': 'your_username',
    'password': 'your_password',
    'host': 'your_host',  # e.g., 'localhost' or cloud MySQL host
    'port': 3306,
    'database': 'FinancialAssistantDB'
}

# ML - Initialize Linear Regression Model
model = LinearRegression()

def connect_to_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def insert_data_to_db():
    # Load CSV files
    stocks_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/Stock/stocks_data.csv')
    users_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/Stock/users_data.csv')
    price_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/Stock/historical_prices.csv')

    # Insert Stock Data
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        for _, row in stocks_df.iterrows():
            cursor.execute("""
                INSERT INTO Stocks (Symbol, CompanyName, MarketCap, Sector, Industry)
                VALUES (%s, %s, %s, %s, %s)
            """, (row['Symbol'], row['CompanyName'], row['MarketCap'], row['Sector'], row['Industry']))
        conn.commit()

        # Insert User Data
        for _, row in users_df.iterrows():
            cursor.execute("""
                INSERT INTO Users (Username, Email, PasswordHash, Preferences, PortfolioValue)
                VALUES (%s, %s, %s, %s, %s)
            """, (row['Username'], row['Email'], row['PasswordHash'], row['Preferences'], row['PortfolioValue']))
        conn.commit()

        # Insert Price Data
        for _, row in price_df.iterrows():
            stock_id = stocks_df[stocks_df['Symbol'] == row['StockID']].index[0] + 1  # Assuming stock IDs start at 1
            cursor.execute("""
                INSERT INTO HistoricalPrices (StockID, Date, OpenPrice, ClosePrice, HighPrice, LowPrice, Volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (stock_id, row['Date'], row['OpenPrice'], row['ClosePrice'], row['HighPrice'], row['LowPrice'], row['Volume']))
        conn.commit()

        cursor.close()
        conn.close()
        print("Data inserted successfully.")

# Uncomment the line below to insert the data into the database
insert_data_to_db()

def train_ml_model():
    # Load the historical prices data
    price_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/Stock/historical_prices.csv')
    stocks_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/Stock/stocks_data.csv')
    # Prepare the data
    feature_columns = ['OpenPrice', 'HighPrice', 'LowPrice', 'Volume']  # Features
    target_column = 'ClosePrice'  # Target to predict

    # Merge dataframes for stock and price details
    merged_df = pd.merge(price_df, stocks_df, left_on='StockID', right_on='Symbol', how='left')

    # Prepare feature matrix X and target vector y
    X = merged_df[feature_columns]
    y = merged_df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")

    # Visualize the predictions
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Predicted vs Actual Prices')
    plt.show()

# Uncomment below to train and evaluate the ML model
train_ml_model()

def fetch_stock_data(symbol):
    # Fetch stock data from MySQL for a given symbol
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Stocks WHERE Symbol = %s", (symbol,))
        stock = cursor.fetchone()
        conn.close()
        if stock:
            return f"Stock Info for {stock[1]}: MarketCap - ${stock[2]}, Sector - {stock[3]}, Industry - {stock[4]}"
        else:
            return f"Sorry, stock with symbol {symbol} not found."

def predict_stock(symbol):
    # Predict stock price using ML model for a given symbol
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM HistoricalPrices WHERE StockID = (SELECT StockID FROM Stocks WHERE Symbol = %s)", (symbol,))
        price_data = cursor.fetchall()

        if not price_data:
            return f"No historical data found for {symbol}."

        # Prepare data for prediction (last 5 rows for simplicity)
        recent_data = price_data[-5:]
        features = np.array([[x[2], x[4], x[5], x[6]] for x in recent_data])  # Open, High, Low, Volume
        predicted_price = model.predict(features)

        return f"Predicted closing price for {symbol}: ${predicted_price[-1]:.2f}"

def chatbot():
    print("Welcome to the Financial and Stock Assistant Chatbot!")

    while True:
        user_input = input("You: ").lower()

        if 'stock info' in user_input:
            symbol = input("Enter the stock symbol: ")
            print(fetch_stock_data(symbol))
        elif 'predict price' in user_input:
            symbol = input("Enter the stock symbol: ")
            print(predict_stock(symbol))
        elif 'exit' in user_input:
            print("Goodbye!")
            break
        else:
            print("Sorry, I didn't understand that. Try 'stock info' or 'predict price'.")

# Uncomment to start the chatbot
chatbot()
