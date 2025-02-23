import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import numpy as np

real_estate_df = pd.read_csv("Real_Estate.csv")
crypto_df = pd.read_csv("Cryptocurrencies.csv")
green_df = pd.read_csv("Green_Investments.csv")
bank_faq_df = pd.read_csv("BankFAQs.csv")
def load_data():
    banking_file = "Comprehensive_Banking_Database.csv"
    stocks_file = "stocks_data.csv"
    historical_prices_file = "historical_prices.csv"
    return pd.read_csv(banking_file), pd.read_csv(stocks_file), pd.read_csv(historical_prices_file)

df, stocks_df, historical_prices_df = load_data()
model = SentenceTransformer('all-MiniLM-L6-v2')

def answer_customer_query(query):
    best_match = bank_faq_df.loc[bank_faq_df['Question'].str.contains(query, case=False, na=False)]
    if best_match.empty:
        return "I'm sorry, I couldn't find an answer to your question."
    return best_match.iloc[0]['Answer']

def get_account_info(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    if customer.empty:
        return "Customer not found."
    info = customer[['First Name', 'Last Name', 'Account Type', 'Account Balance']].to_dict(orient='records')[0]
    return f"Hello {info['First Name']} {info['Last Name']}, your account type is {info['Account Type']} and your current balance is ${info['Account Balance']:.2f}."

def get_transaction_history(customer_id):
    transactions = df[df['Customer ID'] == customer_id][['Transaction Date', 'Transaction Type', 'Transaction Amount']]
    if transactions.empty:
        return "No transactions found."
    result = "Here is your transaction history:\n"
    for _, row in transactions.iterrows():
        result += f"On {row['Transaction Date']}, you made a {row['Transaction Type']} of ${row['Transaction Amount']:.2f}.\n"
    return result.strip()

def get_loan_status(customer_id):
    loan = df[df['Customer ID'] == customer_id][['Loan Type', 'Loan Amount', 'Loan Status']]
    if loan.empty:
        return "No loan records found."
    result = "Here is your loan status:\n"
    for _, row in loan.iterrows():
        result += f"You have a {row['Loan Type']} of ${row['Loan Amount']:.2f}, and the status is {row['Loan Status']}.\n"
    return result.strip()

def get_financial_insights(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    if customer.empty:
        return "Customer not found."
    balance = customer['Account Balance'].values[0]
    transactions = df[df['Customer ID'] == customer_id][['Transaction Type', 'Transaction Amount']]
    total_spent = transactions[transactions['Transaction Type'] == 'Withdrawal']['Transaction Amount'].sum()
    total_deposited = transactions[transactions['Transaction Type'] == 'Deposit']['Transaction Amount'].sum()

    insights = f"Financial Insights for Customer {customer_id}:\n"
    insights += f"Your current balance is ${balance:.2f}.\n"
    insights += f"Total deposited: ${total_deposited:.2f}.\n"
    insights += f"Total spent: ${total_spent:.2f}.\n"

    if total_spent > total_deposited:
        insights += "You are spending more than you deposit. Consider adjusting your budget.\n"
    elif balance < 500:
        insights += "Your balance is low. Consider adding more funds to avoid overdrafts.\n"
    else:
        insights += "Your financial health looks stable. Keep up the good management!\n"

    labels = ['Deposited', 'Spent', 'Balance']
    values = [total_deposited, total_spent, balance]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['green', 'red', 'blue'])
    plt.xlabel("Financial Aspects")
    plt.ylabel("Amount in USD")
    plt.title(f"Financial Overview for Customer {customer_id}")
    plt.show()

    return insights.strip()

def get_stock_info(stock_symbol):
    stock = stocks_df[stocks_df['Symbol'] == stock_symbol]
    if stock.empty:
        return "Stock symbol not found."
    info = stock.iloc[0]
    return f"Stock: {info['Symbol']} - {info['CompanyName']}\nIndustry: {info['Industry']}\nMarket Cap: ${info['MarketCap']:.2f}"

def get_current_stock_price(stock_symbol):
    stock_data = historical_prices_df[historical_prices_df['StockID'] == stock_symbol]
    if stock_data.empty:
        return "Stock symbol not found."
    latest_data = stock_data.sort_values('Date', ascending=False).iloc[0]
    return f"The latest closing price for {stock_symbol} is ${latest_data['ClosePrice']:.2f}."

def predict_stock_price(stock_symbol):
    stock_data = historical_prices_df[historical_prices_df['StockID'] == stock_symbol]
    if stock_data.empty:
        return "Stock symbol not found."

    stock_data = stock_data.sort_values('Date')
    stock_data['Days'] = np.arange(len(stock_data))
    X = stock_data[['Days']]
    y = stock_data['ClosePrice']

    model = LinearRegression()
    model.fit(X, y)

    future_day = np.array([[len(stock_data) + 1]])
    predicted_price = model.predict(future_day)[0]

    return f"The predicted stock price for {stock_symbol} for the next trading day is ${predicted_price:.2f}."

def get_investment_recommendations(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    if customer.empty:
        return "Customer not found."
    balance = customer['Account Balance'].values[0]
    recommendations = "Investment Recommendations:\n"
    if balance > 10000:
        recommendations += "Consider investing in high-yield stocks like TSLA or AAPL.\n"
    else:
        recommendations += "You may want to start with safer index funds or ETFs.\n"
    return recommendations.strip()

def retrieve_relevant_response(query):
    questions = ["account info", "transactions", "loan status", "financial insights", "stock prediction", "investment recommendations", "stock info", "current stock price", "fraud detection", "credit score", "expense prediction", "loan approval", "feedback sentiment", "real estate trends", "cryptocurrency trends", "green investments", "real estate", "cryptocurrency", "query", "tax compliance", "rewards"]
    embeddings = model.encode(questions, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_match_idx = scores.argmax().item()
    return questions[best_match_idx]

# K-Means for Fraud Detection
from sklearn.cluster import KMeans
def train_kmeans_fraud_model():
    if 'Transaction Amount' not in df.columns or 'Anomaly' not in df.columns:
        return None
    df_clean = df.dropna(subset=['Transaction Amount', 'Anomaly'])
    X = df_clean[['Transaction Amount']]
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    return model
kmeans_fraud_model = train_kmeans_fraud_model()

def detect_anomalies(transaction_amount):
    if kmeans_fraud_model is None:
        return "Fraud detection model unavailable."
    cluster = kmeans_fraud_model.predict([[transaction_amount]])[0]
    return "Anomalous transaction detected." if cluster == -1 else "No anomalies detected."

#Predict credit score using KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def train_credit_score_model():
    X = df[['Account Balance', 'Transaction Amount', 'Credit Limit', 'Credit Card Balance']]
    y = df['Rewards Points']  # Proxy for credit score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

credit_score_model = train_credit_score_model()

def predict_credit_score(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    if customer.empty:
        return "Customer not found."

    features = customer[['Account Balance', 'Transaction Amount', 'Credit Limit', 'Credit Card Balance']]
    score = credit_score_model.predict(features)[0]
    return f"Your estimated credit score is **{score:.2f}** based on your financial history."


from sklearn.linear_model import LinearRegression

#Expense prediction using Multiple Linear Regression
def train_expense_model():
    X = df[['Transaction Amount', 'Account Balance']]
    y = df['Transaction Amount'].shift(-1).fillna(df['Transaction Amount'].mean())  # Predict next transaction

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

expense_model = train_expense_model()

def predict_expense(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    if customer.empty:
        return "Customer not found."

    features = customer[['Transaction Amount', 'Account Balance']]
    future_expense = expense_model.predict(features)[0]
    return f"Your predicted next expense is **${future_expense:.2f}** based on past transactions."


#Loan approval model using decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_loan_approval_model():
    df['Loan Status'] = LabelEncoder().fit_transform(df['Loan Status'])  # Encode Approved/Rejected
    X = df[['Loan Amount', 'Interest Rate', 'Loan Term', 'Account Balance']]
    y = df['Loan Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

loan_model = train_loan_approval_model()

def predict_loan_approval(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    if customer.empty:
        return "Customer not found."

    features = customer[['Loan Amount', 'Interest Rate', 'Loan Term', 'Account Balance']]
    prediction = loan_model.predict(features)
    return "Your loan is likely to be **Approved**" if prediction[0] == 1 else "Your loan may be **Rejected**. Consider improving your financial profile."


def analyze_feedback_sentiment():
    df['Sentiment'] = df['Feedback Type'].apply(lambda x: TextBlob(x).sentiment.polarity)
    avg_sentiment = df['Sentiment'].mean()
    sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
    return f"Overall customer feedback sentiment: {sentiment_label} ({avg_sentiment:.2f})"

def predict_real_estate_trend():
    X = real_estate_df[['Price per Sq Ft ($)', 'Rental Yield (%)']]
    y = real_estate_df['Appreciation Rate (%)']
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X[:1])[0]
    return f"Predicted real estate market trend score: {prediction:.2f}"

def predict_crypto_trend():
    X = crypto_df[['Market Cap (B)', 'Volatility (%)']]
    y = crypto_df['Price ($)']
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X[:1])[0]
    return f"Predicted cryptocurrency price change: {prediction:.2f}%"

def recommend_green_investments():
    top_green = green_df.sort_values(by='ESG Rating', ascending=False).head(3)
    return f"Top Green Investments: {top_green[['Company', 'Revenue Growth (%)']].to_string(index=False)}"

from sklearn.preprocessing import PolynomialFeatures

#Real Estate Investment using polynomial regression
def train_real_estate_model():
    X = real_estate_df[['Price per Sq Ft ($)', 'Rental Yield (%)']]
    y = real_estate_df['Appreciation Rate (%)']

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, poly

real_estate_model, poly_transform = train_real_estate_model()

def recommend_real_estate(amount, return_rate, risk_tolerance):
    X_new = poly_transform.transform(real_estate_df[['Price per Sq Ft ($)', 'Rental Yield (%)']])
    predicted_returns = real_estate_model.predict(X_new)

    real_estate_df['Predicted Appreciation'] = predicted_returns
    suitable_investments = real_estate_df[(real_estate_df['Predicted Appreciation'] >= return_rate) &
                                          (real_estate_df['Price per Sq Ft ($)'] * 100 <= amount)]

    if suitable_investments.empty:
        return "No suitable real estate investments found."

    return suitable_investments[['Location', 'Property Type', 'Price per Sq Ft ($)', 'Predicted Appreciation']]


from sklearn.linear_model import SGDRegressor

#Cryptocurrency Investment using Gradient Descent
def train_crypto_model():
    X = crypto_df[['Price ($)', 'Market Cap (B)']]
    y = crypto_df['Price ($)'].shift(-1).fillna(crypto_df['Price ($)'].mean())  # Predict next price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    return model

crypto_model = train_crypto_model()

def recommend_crypto(amount, return_rate, risk_tolerance):
    predicted_prices = crypto_model.predict(crypto_df[['Price ($)', 'Market Cap (B)']])

    crypto_df['Predicted Price'] = predicted_prices
    suitable_cryptos = crypto_df[(crypto_df['Predicted Price'] / crypto_df['Price ($)'] - 1 >= return_rate / 100) &
                                 (crypto_df['Market Cap (B)'] >= risk_tolerance * 10)]

    if suitable_cryptos.empty:
        return "No suitable cryptocurrency investments found."

    return suitable_cryptos[['Name', 'Price ($)', 'Predicted Price', 'Market Cap (B)']]

def answer_customer_query(query):
    best_match = bank_faq_df.loc[bank_faq_df['Question'].str.contains(query, case=False, na=False)]
    if best_match.empty:
        return "I'm not sure about that, but I can try to help! Can you provide more details?"
    return best_match.iloc[0]['Answer']

#Tax compliance and gamified customer reward using random forest and linear regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_tax_compliance_model():
    if 'Account Balance' not in df.columns or 'Transaction Amount' not in df.columns:
        return None

    df_clean = df.dropna(subset=['Account Balance', 'Transaction Amount'])
    
    X = df_clean[['Account Balance', 'Transaction Amount']]
    y = df_clean['Account Balance'] * 0.15  # Assuming 15% estimated tax liability

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

tax_model = train_tax_compliance_model()

def provide_tax_compliance_assistance(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    
    if customer.empty:
        return "Customer ID not found. Please check and try again."
    
    if tax_model is None:
        return "Tax model unavailable at the moment. Please try later."

    X_test = customer[['Account Balance', 'Transaction Amount']].fillna(0)  # Fill missing values with 0
    predicted_tax = tax_model.predict(X_test)[0]

    return f"Based on your financials, your estimated tax liability is ${predicted_tax:.2f}. Consider consulting a tax expert for tax-saving strategies."

#Gamified customer reward using linear regression
def train_rewards_model():
    if 'Transaction Amount' not in df.columns or 'Rewards Points' not in df.columns:
        return None

    df_clean = df.dropna(subset=['Transaction Amount', 'Rewards Points'])
    X = df_clean[['Transaction Amount']]
    y = df_clean['Rewards Points']

    model = LinearRegression()
    model.fit(X, y)
    return model

rewards_model = train_rewards_model()

def reward_customer_activity(customer_id):
    customer = df[df['Customer ID'] == customer_id]
    if customer.empty or rewards_model is None:
        return "Customer not found or rewards model unavailable."

    total_spent = customer['Transaction Amount'].sum()
    predicted_rewards = rewards_model.predict([[total_spent]])[0]

    return f"Based on your spending habits, you may earn approximately {predicted_rewards:.0f} reward points. Keep transacting to maximize rewards!"

def chatbot():
    customer_id = int(input("Enter Customer ID: "))
    if df[df['Customer ID'] == customer_id].empty:
        print("Customer not found.")
        return
    print("Welcome! You can ask about your account info, transactions, loan status, financial insights, stock prediction, investment recommendations, stock info, or current stock price.")
    while True:
        user_input = input("Ask a question (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        best_match = retrieve_relevant_response(user_input)
        if best_match == "account info":
            print(get_account_info(customer_id))
        elif best_match == "transactions":
            print(get_transaction_history(customer_id))
        elif best_match == "loan status":
            print(get_loan_status(customer_id))
        elif best_match == "financial insights":
            print(get_financial_insights(customer_id))
        elif best_match == "stock prediction":
            stock_symbol = input("Enter stock symbol: ")
            print(predict_stock_price(stock_symbol))
        elif best_match == "investment recommendations":
            print(get_investment_recommendations(customer_id))
        elif best_match == "stock info":
            stock_symbol = input("Enter stock symbol: ")
            print(get_stock_info(stock_symbol))
        elif best_match == "current stock price":
            stock_symbol = input("Enter stock symbol: ")
            print(get_current_stock_price(stock_symbol))
        elif best_match == "fraud detection":
            print(detect_anomalies(customer_id))
        elif best_match == "credit score":
            print(predict_credit_score(customer_id))
        elif best_match == "expense prediction":
            print(predict_expense(customer_id))
        elif best_match == "loan approval":
            print(predict_loan_approval(customer_id))
        elif "feedback sentiment" in user_input:
            print(analyze_feedback_sentiment())
        elif "real estate trends" in user_input:
            print(predict_real_estate_trend())
        elif "cryptocurrency trends" in user_input:
            print(predict_crypto_trend())
        elif "green investments" in user_input:
            print(recommend_green_investments())
        elif "query" in user_input:
            query = input("Enter your query: ")
            print(answer_customer_query(query))
        elif "real estate" in user_input:
            investment = float(input("Enter investment amount: "))
            return_pct = float(input("Enter desired annual return percentage: "))
            risk = float(input("Enter risk tolerance percentage: "))
            print(recommend_real_estate(investment, return_pct, risk))
        elif "cryptocurrency" in user_input:
            investment = float(input("Enter investment amount: "))
            return_pct = float(input("Enter desired annual return percentage: "))
            risk = float(input("Enter risk tolerance percentage: "))
            print(recommend_crypto(investment, return_pct, risk))
        elif "tax compliance" in user_input:
            print(provide_tax_compliance_assistance(customer_id))
        elif "rewards" in user_input:
            print(reward_customer_activity(customer_id))
        else:
            print("Feature not yet implemented.")

if __name__ == "__main__":
    chatbot()
