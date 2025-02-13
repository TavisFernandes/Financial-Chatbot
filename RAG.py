import pandas as pd
from sentence_transformers import SentenceTransformer, util

def load_data():
    file_path = "Comprehensive_Banking_Database.csv"
    return pd.read_csv(file_path)

df = load_data()
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    
    return insights.strip()

def retrieve_relevant_response(query):
    questions = ["account info", "transactions", "loan status", "financial insights"]
    embeddings = model.encode(questions, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_match_idx = scores.argmax().item()
    return questions[best_match_idx]

def chatbot():
    customer_id = int(input("Enter Customer ID: "))
    if df[df['Customer ID'] == customer_id].empty:
        print("Customer not found.")
        return
    print("Welcome! You can ask about your account info, transactions, loan status, or financial insights.")
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
        else:
            print("I'm not sure how to answer that. Please ask about account info, transactions, loan status, or financial insights.")

if __name__ == "__main__":
    chatbot()
