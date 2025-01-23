import os
import time
import pandas as pd
import sqlite3
from transformers import pipeline

# Set environment variable to use CPU only (for TensorFlow issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load dataset
data_path = "C:/Users/admin/OneDrive/Desktop/Financial-Chatbot Datasets/financial_chatbot_dataset.csv"
df = pd.read_csv(data_path)

# Initialize NLP model (do this only once to avoid reloading on each request)
nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Database setup
conn = sqlite3.connect("budjet.db")
cursor = conn.cursor()

# Create tables for budgets, goals, and spending
cursor.execute("""
CREATE TABLE IF NOT EXISTS budgets (
    user_id INTEGER,
    budget_category TEXT,
    budget_amount REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS goals (
    user_id INTEGER,
    goal_target_amount REAL,
    goal_deadline TEXT,
    goal_progress REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS spending (
    user_id INTEGER,
    category TEXT,
    amount INTEGER
)
""")

# Insert data into SQLite database
cursor.executemany(
    "INSERT INTO budgets (user_id, budget_category, budget_amount) VALUES (?, ?, ?)",
    df[['user_id', 'budget_category', 'budget_amount']].dropna().values.tolist()
)

cursor.executemany(
    "INSERT INTO goals (user_id, goal_target_amount, goal_deadline, goal_progress) VALUES (?, ?, ?, ?)",
    df[['user_id', 'goal_target_amount', 'goal_deadline', 'goal_progress']].dropna().values.tolist()
)

spending_data = df.groupby(['user_id', 'category']).size().reset_index(name='amount')
cursor.executemany(
    "INSERT INTO spending (user_id, category, amount) VALUES (?, ?, ?)",
    spending_data.values.tolist()
)

conn.commit()

# Chatbot function
def chat():
    print("Welcome to your Financial Assistant Chatbot!")
    while True:
        user_input = input("You: ").strip().lower()
        if user_input in ["exit", "quit"]:
            print("Goodbye! Have a great day managing your finances.")
            break

        try:
            if "budget" in user_input:
                user_id = int(input("Enter your user ID: ").strip())
                cursor.execute("SELECT budget_category, budget_amount FROM budgets WHERE user_id = ?", (user_id,))
                budget_data = cursor.fetchall()
                if not budget_data:
                    print("No budget data found for the user.")
                else:
                    print("Here are your budgets:")
                    for category, amount in budget_data:
                        print(f"Category: {category}, Amount: {amount}")

            elif "goal" in user_input:
                user_id = int(input("Enter your user ID: ").strip())
                cursor.execute("SELECT goal_target_amount, goal_deadline, goal_progress FROM goals WHERE user_id = ?", (user_id,))
                goal_data = cursor.fetchall()
                if not goal_data:
                    print("No goal data found for the user.")
                else:
                    print("Here are your goals:")
                    for target, deadline, progress in goal_data:
                        print(f"Target: {target}, Deadline: {deadline}, Progress: {progress}%")

            elif "spend" in user_input:
                user_id = int(input("Enter your user ID: ").strip())
                cursor.execute("SELECT category, amount FROM spending WHERE user_id = ?", (user_id,))
                spending_data = cursor.fetchall()
                if not spending_data:
                    print("No spending data found for the user.")
                else:
                    print("Here's your spending by category:")
                    for category, amount in spending_data:
                        print(f"Category: {category}, Amount: {amount}")

            else:
                print("Let me think...")
                context = "User budget, goals, transactions, and spending data."
                answer = nlp(question=user_input, context=context)
                print(f"Bot: {answer['answer']}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

# Run chatbot
if __name__ == "__main__":
    chat()

# Close the database connection when done
conn.close()