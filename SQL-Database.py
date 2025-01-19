import sqlite3
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Initialize the SQLite database
DATABASE = 'financial_chatbot.db'

# Function to initialize database
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS transactions (
                        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        category TEXT,
                        amount REAL,
                        date TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(user_id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS goals (
                        goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        target_amount REAL,
                        current_amount REAL,
                        deadline TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(user_id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS budget (
                        budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        category TEXT,
                        budget_amount REAL,
                        FOREIGN KEY(user_id) REFERENCES users(user_id))''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Function to get user by name
def get_user_by_name(name):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE name=?', (name,))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to add user
def add_user(name):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
    conn.commit()
    conn.close()

# Function to add transaction
def add_transaction(user_id, category, amount, date):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO transactions (user_id, category, amount, date) VALUES (?, ?, ?, ?)', 
                   (user_id, category, amount, date))
    conn.commit()
    conn.close()

# Function to get transaction insights
def get_transaction_summary(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT category, SUM(amount) FROM transactions WHERE user_id=? GROUP BY category', (user_id,))
    summary = cursor.fetchall()
    conn.close()
    return summary

# Function to add budget
def add_budget(user_id, category, budget_amount):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO budget (user_id, category, budget_amount) VALUES (?, ?, ?)', 
                   (user_id, category, budget_amount))
    conn.commit()
    conn.close()

# Function to get user budget data
def get_user_budget(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT category, budget_amount FROM budget WHERE user_id=?', (user_id,))
    budget_data = cursor.fetchall()
    conn.close()
    return budget_data

# Function to get user goals
def get_user_goals(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT target_amount, current_amount, deadline FROM goals WHERE user_id=?', (user_id,))
    goals = cursor.fetchall()
    conn.close()
    return goals

# Function to add financial goals
def add_goal(user_id, target_amount, current_amount, deadline):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO goals (user_id, target_amount, current_amount, deadline) VALUES (?, ?, ?, ?)', 
                   (user_id, target_amount, current_amount, deadline))
    conn.commit()
    conn.close()

# Function to suggest if user should take loan
def loan_suggestion(income, expense):
    if income < expense:
        return "It's not advisable to take a loan right now."
    else:
        return "You could consider taking a loan if necessary, but make sure to manage your finances well."

@app.route('/register', methods=['POST'])
def register_user():
    try:
        name = request.json.get("name")
        user = get_user_by_name(name)
        if not user:
            add_user(name)
            return jsonify({"message": f"User {name} registered successfully!"})
        else:
            return jsonify({"message": f"User {name} already exists."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/add_transaction', methods=['POST'])
def add_transaction_route():
    try:
        name = request.json.get("name")
        category = request.json.get("category")
        amount = request.json.get("amount")
        date = request.json.get("date")
        
        user = get_user_by_name(name)
        if user:
            user_id = user[0]
            add_transaction(user_id, category, amount, date)
            return jsonify({"message": "Transaction added successfully!"})
        else:
            return jsonify({"message": "User not found."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/set_budget', methods=['POST'])
def set_budget():
    try:
        name = request.json.get("name")
        category = request.json.get("category")
        budget_amount = request.json.get("budget_amount")
        
        user = get_user_by_name(name)
        if user:
            user_id = user[0]
            add_budget(user_id, category, budget_amount)
            return jsonify({"message": "Budget set successfully!"})
        else:
            return jsonify({"message": "User not found."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_summary', methods=['POST'])
def get_summary():
    try:
        name = request.json.get("name")
        user = get_user_by_name(name)
        if user:
            user_id = user[0]
            spending_summary = get_transaction_summary(user_id)
            budget_data = get_user_budget(user_id)
            return jsonify({"spending_summary": spending_summary, "budget_data": budget_data})
        else:
            return jsonify({"message": "User not found."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_goals', methods=['POST'])
def get_goals():
    try:
        name = request.json.get("name")
        user = get_user_by_name(name)
        if user:
            user_id = user[0]
            goals = get_user_goals(user_id)
            return jsonify({"goals": goals})
        else:
            return jsonify({"message": "User not found."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/loan_suggestion', methods=['POST'])
def loan_suggestion_route():
    try:
        name = request.json.get("name")
        income = request.json.get("income")
        expense = request.json.get("expense")
        
        user = get_user_by_name(name)
        if user:
            suggestion = loan_suggestion(income, expense)
            return jsonify({"suggestion": suggestion})
        else:
            return jsonify({"message": "User not found."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/stock_assistant', methods=['POST'])
def stock_assistant():
    # Placeholder for stock assistant functionality
    return jsonify({"message": "Stock Assistant feature coming soon!"})

if __name__ == '__main__':
    app.run(debug=True)