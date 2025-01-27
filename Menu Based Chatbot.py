import sqlite3

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

# Function to add financial goals
def add_goal(user_id, target_amount, current_amount, deadline):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO goals (user_id, target_amount, current_amount, deadline) VALUES (?, ?, ?, ?)', 
                   (user_id, target_amount, current_amount, deadline))
    conn.commit()
    conn.close()

# Function to loan suggestion
def loan_suggestion(income, expense):
    if income < expense:
        return "It's not advisable to take a loan right now."
    else:
        return "You could consider taking a loan if necessary, but manage your finances well."

# Command-line interface
def main():
    while True:
        print("\nFinancial Chatbot Menu:")
        print("1. Register User")
        print("2. Add Transaction")
        print("3. Set Budget")
        print("4. Get Spending Summary")
        print("5. Add Financial Goal")
        print("6. Get Loan Suggestion")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter your name: ")
            if not get_user_by_name(name):
                add_user(name)
                print(f"User {name} registered successfully!")
            else:
                print(f"User {name} already exists.")

        elif choice == "2":
            name = input("Enter your name: ")
            user = get_user_by_name(name)
            if user:
                category = input("Enter category: ")
                amount = float(input("Enter amount: "))
                date = input("Enter date (YYYY-MM-DD): ")
                add_transaction(user[0], category, amount, date)
                print("Transaction added successfully!")
            else:
                print("User not found.")

        elif choice == "3":
            name = input("Enter your name: ")
            user = get_user_by_name(name)
            if user:
                category = input("Enter category: ")
                budget_amount = float(input("Enter budget amount: "))
                add_budget(user[0], category, budget_amount)
                print("Budget set successfully!")
            else:
                print("User not found.")

        elif choice == "4":
            name = input("Enter your name: ")
            user = get_user_by_name(name)
            if user:
                summary = get_transaction_summary(user[0])
                for category, total in summary:
                    print(f"Category: {category}, Total: {total}")
            else:
                print("User not found.")

        elif choice == "5":
            name = input("Enter your name: ")
            user = get_user_by_name(name)
            if user:
                target = float(input("Enter target amount: "))
                current = float(input("Enter current amount: "))
                deadline = input("Enter deadline (YYYY-MM-DD): ")
                add_goal(user[0], target, current, deadline)
                print("Goal added successfully!")
            else:
                print("User not found.")

        elif choice == "6":
            income = float(input("Enter income: "))
            expense = float(input("Enter expenses: "))
            print(loan_suggestion(income, expense))

        elif choice == "7":
            print("Exiting chatbot. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

