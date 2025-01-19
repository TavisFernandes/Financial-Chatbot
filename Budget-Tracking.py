import os
import time
from flask import Flask, request, jsonify
import pandas as pd
from transformers import pipeline

# Set environment variable to use CPU only (for TensorFlow issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Load dataset
data_path = "C:/Users/admin/OneDrive/Desktop/Financial-Chatbot Datasets/financial_chatbot_dataset.csv"
df = pd.read_csv(data_path)

# Initialize NLP model (do this only once to avoid reloading on each request)
nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Store preprocessed data for faster access
user_budget_data = {}
user_goal_data = {}
user_spending_data = {}

# Preprocess data and store in memory for quick access
def preprocess_data():
    global user_budget_data, user_goal_data, user_spending_data
    # Preprocess budget data
    user_budget_data = df.groupby('user_id')[['budget_category', 'budget_amount']].apply(lambda x: x.to_dict(orient="records")).to_dict()
    # Preprocess goal data
    user_goal_data = df.groupby('user_id')[['goal_target_amount', 'goal_deadline', 'goal_progress']].apply(lambda x: x.to_dict(orient="records")).to_dict()
    # Preprocess spending data
    user_spending_data = df.groupby('user_id')['category'].apply(lambda x: x.value_counts().to_dict()).to_dict()

# Run preprocessing once during app startup
preprocess_data()

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()  # Track execution time for performance monitoring
    try:
        # Get user input
        user_input = request.json.get("message", "").lower()
        user_id = request.json.get("user_id", 1)  # Default user ID for testing
        
        # Process user queries
        if "budget" in user_input:
            budget_data = user_budget_data.get(user_id, [])
            if not budget_data:
                return jsonify({"response": "No budget data found for the user."})
            response = budget_data
            return jsonify({"response": "Here are your budgets:", "data": response})
        
        elif "goal" in user_input:
            goal_data = user_goal_data.get(user_id, [])
            if not goal_data:
                return jsonify({"response": "No goal data found for the user."})
            response = goal_data
            return jsonify({"response": "Here are your goals:", "data": response})
        
        elif "spend" in user_input:
            spending_data = user_spending_data.get(user_id, {})
            if not spending_data:
                return jsonify({"response": "No spending data found for the user."})
            response = spending_data
            return jsonify({"response": "Here's your spending by category:", "data": response})
        
        else:
            # Use NLP model for generic questions
            context = "User budget, goals, transactions, and spending data."
            answer = nlp(question=user_input, context=context)
            return jsonify({"response": answer['answer']})

    except Exception as e:
        return jsonify({"error": str(e)})
    
    finally:
        # Track execution time
        execution_time = time.time() - start_time
        print(f"Request executed in {execution_time:.2f} seconds")

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=False, threaded=True)




