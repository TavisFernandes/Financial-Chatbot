import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Real_Estate.csv')

# Encoding categorical variables if necessary
if 'Location' in df.columns:
    df['Location'] = LabelEncoder().fit_transform(df['Location'])
if 'Property Type' in df.columns:
    df['Property Type'] = LabelEncoder().fit_transform(df['Property Type'])

# Defining features and target variable
X = df[['Price per Sq Ft ($)', 'Rental Yield (%)', 'Appreciation Rate (%)']]
y = np.random.randint(0, 2, size=len(df))  # Placeholder binary target variable for classification

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Training and evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

# Regression Task
# Using Price per Sq Ft ($) as the target variable for regression
X_reg = df[['Rental Yield (%)', 'Appreciation Rate (%)']]
y_reg = df['Price per Sq Ft ($)']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

print("\nRegression Model: Random Forest Regressor")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
