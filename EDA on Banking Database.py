import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'C:/Users/admin/OneDrive/Desktop/Financial-Chatbot Datasets/Comprehensive_Banking_Database.csv'  # Replace with your file path
data = pd.read_csv(file_path)

date_columns = [
    'Date Of Account Opening', 'Last Transaction Date', 'Transaction Date', 
    'Approval/Rejection Date', 'Payment Due Date', 'Last Credit Card Payment Date', 
    'Feedback Date', 'Resolution Date'
]
data[date_columns] = data[date_columns].apply(pd.to_datetime, errors='coerce')

financial_metrics = ['Account Balance', 'Transaction Amount', 'Loan Amount', 'Credit Limit', 'Rewards Points']
data[financial_metrics] = data[financial_metrics].apply(pd.to_numeric, errors='coerce')

cleaned_data = data[financial_metrics].dropna()

numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
fig, axes = plt.subplots(len(numerical_columns)//4 + 1, 4, figsize=(20, 15))
axes = axes.flatten()
for i, col in enumerate(numerical_columns):
    sns.histplot(data[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

sns.pairplot(cleaned_data, diag_kind='kde', corner=True)
plt.show()

correlation = cleaned_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap - Cleaned Data')
plt.show()

print(data.describe())

missing_values = data.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nValue counts for {col}:")
    print(data[col].value_counts())
