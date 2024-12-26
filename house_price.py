import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
file_path = 'content/BostonHousing (1).csv'
data = pd.read_csv(file_path)


print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
data.info()

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())


print("\nSummary Statistics:\n", data.describe())


plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Define Features and Target

X = data.drop(columns=["medv"], errors='ignore') 
y = data["medv"] if "medv" in data.columns else None 

# Check if target column exists:-
if y is None:
    print("The dataset does not contain a 'MEDV' column. Please specify the target variable.")
else:
    # Split Data into Training and Testing Sets:-
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model:-
    model = LinearRegression()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Coefficients
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })
    print("\nModel Coefficients:\n", coefficients)

    #print
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title("Predicted vs Actual Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.show()
