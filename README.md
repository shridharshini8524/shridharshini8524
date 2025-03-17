```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load data with error handling
try:
    data = pd.read_csv('crop_yield_data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: File 'crop_yield_data.csv' not found.")
    exit()

# Display first few rows of the dataset
print(data.head())

# Check for missing values
if data.isnull().sum().any():
    print("Warning: Missing values detected. Filling with column mean.")
    data.fillna(data.mean(), inplace=True)

# Verify column names
expected_columns = {'rainfall', 'temperature', 'humidity', 'yield'}
if not expected_columns.issubset(data.columns):
    print(f"Error: Expected columns {expected_columns}, but found {set(data.columns)}")
    exit()

# Define Features and Target
X = data[['rainfall', 'temperature', 'humidity']]
y = data['yield']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check data shape
print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {y_test.shape}")

# Create and Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```
