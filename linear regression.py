# Importing necessary libraries
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Boston housing dataset
data = pd.read_csv('data4.csv')
X = data.iloc[:, 1].values.reshape(-1, 1)  # Reshape X to 2D
y = data.iloc[:, -1].values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Create a Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Model Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print model parameters
print(f'Intercept: {model.intercept_}') 
print(f'Coefficient: {model.coef_[0]}') 
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-Squared Score (R²): {r2}')

# User input for price and predicting area
user_price = float(input("Enter the house price : "))
predicted_area = (model.intercept_ + model.coef_[0] * user_price)
print(f'Predicted area : {predicted_area}')