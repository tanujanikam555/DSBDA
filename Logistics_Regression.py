import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('data5.csv')

# Drop 'Gender' column
df1 = df.drop("Genre", axis=1)

# Define features and target variable
y = df1.iloc[:, -1].values  # Target variable
X = df1.iloc[:, :-1].values  # Feature set

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# Train Logistic Regression model
LR = LogisticRegression()
LR.fit(X_train, y_train)

# Compute confusion matrix
y_pred = LR.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate metrics
accuracy = (tn + tp) * 100 / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)
specificity = tn / (tn + fp)

# Display metrics
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Specificity: {specificity:.2f}")

# Take user input for feature values
print("Enter values for the following features:")
input_features = []
for i in range(X_train.shape[1]):
    val = float(input(f"Feature {i+1}: "))
    input_features.append(val)
    
# Convert input into numpy array and reshape
input_array = np.array(input_features).reshape(1, -1)
    
# Make prediction
prediction = LR.predict(input_array)[0]
print(f"Predicted Output: {prediction}")