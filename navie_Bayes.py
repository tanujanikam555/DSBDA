# import libraries
import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('data6.csv')

# Prepare the features and target variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# Train the Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate and display evaluation metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix and evaluation metrics
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# Accept user input for prediction
age = float(input("Enter age: "))
salary = float(input("Enter salary: "))

# Reshape the input to a 2D array
input_data = [[age, salary]]  

# Make the prediction
prediction = classifier.predict(input_data)

print(f"The predicted class for age {age} and salary {salary} is: {prediction[0]}")