import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("BankNote_Authentication.csv")

X, y = df.iloc[:, :-1], df.iloc[:, -1]

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)
y=y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create and train the MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(6),  # Two hidden layers with 100 and 50 neurons
    activation='relu',  # ReLU activation function
    solver='sgd',  # sgd optimizer
    alpha=0.0001,  # L2 penalty (regularization term)
    batch_size=1,
    learning_rate='constant',
    learning_rate_init=0.003,
    max_iter=800,  # Maximum number of iterations
    early_stopping=True,  # Use early stopping
    validation_fraction=0.1,  # Fraction of training data for validation
    n_iter_no_change=10,  # Number of iterations with no improvement
    random_state=42  # Random seed for reproducibility
)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
predicts = mlp.predict(X_test)

y_true = y_test.flatten()
y_pred = predicts.flatten()
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))