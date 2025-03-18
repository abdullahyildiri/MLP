
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("BankNote_Authentication.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



def initialize_parameters(n_x, n_h1, n_h2, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h1, n_x) *  np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) *np.sqrt(2.0 / n_h1)
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) *  np.sqrt(2.0 / n_h2)
    b3 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return parameters


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(A):
    return (A > 0).astype(float)


def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)  # ReLU Kullanıldı
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)  # ReLU Kullanıldı
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache



def compute_cost(A3, Y):
    m =A3.shape[1]
    cost = - (np.dot(np.log(A3), Y) + np.dot(np.log(1 - A3), (1 - Y))) / m
    cost = float(np.squeeze(cost))
    return cost


def backpropagation(X, Y, cache, parameters):
    m = X.shape[0]
    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]

    dZ3 = A3 - Y.T
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = np.dot(W3.T, dZ3) * relu_derivative(A2)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(A1)
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    return grads



# Update parameters
def update_parameters(parameters, grads, learning_rate=0.03):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    paramaters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    return parameters


# Train model
def nn_model(X, Y, n_x, n_h1, n_h2, n_y, n_steps=1000):
    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    for i in range(n_steps):
        A3, cache = forward_propagation(X, parameters)
        cost = compute_cost(A3, Y)
        grads = backpropagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate=0.01)

    return parameters


# Train and test
parameters = nn_model(X_train, y_train, X_train.shape[1], 5, 5, 1, n_steps=1000)


def predict(parameters, X):
    A3, cache= forward_propagation(X, parameters)
    return (A3 > 0.5).astype(int)

predicts = predict(parameters, X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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


def perform_grid_search():
    parameters_n_h = [i for i in range(3, 11)]
    parameters_n_steps = [i for i in range(100, 1100, 100)]
    results = []

    for n_h in parameters_n_h:
        for n_step in parameters_n_steps:
            parameters=  nn_model(X_train, y_train, X_train.shape[1], n_h1=n_h, n_h2=n_h, n_y=1, n_steps=n_step)
            y_pred = predict(parameters, X_test).flatten()
            acc = accuracy_score(y_test.flatten(), y_pred)
            results.append((n_h, n_step, acc))
            print(f"n_h: {n_h}, n_step: {n_step}, acc: {acc:.4f}")

    best_result = max(results, key=lambda x: x[2])
    print("\nBest model configuration:")
    print(f"n_h: {best_result[0]}, n_step: {best_result[1]}, accuracy: {best_result[2]:.4f}")

    return best_result

perform_grid_search()













