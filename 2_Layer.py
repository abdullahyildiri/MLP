import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

df = pd.read_csv("BankNote_Authentication.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def initialize_parameters(n_x, n_h, n_y=1):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X.T) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def compute_cost(A2, Y):
    m = A2.shape[1]
    cost = -np.sum(Y.T * np.log(A2) + (1 - Y.T) * np.log(1 - A2)) / m
    cost = float(np.squeeze(cost))
    return cost


def backpropagation(X, Y, cache, parameters):
    m = X.shape[0]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y.T
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_parameters(parameters, grads, learning_rate=0.01):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def nn_model(X, Y, n_x, n_h, n_y=1, n_steps=1000, print_cost=True, learning_rate=0.01):
    parameters = initialize_parameters(n_x, n_h, n_y)

    costs = []
    for i in range(n_steps):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backpropagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost:.4f}")

    return parameters, costs


def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5).astype(int)


parameters, costs = nn_model(X_train, y_train, X_train.shape[1], n_h=6, n_y=1, n_steps=1000)

y_pred = predict(parameters, X_test)
y_true = y_test.flatten()
y_pred = y_pred.flatten()

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
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
            parameters, cache= nn_model(X_train, y_train, X_train.shape[1], n_h=n_h, n_y=1, n_steps=n_step, print_cost=False)
            y_pred = predict(parameters, X_test).flatten()
            acc = accuracy_score(y_test.flatten(), y_pred)
            results.append((n_h, n_step, acc))
            print(f"n_h: {n_h}, n_step: {n_step}, acc: {acc:.4f}")

    best_result = max(results, key=lambda x: x[2])
    print("\nBest model configuration:")
    print(f"n_h: {best_result[0]}, n_step: {best_result[1]}, accuracy: {best_result[2]:.4f}")

    return best_result

perform_grid_search()
