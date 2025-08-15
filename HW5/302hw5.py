import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([[1.2], [2.3], [3.1], [4.5], [5.7]])
y = np.array([0, 0, 1, 1, 1])

# Add bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape: (5, 2)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Log-likelihood function
def log_likelihood(w, X, y):
    z = X.dot(w)
    return np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))

# Gradient of the log-likelihood
def gradient(w, X, y):
    z = X.dot(w)
    return X.T.dot(y - sigmoid(z))

# Gradient ascent
def train_logistic_regression(X, y, lr=0.1, n_iter=1000):
    w = np.zeros(X.shape[1])
    for i in range(n_iter):
        grad = gradient(w, X, y)
        w += lr * grad
    return w

# Train from scratch
weights = train_logistic_regression(X_b, y)
print("Weights (from scratch):", weights)

# Predict probability for x = 3.0
x_test = 3.0
x_test_b = np.array([1, x_test])
prob = sigmoid(x_test_b.dot(weights))
print(f"P(y=1 | x={x_test}) =", prob)

# Predict class
prediction = 1 if prob >= 0.5 else 0
print(f"Predicted class for x={x_test}:", prediction)

# Visualization
x_vals = np.linspace(0, 7, 100)
x_vals_b = np.c_[np.ones((100, 1)), x_vals.reshape(-1, 1)]
y_probs = sigmoid(x_vals_b.dot(weights))

plt.plot(x_vals, y_probs, label='Sigmoid Curve')
plt.scatter(X, y, color='red', label='Data Points')
plt.axvline(x=x_test, color='green', linestyle='--', label='Test Point')
plt.title('Logistic Regression (from scratch)')
plt.xlabel('Tumor Size')
plt.ylabel('P(y=1)')
plt.legend()
plt.grid(True)
plt.show()
