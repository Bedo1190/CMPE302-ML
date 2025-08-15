import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Problem 1
def simulate_conditional_probability(trials=10000):
    marbles = ['red'] * 6 + ['blue'] * 3 + ['green'] * 1
    A_and_B = 0
    B = 0
    for _ in range(trials):
        draw = np.random.choice(marbles)
        if draw != 'green':
            B += 1
            if draw == 'red':
                A_and_B += 1
    return A_and_B / B

p_a_given_b = simulate_conditional_probability()
print("P(A|B) ≈", p_a_given_b)

# Problem 2
X = np.array([[1], [2], [3], [4]])
Y = np.array([[2.1], [2.9], [3.9], [5.2]])
X_b = np.c_[X, np.ones((X.shape[0], 1))]  # Add bias
w_mle = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ Y
print("MLE Weights:", w_mle.flatten())

# Problem 3
beta = 1  
sigma2 = 1  
lambda_ = sigma2 / beta
w_map = np.linalg.inv(X_b.T @ X_b + lambda_ * np.eye(2)) @ X_b.T @ Y
print("MAP Weights:", w_map.flatten())

# Problem 4
residuals = Y - X_b @ w_mle
sigma2_est = np.mean(residuals**2)
print("Estimated Noise Variance σ²:", sigma2_est)

# Problem 5 & 6
def rbf_kernel(X, centers, gamma):
    return np.exp(-gamma * cdist(X, centers, "sqeuclidean"))

X_train = X.reshape(-1, 1)
Y_train = Y
centers = X_train

lambda_rbf = 0.1
gamma = 1.0  

Phi_train = rbf_kernel(X_train, centers, gamma)

w_rbf = np.linalg.inv(Phi_train.T @ Phi_train + lambda_rbf * np.eye(len(centers))) @ Phi_train.T @ Y_train

X_test = np.linspace(0, 5, 100).reshape(-1, 1)
Phi_test = rbf_kernel(X_test, centers, gamma)
Y_pred = Phi_test @ w_rbf

plt.scatter(X_train, Y_train, color='red', label='Training Data')
plt.plot(X_test, Y_pred, color='blue', label='RBF Prediction')
plt.title("RBF Regression")
plt.legend()
plt.show()
