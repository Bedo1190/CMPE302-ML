import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Question 1
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
Y = np.array([[2.1], [2.9], [3.9], [5.2]])

def ridge_regression(X, Y, delta):
    I = np.eye(X.shape[1])
    W = np.linalg.inv(X.T @ X + delta**2 * I) @ X.T @ Y
    return W

W_ridge = ridge_regression(X, Y, delta=10)
print("Ridge Regression Weights for delta=10:", W_ridge.flatten())

delta_values = np.linspace(0.1, 50, 100)
weights = [ridge_regression(X, Y, d).flatten() for d in delta_values]
weights = np.array(weights)

plt.plot(delta_values, weights[:, 0], label='W1')
plt.plot(delta_values, weights[:, 1], label='W0')
plt.xlabel('Delta')
plt.ylabel('Weights')
plt.legend()
plt.title('Ridge Regression Weights vs Delta')
plt.show()

# Question 2
np.random.seed(0)
X_mle = np.linspace(0, 10, 50)
Y_mle = 2.5 * X_mle + np.random.normal(0, 2, 50)

W_mle = np.sum(X_mle * Y_mle) / np.sum(X_mle ** 2)
print("MLE Estimate for W:", W_mle)

plt.scatter(X_mle, Y_mle, label='Data')
plt.plot(X_mle, W_mle * X_mle, label='MLE Fit', color='red')
plt.axhline(y=2.5 * X_mle.mean(), color='green', linestyle='--', label='True Slope: 2.5')
plt.legend()
plt.title('MLE Linear Fit')
plt.show()

# Question 3
P = np.array([0.5, 0.5])
Q = np.array([0.3, 0.7])

entropy_P = -np.sum(P * np.log2(P))
cross_entropy = -np.sum(P * np.log2(Q))
kl_divergence = cross_entropy - entropy_P

print("Entropy H(P):", entropy_P)
print("Cross-Entropy CE(P, Q):", cross_entropy)
print("KL-Divergence KL(P||Q):", kl_divergence)

# Question 4
np.random.seed(0)
X_map = np.linspace(0, 5, 30)
Y_map = 3 * X_map + np.random.normal(0, 1, 30)

lambda_val = 1
W_map = np.sum(X_map * Y_map) / (np.sum(X_map ** 2) + lambda_val)
print("MAP Estimate for W (lambda=1):", W_map)

# Bonus
np.random.seed(0)
X_bias_var = np.linspace(0, 1, 100)
Y_bias_var = np.sin(2 * np.pi * X_bias_var) + np.random.normal(0, 0.1, 100)

plt.scatter(X_bias_var, Y_bias_var, label='Data')

for deg in [1, 3, 10]:
    coeffs = np.polyfit(X_bias_var, Y_bias_var, deg)
    poly_eq = np.poly1d(coeffs)
    plt.plot(X_bias_var, poly_eq(X_bias_var), label=f'Degree {deg}')

plt.legend()
plt.title('Bias-Variance Tradeoff')
plt.show()
