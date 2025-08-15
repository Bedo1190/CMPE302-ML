import numpy as np
import matplotlib.pyplot as plt

# Problem 1
X = np.array([[1], [2], [3], [4], [5]])
Y = np.array([[2.2], [2.8], [3.6], [4.5], [5.1]])

X_b = np.c_[np.ones((X.shape[0], 1)), X]

theta_closed = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
print("Closed-Form Solution Weights:", theta_closed.flatten())

# Problem 2
def gradient_descent(X, Y, alpha=0.01, iterations=1000):
    m = len(Y)
    theta = np.zeros((X.shape[1], 1))
    for _ in range(iterations):
        gradients = (2/m) * X.T.dot(X.dot(theta) - Y)
        theta -= alpha * gradients
    return theta

X_gd = np.c_[np.ones((X.shape[0], 1)), X]
theta_gd = gradient_descent(X_gd, Y)
print("Gradient Descent Weights:", theta_gd.flatten())

# Problem 3
X_multi = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
Y_multi = np.array([[2.2], [2.9], [3.7], [4.6], [5.2]])

X_multi_b = np.c_[np.ones((X_multi.shape[0], 1)), X_multi]

theta_multi = np.linalg.pinv(X_multi_b.T.dot(X_multi_b)).dot(X_multi_b.T).dot(Y_multi)
print("Multivariate Linear Regression Weights:", theta_multi.flatten())

# Problem 4
theta_multi_gd = gradient_descent(X_multi_b, Y_multi)
print("Gradient Descent Weights (Multivariate):", theta_multi_gd.flatten())

# Problem 5
X_outlier = np.array([[1], [2], [3], [4], [5], [10]])
Y_outlier = np.array([[2.2], [2.8], [3.6], [4.5], [5.1], [15]])

X_outlier_b = np.c_[np.ones((X_outlier.shape[0], 1)), X_outlier]

theta_outlier = np.linalg.pinv(X_outlier_b.T.dot(X_outlier_b)).dot(X_outlier_b.T).dot(Y_outlier)

plt.scatter(X_outlier, Y_outlier, color='red', label='Data')
plt.plot(X_outlier, X_outlier_b.dot(theta_outlier), label='Regression Line', color='blue')
plt.legend()
plt.title("Effect of Outliers on Linear Regression")
plt.show()
