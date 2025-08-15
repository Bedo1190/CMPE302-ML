import numpy as np
import matplotlib.pyplot as plt

# Question 1
Phi = np.array([[1, 10, 4], [1, 8, 25], [1, 5, 40], [1, 2, 38]])
Y = np.array([[9], [32], [61], [70]])

W = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ Y
print("Weight Vector W:", W.flatten())

# Question 2
X_poly = np.array([0, 1, 2, 3])
Y_poly = np.array([-6, -4, -20, 0])

coeffs = np.polyfit(X_poly, Y_poly, 3)
poly_eq = np.poly1d(coeffs)
print("Polynomial Equation:", poly_eq)

x_range = np.linspace(min(X_poly), max(X_poly), 100)
y_range = poly_eq(x_range)
plt.scatter(X_poly, Y_poly, color='red', label='Data Points')
plt.plot(x_range, y_range, label='Polynomial Fit', color='blue')
plt.legend()
plt.title("Polynomial Regression (Degree 3)")
plt.show()

# Question 3
rainfall = np.array([1, 2, 3])
probabilities = np.array([0.2, 0.7, 0.1])
expected_value = np.sum(rainfall * probabilities)
print("Expected Rainfall Value:", expected_value)

# Question 4
X_stock = np.array([1245, 1415, 1312, 1427, 1510, 1590])
Y_stock = np.array([100, 123, 129, 143, 150, 197])

cov_matrix = np.cov(X_stock, Y_stock, bias=True)
cov_value = cov_matrix[0, 1]
print("Covariance between X and Y:", cov_value)
