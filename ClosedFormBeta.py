import numpy as np

# Define the feature matrix X and target vector y
X = np.array([[1, 1],
              [1, 2],
              [1, 3]])

y = np.array([[3],
              [5],
              [7]])

# Compute X^T X
XTX = np.dot(X.T, X)

# Compute X^T y
XTy = np.dot(X.T, y)

# Compute (X^T X)^-1
XTX_inverse = np.linalg.inv(XTX)

# Compute beta
beta = np.dot(XTX_inverse, XTy)

# Extract beta_0 and beta_1
beta_0 = beta[0][0]
beta_1 = beta[1][0]

print("beta_0:", beta_0)
print("beta_1:", beta_1)