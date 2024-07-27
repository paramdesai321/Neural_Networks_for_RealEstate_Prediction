import numpy as np
import random
import matplotlib.pyplot as plt


# Generate random data
x = np.random.rand(10)
y = np.random.rand(10)

# Plot the initial data
plt.scatter(x, y)

# Initialize weights and bias
w = np.random.rand(1)
b = random.uniform(-100, 100)


# Linear regression function
def f(x, w, b):
    return x * w + b


# Learning rate
k = 0.1

# Gradient Descent
for i in range(400):
    y_pred = f(x, w, b)
    loss = ((y_pred - y) ** 2).mean()

    w_grad = 2 * ((y_pred - y) * x).mean()
    b_grad = 2 * (y_pred - y).mean()

    w -= k * w_grad
    b -= k * b_grad

# Final prediction
y_pred = f(x, w, b)

# Plot the results
plt.scatter(x, y)
plt.plot(x, f(x, w, b), c='r')
plt.show()

# Final loss
loss = ((y_pred - y) ** 2).mean()
print('Final loss:', loss)
print('Predicted values:', y_pred)
print("Target values:", y)
