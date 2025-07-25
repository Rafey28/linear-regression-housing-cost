import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # House size in 1000 sqft
y = np.array([100, 150, 200, 250, 300])  # Price in $1000s

# Cost function
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    return cost / (2 * m)

# Gradient function
def compute_gradient(x, y, w, b):
    m = len(x)
    dw = 0
    db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dw += (f_wb - y[i]) * x[i]
        db += (f_wb - y[i])
    return dw / m, db / m

# Gradient descent
def gradient_descent(x, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dw, db = compute_gradient(x, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost={cost:.2f}, w={w:.2f}, b={b:.2f}")
    return w, b

# Prediction function
def predict(x, w, b):
    return w * x + b

# Run model
w_final, b_final = gradient_descent(x, y, w=0, b=0, alpha=0.01, num_iters=1000)
print(f"Final Model: y = {w_final:.2f}x + {b_final:.2f}")
price = predict(6, w_final, b_final)
print(f"Predicted price for 6k sqft: ${price:.2f}k")

# Plot
plt.scatter(x, y, label='Actual Data')
plt.plot(x, predict(x, w_final, b_final), color='red', label='Prediction')
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price ($1000s)")
plt.legend()
plt.show()
