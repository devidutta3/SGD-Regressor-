import numpy as np

X = np.array([1,2,3])
Y = np.array([50,55,60])

m = 1
b = 45

epochs = 10
lr = 0.01

for epoch in range(epochs):

    prediction = m * X + b

    error = Y - prediction

    mse = np.mean(error**2)

    derivative_m = -2 * np.sum(X * error) / len(X)
    derivative_b = -2 * np.sum(error) / len(X)

    m = m - lr * derivative_m
    b = b - lr * derivative_b

    print(f"Epoch {epoch+1}")
    print(f"Prediction: {prediction}")
    print(f"MSE: {mse:.4f}")
    print(f"m = {m:.4f}, b = {b:.4f}")
    print()