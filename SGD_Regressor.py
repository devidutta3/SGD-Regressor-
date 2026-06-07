from sklearn.linear_model import LinearRegression
# Dataset
X = [1, 2, 3]
Y = [50, 55, 60]

# Initial Parameters
m = 1
b = 45

# Hyperparameter
learning_rate = 0.01

# Number of Iterations
epochs = 10

for epoch in range(epochs):

    # Predictions
    predictions = []
    for x in X:
        predictions.append(m * x + b)

    # Errors
    errors = []
    for actual, predicted in zip(Y, predictions):
        errors.append(actual - predicted)

    # MSE Cost
    mse = sum(error ** 2 for error in errors) / len(errors)

    # Gradient of m
    gradient_m = -(2 / len(X)) * sum(
        x * error for x, error in zip(X, errors)
    )

    # Gradient of b
    gradient_b = -(2 / len(X)) * sum(errors)

    # Gradient Descent Update
    m = m - learning_rate * gradient_m
    b = b - learning_rate * gradient_b

    print(
        f"Epoch {epoch+1}: "
        f"Cost={mse:.2f}, "
        f"m={m:.4f}, "
        f"b={b:.4f}"
    )

print("\nFinal Parameters")
print(f"m = {m:.4f}")
print(f"b = {b:.4f}")