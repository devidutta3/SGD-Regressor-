from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = [
    [1],
    [2],
    [3]
]

Y = [
    12,
    25,
    40
]

model = LinearRegression()
model.fit(X, Y)

predictions = model.predict(X)

mae = mean_absolute_error(Y, predictions)

print("Predictions:", predictions)
print("MAE:", mae)