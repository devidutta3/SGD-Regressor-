from sklearn.linear_model import LinearRegression

hours = [
    [1],
    [2],
    [3],
]

X = hours

marks = [
    12,
    25,
    40
]

Y = marks

model = LinearRegression()
model.fit(X, Y)

prediction = model.predict([[4]])
print(prediction)