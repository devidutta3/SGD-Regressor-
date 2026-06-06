import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3]])
y = np.array([12, 25, 40])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(np.array([[4]]))
print(prediction)