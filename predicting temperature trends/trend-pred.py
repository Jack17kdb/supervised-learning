import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("temperature_data.csv")
print(df.head(), "\n")

x = df[["Day"]]
y = df["Temperature"]

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

y_pred = model.predict(x_poly)

plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, y_pred, color="red", label="Polynomial Fit")
plt.title("Polynomial Regression - Temperature Trend")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

day_26 = np.array([[26]])
day_26_poly = poly.transform(day_26)
prediction = model.predict(day_26_poly)

print(f"\nPredicted temperature for day 26: {prediction[0]:.2f}°C")
