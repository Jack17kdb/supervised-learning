import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("house_prices.csv")
print(df.head(), "\n")

x = df.drop("Price", axis=1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict(x_test)

mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f"mean_squared_error: {mse}\nr2_score: {r2}\n")

new_house = pd.DataFrame([[2000, 3, 7]], columns=["SquareFeet", "Bedrooms", "Age"])
predicted_price = model.predict(new_house)

print(f"Predicted price: {predicted_price[0]}\n")
