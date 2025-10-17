import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("car_data.csv")
print(df.head(), "\n")

x = df.drop("Price", axis=1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f"Predicted prices: {y_pred} \n")

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}\n")
print(f"R² Score: {r2}\n")

new_car = pd.DataFrame([[2025, 2.8, 210, 29]], columns=["Year", "EngineSize", "Horsepower", "MPG"])

predicted_price = model.predict(new_car)

print("Predicted Price for new car:", round(predicted_price[0], 2))

