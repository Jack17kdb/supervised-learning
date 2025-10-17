import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("tumor_data.csv")
print(df.head(), "\n")

x = df.drop("Result", axis=1)
y = df['Result']

x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = SVC(kernel="linear", C=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_tumor = pd.DataFrame([[1.9, 0.71, 0.64]], columns=["TumorSize", "Smoothness", "Compactness"])
new_tumor = scaler.transform(new_tumor)
prediction = model.predict(new_tumor)

print("Prediction for new tumor:", prediction[0])
