import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("iris_data.csv")
print(df.head(), "\n")

x = df.drop("Species", axis=1)
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 8))
plot_tree(model, feature_names=x.columns, class_names=model.classes_, filled=True)
plt.show()

new_flower = pd.DataFrame([[5.3, 3.8, 4.0, 1.2]], columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"])

prediction = model.predict(new_flower)

print("Predicted species for the new flower:", prediction[0])

