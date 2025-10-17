import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("student_results.csv")
print(df.head(), "\n")

x = df.drop("Pass", axis=1)
y = df['Pass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

new_student = pd.DataFrame([[7, 40, 1]], columns=["Hours_Studied", "Attendance", "Assignments_Submitted"])
prediction = model.predict(new_student)

print("Predicted result for student (7 hrs, 40% attendance, 1 assignments):", "Pass" if prediction[0] == 1 else "Fail")

