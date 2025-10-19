import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("email_data.csv")
print(df.head(), "\n")

x = df["Message"]
y = df["Label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer(stop_words="english")
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_email = ["You won a free vacation trip! Click here to claim."]
new_email_vec = vectorizer.transform(new_email)
prediction = model.predict(new_email_vec)
print("Prediction for new email:", prediction[0])

