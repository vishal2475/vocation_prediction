import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


file_path = r"C:\Users\C3STREAMLAND\Downloads\vacation_data.csv"
df = pd.read_csv(file_path)


train_feedbacks = [
    "very good place", "excellent service", "bad experience",
    "worst food", "beautiful location", "not clean",
    "amazing view", "friendly staff", "terrible hotel", "not worth the money"
]
train_labels = [
    "Positive", "Positive", "Negative",
    "Negative", "Positive", "Negative",
    "Positive", "Positive", "Negative", "Negative"
]


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_feedbacks)

model = LogisticRegression()
model.fit(X_train, train_labels)

X_test = vectorizer.transform(df["Feedback"])
df["Sentiment"] = model.predict(X_test)


filtered = df[
    (df["Cost"] <= 3000) &
    (df["Rating"] >= 4.0) &
    (df["Travel_Time"] <= 10) &
    (df["Sentiment"] == "Positive")
]


if filtered.empty:
    print(
        "No destination satisfies the current criteria. "
        "Try adding more data or relaxing the filters."
    )
else:
    best = filtered.sort_values(by="Rating", ascending=False).iloc[0]
    print("\n   BEST VACATION DESTINATION")
    print(f" Destination  : {best.Place}")
    print(f" Cost         : {best.Cost}")
    print(f" Rating       : {best.Rating}")
    print(f" Sentiment    : {best.Sentiment}")
    print(f" Travel Time  : {best.Travel_Time} hours")
    print(f" Feedback     : {best.Feedback}\n")


df.to_csv("vacation_data_with_sentiment.csv", index=False)
