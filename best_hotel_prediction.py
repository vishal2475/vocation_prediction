import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Salary input
monthly_salary = 30000
max_affordable = monthly_salary * 0.25  # 25% of salary

# Step 2: Hotel data (you can extend or use CSV)
data = {
    "Hotel": ["Royal Palace", "Sunview Resort", "Budget Inn", "Ocean Pearl", "City Stay"],
    "Cost": [8000, 6000, 2500, 7200, 4000],
    "Rating": [4.5, 4.2, 3.8, 4.7, 4.0],
    "Feedback": [
        "Amazing service and peaceful environment",
        "Very nice rooms with good food",
        "Rooms were dirty and poorly maintained",
        "Excellent beach view and great hospitality",
        "Good value for money"
    ]
}

df = pd.DataFrame(data)

# Step 3: Train mini sentiment model
train_feedback = [
    "great service", "bad experience", "excellent stay", "dirty rooms",
    "friendly staff", "worst hotel", "amazing place", "not clean", "fantastic", "terrible"
]
train_labels = [
    "Positive", "Negative", "Positive", "Negative", "Positive",
    "Negative", "Positive", "Negative", "Positive", "Negative"
]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_feedback)
clf = LogisticRegression()
clf.fit(X_train, train_labels)

# Step 4: Predict feedback sentiment
X_test = vectorizer.transform(df["Feedback"])
df["Sentiment"] = clf.predict(X_test)

# Step 5: Apply filters
filtered = df[
    (df["Cost"] <= max_affordable) &
    (df["Rating"] >= 4.0) &
    (df["Sentiment"] == "Positive")
]

# Step 6: Recommend the best one
if filtered.empty:
    print(" No hotel matched the budget + rating + feedback criteria.")
else:
    best = filtered.sort_values(by="Rating", ascending=False).iloc[0]
    print("\n Best Hotel Recommendation:")
    print(f" Hotel   : {best.Hotel}")
    print(f" Cost    : ₹{best.Cost}")
    print(f" Rating  : {best.Rating}")
    print(f" Review  : {best.Feedback}")
    print(f" Sentiment: {best.Sentiment}")
