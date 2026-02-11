import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset (tab separated WITH header)
df = pd.read_csv("student_habits_performance.csv", sep="\t")

print("Dataset Loaded Successfully")
print(df.head())
print(df.dtypes)


# Convert numeric columns properly
numeric_cols = [
    "age",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating",
    "exam_score"
]

# Convert to numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Drop missing values
df = df.dropna()


print("After Cleaning:")
print(df[numeric_cols].head())


# Features & Target
X = df[[
    "age",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating"
]]

y = df["exam_score"]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


print(" Model Trained and Saved Successfully!")
