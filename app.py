from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form values
        age = float(request.form["age"])
        study = float(request.form["study"])
        social = float(request.form["social"])
        netflix = float(request.form["netflix"])
        attendance = float(request.form["attendance"])
        sleep = float(request.form["sleep"])
        exercise = float(request.form["exercise"])
        mental = float(request.form["mental"])

        # Convert to DataFrame (same format as training data)
        input_data = pd.DataFrame([[
            age,
            study,
            social,
            netflix,
            attendance,
            sleep,
            exercise,
            mental
        ]], columns=[
            "age",
            "study_hours_per_day",
            "social_media_hours",
            "netflix_hours",
            "attendance_percentage",
            "sleep_hours",
            "exercise_frequency",
            "mental_health_rating"
        ])

        # Make prediction
        result = model.predict(input_data)[0]

        # Round result
        prediction = round(result, 2)

        # Send to result page
        return render_template(
            "result.html",
            prediction=prediction
        )

    except Exception as e:

        # If error occurs, show on result page
        return render_template(
            "result.html",
            prediction="Error: " + str(e)
        )


# Run Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

