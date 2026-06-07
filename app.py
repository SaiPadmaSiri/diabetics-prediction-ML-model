from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return jsonify({
        "status": "API Running Successfully"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [[
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["bloodpressure"]),
            float(data["skinthickness"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["dpf"]),
            float(data["age"])
        ]]

        prediction = model.predict(features)

        result = (
            "Diabetes Detected"
            if prediction[0] == 1
            else "No Diabetes"
        )

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
