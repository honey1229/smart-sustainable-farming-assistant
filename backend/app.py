from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to trained crop recommendation model
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.pkl")

# Load the trained model
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return jsonify({
        "message": "Smart Sustainable Farming Assistant API is running!"
    })


@app.route("/predict/crop", methods=["POST"])
def predict_crop():

    try:
        # Get JSON data from request
        data = request.get_json()

        # Create DataFrame with the same feature order
        input_data = pd.DataFrame([{
            "N": data["N"],
            "P": data["P"],
            "K": data["K"],
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "ph": data["ph"],
            "rainfall": data["rainfall"]
        }])

        # Make prediction
        prediction = model.predict(input_data)

        # Return prediction
        return jsonify({
            "recommended_crop": prediction[0]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)