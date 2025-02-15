from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidney Beans", 5: "Pigeon Peas",
    6: "Moth Beans", 7: "Mung Bean", 8: "Black Gram", 9: "Lentil", 10: "Pomegranate",
    11: "Banana", 12: "Mango", 13: "Grapes", 14: "Watermelon", 15: "Muskmelon",
    16: "Apple", 17: "Orange", 18: "Papaya", 19: "Coconut", 20: "Cotton",
    21: "Jute", 22: "Coffee"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values
    features = [float(request.form[key]) for key in ['N', 'P', 'K', 'temp', 'hum', 'ph', 'rain']]
    
    # Scale features
    transformed_features = scaler.transform([features])

    # Predict crop
    prediction = model.predict(transformed_features)[0]
    crop_name = crop_dict.get(prediction, "Unknown Crop")

    return render_template('result.html', crop=crop_name)

if __name__ == "__main__":
    app.run(debug=True)
