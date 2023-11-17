from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScalar
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model and other necessary data
with open('optimized_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {}
    for feature in model.columns[:-1]:  # Assuming model columns are the features
        data[feature] = [request.form.get(feature)]

    # Preprocess the input data
    input_data = pd.DataFrame(data)
    input_data = pd.get_dummies(input_data)  # One-hot encode categorical features
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction_proba = model.predict_proba(input_data_scaled)[:, 1]
    churn_confidence = round(prediction_proba[0] * 100, 2)

    # Return the result as JSON
    result = {
        'Churn Prediction Confidence': churn_confidence
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
