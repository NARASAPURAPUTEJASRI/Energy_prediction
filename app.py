from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib  # Assuming your model is saved with joblib
from sklearn.preprocessing import OneHotEncoder
import os

print("Current working directory:", os.getcwd())

app = Flask(__name__)

# Initialize and load the model
model = joblib.load('plastic_model.pkl')

# Initialize OneHotEncoder for Plastic_Type (if necessary)
plastic_types = ["ABS", "HDPE", "LDPE", "PC", "PET", "PMMA", "PP", "PS", "PUR", "PVC"]
encoder = OneHotEncoder(categories='auto')
encoder.fit([[plastic_type] for plastic_type in plastic_types])

@app.route('/')
def home():
    return render_template('home.html')  # Same as before

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        # Extract data from the incoming request (same logic retained)
        temperature = float(data["temperature"])
        pressure = float(data["pressure"])
        reaction_time = float(data["reaction_time"])
        reactor_type = int(data["reactor_type"])
        catalyst_conc = float(data["catalyst_conc"])
        plastic_type_index = int(data["plastic_type"])

        # One-hot encode the plastic type
        one_hot_vector = [0] * len(plastic_types)  # Same as before
        one_hot_vector[plastic_type_index] = 1

        # Prepare the input data for the model (same as before)
        input_data = np.array([[temperature, pressure, reaction_time, reactor_type, catalyst_conc] + one_hot_vector])

        # Make the prediction using the loaded model (same as before)
        predicted_energy = model.predict(input_data)[0]

        # Return the predicted energy as a JSON response
        return jsonify({"predicted_energy": predicted_energy})

    return render_template('index.html')  # Added this line to return 'index.html' for GET requests

if __name__ == '__main__':
    app.run(debug=True)

