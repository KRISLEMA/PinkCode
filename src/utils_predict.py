import joblib
import pandas as pd
from preprocessing import load_and_preprocess_data

# Load the trained model
model = joblib.load("C:/Users/us/Documents/PinkCode/models/breast_cancer_model.pkl")

# Function to make a prediction
def make_prediction(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    
    # Convert prediction to meaningful output
    result = "Malignant (Cancer Detected)" if prediction[0] == 1 else "Benign (No Cancer)"
    return result

# Example Test Input (Replace with real input values)
test_input = {
    "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0, 
    "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001, "concave points_mean": 0.1471, 
    "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871, "radius_se": 1.095, "texture_se": 0.9053, 
    "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399, "compactness_se": 0.04904, 
    "concavity_se": 0.05373, "concave points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0, 
    "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119, "concave points_worst": 0.2654, 
    "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
}

# Run prediction
result = make_prediction(test_input)
print("Prediction:", result)
