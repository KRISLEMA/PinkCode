import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Load the dataset
df = pd.read_csv("C:/Users/us/Documents/PinkCode/data/structured_data/data.csv")

# Drop unnecessary columns
if 'Unnamed: 32' in df.columns:
    df.drop(columns=['Unnamed: 32'], inplace=True)

# Encode categorical labels
if 'diagnosis' in df.columns:
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])  # 0 = Benign, 1 = Malignant

# Normalize numerical features
scaler = StandardScaler()
df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])  # Assuming first two columns are ID & label

# Define features and labels
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_path = "C:/Users/us/Documents/PinkCode/models/breast_cancer_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create models folder if it doesn't exist
joblib.dump(model, model_path)

print("✅ Model trained and saved successfully at:", model_path)
