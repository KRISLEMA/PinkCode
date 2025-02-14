import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns (modify as needed)
    if 'Unnamed: 32' in df.columns:
        df.drop(columns=['Unnamed: 32'], inplace=True)

    # Encode categorical labels (if needed)
    if 'diagnosis' in df.columns:
        label_encoder = LabelEncoder()
        df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])  # 0 = Benign, 1 = Malignant

    # Normalize numerical features
    scaler = StandardScaler()
    df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])  # Assuming first two columns are ID & label

    return df

# Test
if __name__ == "__main__":
    processed_df = load_and_preprocess_data("C:/Users/us/Documents/PinkCode/data/structured_data/data.csv")
    print(processed_df.head())
