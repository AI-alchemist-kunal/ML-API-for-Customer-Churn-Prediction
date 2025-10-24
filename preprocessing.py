import pandas as pd

def clean_data(df):
    """Cleans the raw Telco Churn DataFrame."""
    df_clean = df.copy()
    df_clean = df_clean.drop("customerID", axis=1, errors='ignore') # ignore if already dropped
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].mean())
    no_internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in no_internet_cols:
        if col in df_clean.columns:
             df_clean[col] = df_clean[col].replace('No internet service', 'No')
    df_clean['Churn'] = df_clean['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    print("Data cleaning complete.")
    return df_clean

def encode_features(df):
    """One-hot encodes categorical features, expects 'Churn' column to exist."""
    # Ensure Churn column exists before dropping it temporarily
    if 'Churn' not in df.columns:
        raise ValueError("DataFrame must include 'Churn' column for encoding.")
        
    target = df['Churn']
    features = df.drop('Churn', axis=1)
    df_encoded = pd.get_dummies(features, drop_first=True)
    # Add Churn back
    df_encoded['Churn'] = target 
    print("Feature encoding complete.")
    return df_encoded

if __name__ == "__main__":
    # Test block
    import data_loader
    raw_df = data_loader.load_data()
    if raw_df is not None:
        cleaned_df = clean_data(raw_df)
        encoded_df = encode_features(cleaned_df)
        print("Preprocessing test successful!")
        print(encoded_df.head())