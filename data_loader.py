import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def load_data(url=DATA_URL):
    """Loads the Telco Customer Churn dataset from a URL."""
    print(f"Loading data from {url}...")
    try:
        df = pd.read_csv(url)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    df_raw = load_data()
    if df_raw is not None:
        print("Data Loading test successful!")
        print(df_raw.head())