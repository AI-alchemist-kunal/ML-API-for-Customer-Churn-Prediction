import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Import functions from our other scripts
import data_loader
import preprocessing

# --- Configuration ---
MODEL_PATH = "model.pkl"
COLUMNS_PATH = "training_columns.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def train_model():
    """Loads, preprocesses, trains, and saves the churn model and columns."""
    # 1. Load Data
    print("--- 1. Loading Data ---")
    raw_df = data_loader.load_data()
    if raw_df is None:
        return

    # 2. Preprocess Data
    print("\n--- 2. Preprocessing Data ---")
    cleaned_df = preprocessing.clean_data(raw_df)
    encoded_df = preprocessing.encode_features(cleaned_df)

    # 3. Prepare for Training
    print("\n--- 3. Preparing Data for Training ---")
    X = encoded_df.drop('Churn', axis=1)
    y = encoded_df['Churn']
    training_columns = X.columns.tolist() # Save column names BEFORE splitting

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # 4. Train Model
    print("\n--- 4. Training Model ---")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    print("\n--- 5. Evaluating Model ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    # 6. Save Artifacts
    print("\n--- 6. Saving Model and Columns ---")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(training_columns, COLUMNS_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Training columns saved to {COLUMNS_PATH}")

if __name__ == "__main__":
    train_model()
    print("\n--- Training Complete ---")