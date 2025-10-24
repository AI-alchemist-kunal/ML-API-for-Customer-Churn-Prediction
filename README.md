# Project 2: Customer Churn Prediction API

This project is an end-to-end machine learning application that predicts customer churn. The model is trained using Python scripts and deployed as a live REST API using FastAPI and Pydantic.

**Live API Demo:** `[Add your Render.com URL here once deployed]`
**Interactive Docs:** `[Add your Render.com URL]/docs`

## 1. Business Problem
A telecom company is losing customers and wants to identify which customers are at high risk of "churning" (leaving). By predicting churn, the company can proactively offer incentives to high-risk customers and reduce revenue loss.

## 2. Technical Solution
1.  **Data Loading (`data_loader.py`):** Loads the Telco Customer Churn dataset.
2.  **Preprocessing (`preprocessing.py`):** Cleans the data (handles missing values, incorrect types) and one-hot encodes categorical features using Pandas.
3.  **Model Training (`model_trainer.py`):**
    * Imports loading and preprocessing functions.
    * Splits the data into training and testing sets.
    * Trains a Logistic Regression model using Scikit-learn.
    * Saves the final trained model (`model.pkl`) and the list of feature columns used during training (`training_columns.pkl`) using Joblib.
4.  **API Development (`main.py`):**
    * A **FastAPI** server loads the saved `model.pkl` and `training_columns.pkl` on startup.
    * A **Pydantic** model (`CustomerFeatures`) defines the expected input data structure and types for API requests.
    * A `/predict` endpoint receives raw JSON data for a customer.
    * The incoming data is converted to a Pandas DataFrame, undergoes the same one-hot encoding, and is then **re-indexed** using the saved `training_columns.pkl`. This crucial step ensures the API input perfectly matches the structure the model was trained on.
    * The model predicts churn (1 or 0) and the probability of churn.
    * The prediction and probability are returned as a JSON response.

## 3. How to Run Locally
1.  Clone the main repository (`Data Science Projects`).
2.  Navigate to the `project_2_ml_api` folder: `cd project_2_ml_api`
3.  Create and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # (or .\.venv\Scripts\activate on Windows)
    ```
4.  Install dependencies: `pip install -r requirements.txt`
5.  Train the model (this creates `model.pkl` and `training_columns.pkl`):
    ```bash
    python model_trainer.py
    ```
6.  Run the API server:
    ```bash
    uvicorn main:app --reload
    ```
7.  Open `http://127.0.0.1:8000/docs` in your browser to interact with the API documentation and test predictions.