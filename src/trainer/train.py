from google.cloud import storage
from datetime import datetime
import pytz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import gcsfs
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize variables
fs = gcsfs.GCSFileSystem()
storage_client = storage.Client()
bucket_name = os.getenv("BUCKET_NAME")
MODEL_DIR = os.environ['AIP_STORAGE_URI']

def load_data(gcs_train_data_path):
    """
    Loads the training data from Google Cloud Storage (GCS).
    
    Parameters:
    gcs_train_data_path (str): GCS path where the training data CSV is stored.
    
    Returns:
    DataFrame: A pandas DataFrame containing the training data.
    """
    with fs.open(gcs_train_data_path) as f:
        df = pd.read_csv(f)
    
    return df


def train_model(X_train, y_train):
    """
    Trains a Random Forest Regressor model on the provided data.
    
    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series): The training labels.
    
    Returns:
    RandomForestRegressor: The trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_and_upload_model(model, local_model_path, gcs_model_path):
    """
    Saves the model locally and uploads it to GCS.
    
    Parameters:
    model (RandomForestRegressor): The trained model to be saved and uploaded.
    local_model_path (str): The local path to save the model.
    gcs_model_path (str): The GCS path to upload the model.
    """
    # Save the model locally
    joblib.dump(model, local_model_path)

    # Upload the model to GCS
    with fs.open(gcs_model_path, 'wb') as f:
        joblib.dump(model, f)

def main():
    """
    Main function to orchestrate the loading of data, training of the model,
    and uploading the model to Google Cloud Storage.
    """
    # Load and transform data
    gcs_train_data_path = "gs://credit-card-fraud-detection-group5/data/train/clean_train_data.csv"
    df = load_data(gcs_train_data_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['trans_date_trans_time', 'is_fraud']), df['is_fraud'], test_size=0.2)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the model locally and upload to GCS
    edt = pytz.timezone('US/Eastern')
    current_time_edt = datetime.now(edt)
    version = current_time_edt.strftime('%Y%m%d_%H%M%S')
    local_model_path = "model.pkl"
    gcs_model_path = f"{MODEL_DIR}/model_{version}.pkl"
    print(gcs_model_path)
    save_and_upload_model(model, local_model_path, gcs_model_path)

if __name__ == "__main__":
    main()