import pickle
import pandas as pd
import json
from category_encoders import WOEEncoder
import gcsfs

fs = gcsfs.GCSFileSystem()

def normalize(data):
    '''Normalize numeric columns and upload to cloud storage'''
    df = pickle.loads(data)

    mean = df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']].mean()
    std = df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']].std()
   
    df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']] = (df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distanc_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']] - mean)/std

    gcs_train_data_path = "gs://credit-card-fraud-detection-group5/data/train/clean_train_data.csv"
    normalization_stats_gcs_path = "gs://credit-card-fraud-detection-group5/data/scaler/normalization.json"

    with fs.open(gcs_train_data_path, 'w') as f:
        df.to_csv(f, index=False)

    normalization_stats = {
        'mean': mean.to_dict(),
        'std': std.to_dict()
    }
    # Save the normalization statistics to a JSON file on GCS
    with fs.open(normalization_stats_gcs_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)