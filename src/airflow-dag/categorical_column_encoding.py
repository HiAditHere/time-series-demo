import pickle
import pandas as pd
from category_encoders import WOEEncoder
import gcsfs

fs = gcsfs.GCSFileSystem()

def encode_categorical_col(data):
    ''' Encodes Categorical Col using WOE Encoder'''
    df = pickle.loads(data)
   
    for col in ['city','job','merchant', 'category']:
        df[col] = WOEEncoder().fit_transform(df[col],df['is_fraud'])

    gcs_train_data_path = "gs://credit-card-fraud-detection-group5/data/train/train_data.csv"

    with fs.open(gcs_train_data_path, 'w') as f:
        df.to_csv(f, index=False)