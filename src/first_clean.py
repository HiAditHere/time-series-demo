import pandas as pd
import json
import gcsfs
import os
from geopy.distance import great_circle
import numpy as np
from category_encoders import WOEEncoder

# import openpyxl

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()

train_data_gcs_path = "gs://credit-card-fraud-detection-group5/data/train/train_data.csv"

print("Start")

df = pd.read_csv(train_data_gcs_path)
df.to_csv("TP.csv")

print(1)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.weekday
df['month'] = df['trans_date_trans_time'].dt.month

print(2)
df['merchant'] = df['merchant'].apply(lambda x : x.replace('fraud_',''))

print(3)
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (df['trans_date_trans_time'].dt.year - df['dob'].dt.year).astype(int)
df.drop(columns='dob',inplace=True)

print(4)
df['distance_km'] = df.apply(lambda col : round(great_circle((col['lat'],col['long']),(col['merch_lat'],col['merch_long'])).kilometers,2),axis=1)
df.drop(columns=['lat','long','merch_lat','merch_long'],inplace=True)

print(5)
df = pd.get_dummies(df,columns=['gender'],drop_first=True)

print(6)
df.sort_values(['cc_num', 'trans_date_trans_time'],inplace=True)
df['hours_diff_bet_trans']=((df.groupby('cc_num')[['trans_date_trans_time']].diff())/np.timedelta64(1,'h'))
df['hours_diff_bet_trans'].fillna(0,inplace=True)

print(7)
freq = df.groupby('cc_num').size()
df['cc_freq'] = df['cc_num'].apply(lambda x : freq[x])

print(8)
df.drop(columns=['cc_num','city_pop'],inplace=True)
#Reorder columns
df = df[['cc_freq','city','job','age','gender_M','merchant', 'category',
        'distance_km','month','day','hour','hours_diff_bet_trans','amt','is_fraud', 'trans_date_trans_time']]

for col in ['city','job','merchant', 'category']:
    df[col] = WOEEncoder().fit_transform(df[col],df['is_fraud'])

print(9)
#mean = df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']].mean()
#std = df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']].std()

columns_to_calculate = ['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']

# Convert the selected columns to a NumPy array for faster calculations
data_array = df[columns_to_calculate].to_numpy()

# Calculate mean and standard deviation using NumPy
mean = np.mean(data_array, axis=0)
std = np.std(data_array, axis=0)

# Create a pandas Series with the results
mean = pd.Series(mean, index=columns_to_calculate)
std = pd.Series(std, index=columns_to_calculate)

print(10)
df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']] = (df[['cc_freq', 'city', 'age', 'merchant', 'category', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']] - mean)/std

gcs_train_data_path = "gs://credit-card-fraud-detection-group5/data/train/clean_train_data.csv"
normalization_stats_gcs_path = "gs://credit-card-fraud-detection-group5/data/scaler/normalization.json"

print("reached here")

with fs.open(gcs_train_data_path, 'w') as f:
    df.to_csv(f, index=False)

normalization_stats = {
    'mean': mean.to_dict(),
    'std': std.to_dict()
}
# Save the normalization statistics to a JSON file on GCS
with fs.open(normalization_stats_gcs_path, 'w') as json_file:
    json.dump(normalization_stats, json_file)