import pandas as pd
import json
import gcsfs
import os
# import openpyxl

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()

def update_datasets(monthly_dataframes, train_data_gcs_path, test_data_gcs_path):
    """
    Updates the training and test datasets by adding the next month of data to the training set and the following
    month to the test set.
    :param monthly_dataframes: Dictionary containing the monthly dataframes
    :param train_data_gcs_path: GCS path to the training data
    :param test_data_gcs_path: GCS path to the test data
    :param normalization_stats_gcs_path: GCS path to save the normalization statistics to
    """
    # Sorting the months by year and month
    sorted_months = sorted(monthly_dataframes.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))

    # Attempt to load existing train and test datasets
    try:
        with fs.open(train_data_gcs_path, 'r') as f:
            train_data = pd.read_csv(f)
        with fs.open(test_data_gcs_path, 'r') as f:
            test_data = pd.read_csv(f)
    except FileNotFoundError:
        # If files do not exist, initialize the first two months as training and the third month as testing
        train_data = pd.concat([monthly_dataframes[sorted_months[0]], monthly_dataframes[sorted_months[1]]], ignore_index=True)
        test_data = monthly_dataframes[sorted_months[2]]
        next_month_index = 3  # Start from the fourth month next time
    else:
        # Find the last month in the test data to determine what to add next
        last_test_date = pd.to_datetime(test_data['trans_date_trans_time']).dt.to_period('M').max()
        last_test_year_month = str(last_test_date)
        next_month_index = sorted_months.index(last_test_year_month) + 1

        # Move the existing test data to the training set
        train_data = pd.concat([train_data, test_data], ignore_index=True)

        # Set the new month as the test data
        test_data = monthly_dataframes[sorted_months[next_month_index]]

    # Save the updated datasets to GCS
    if not train_data.empty:
        with fs.open(train_data_gcs_path, 'w') as f:
            train_data.to_csv(f, index=False)
    if not test_data.empty:
        with fs.open(test_data_gcs_path, 'w') as f:
            test_data.to_csv(f, index=False)


def main():
    # Define GCS paths for the data
    train_data_gcs_path = "gs://credit-card-fraud-detection-group5/data/train/train_data.csv"
    test_data_gcs_path = "gs://credit-card-fraud-detection-group5/data/test/test_data.csv"
    
    # air_quality_data = pd.read_excel(os.path.join("..", "data", "raw_data", "AirQualityUCI.xlsx"))
    gcs_train_data_path = "gs://credit-card-fraud-detection-group5/data/fraudTrain.csv"
    credit_card_fraud_data = pd.read_csv(gcs_train_data_path)

    print(type(credit_card_fraud_data['trans_date_trans_time'].iloc[0]))
    
    credit_card_fraud_data['YearMonth'] = pd.to_datetime(credit_card_fraud_data['trans_date_trans_time']).dt.to_period('M')
    monthly_groups = credit_card_fraud_data.groupby('YearMonth')
    monthly_dataframes = {str(period): group.drop('YearMonth', axis=1) for period, group in monthly_groups}

    # Update the datasets
    update_datasets(monthly_dataframes, train_data_gcs_path, test_data_gcs_path)

if __name__ == '__main__':
    main()