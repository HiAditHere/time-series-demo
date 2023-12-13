import datetime as dt
from airflow import DAG
from airflow.operators.bash_operator import BashOperator, PythonOperator
from date_transformations import date_transformations
from merchant_transformation import merchant_transformations
from extracting_age import extracting_age
from distance_calculation import distance_calculation
from gender_column import gender_ohe
from transaction_gap import transaction_gap
from card_frequency import card_frequency
from drop import drop_col
from categorical_column_encoding import encode_categorical_col
from normalize_numeric import normalize
import gcfs
import pandas as pd
import pickle

fs = gcsfs.GCSFileSystem()

LOCAL_PREPROCESS_FILE_PATH = '/tmp/preprocess.py'
GITHUB_PREPROCESS_RAW_URL = 'https://raw.githubusercontent.com/HiAditHere/time-series-demo/main/src/data_window.py'  # Adjust the path accordingly

LOCAL_TRAIN_FILE_PATH = '/tmp/train.py'
GITHUB_TRAIN_RAW_URL = 'https://raw.githubusercontent.com/HiAditHere/time-series-demo/main/src/trainer/train.py'  # Adjust the path accordingly

default_args = {
    'owner': 'Time_Series_IE7374',
    'start_date': dt.datetime(2023, 10, 24),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Model retraining at 9 PM everyday',
    schedule_interval='0 21 * * *',  # Every day at 9 pm
    catchup=False,
)

# Tasks for pulling scripts from GitHub
pull_preprocess_script = BashOperator(
    task_id='pull_preprocess_script',
    bash_command=f'curl -o {LOCAL_PREPROCESS_FILE_PATH} {GITHUB_PREPROCESS_RAW_URL}',
    dag=dag,
)

pull_train_script = BashOperator(
    task_id='pull_train_script',
    bash_command=f'curl -o {LOCAL_TRAIN_FILE_PATH} {GITHUB_TRAIN_RAW_URL}',
    dag=dag,
)

def load_data():

    gcs_train_data_path = "gs://credit-card-fraud-detection-group5/data/train/train_data.csv"
    with fs.open(gcs_train_data_path) as f:
        df = pd.read_csv(f)

    pkl_df = pickle.dumps(df)
    return pkl_df
    return df

'''

pull_load_data = BashOperator(
    task_id='pull_load_data',
    bash_command=f'curl -o {LOCAL_LOAD_DATA_PATH} {GITHUB_LOAD_DATA_PATH}',
    dag = dag,
)

pull_data_column_task = BashOperator(
    task_id='pull_data_column_task',
    bash_command=f'curl -o {LOCAL_DATA_COLUMN_TASK} {GITHUB_DATA_COLUMN_TASK}',
    dag = dag,
)

pull_merchant_column_task = BashOperator(
    task_id='pull_merchant_column_task',
    bash_command=f'curl -o {LOCAL_MERCHANT_COLUMN_TASK} {GITHUB_MERCHANT_COLUMN_TASK}',
    dag = dag,
)

pull_dob_task = BashOperator(
    task_id='pull_dob_task',
    bash_command=f'curl -o {LOCAL_DOB_TASK} {GITHUB_DOB_TASK}',
    dag = dag,
)

pull_distance_task = BashOperator(
    task_id='pull_distance_task',
    bash_command=f'curl -o {LOCAL_PULL_DISTANCE_TASK} {GITHUB_PULL_DISTANCE_TASK}',
    dag = dag,
)

pull_ohe_task = BashOperator(
    task_id='pull_ohe_task',
    bash_command=f'curl -o {LOCAL_PULL_OHE_TASK} {GITHUB_PULL_OHE_TASK}',
    dag = dag,
)

pull_transaction_gap_task = BashOperator(
    task_id='pull_transaction_gap_task',
    bash_command=f'curl -o {LOCAL_PULL_TRANSACTION_GAP_TASK} {GITHUB_PULL_TRANSACTION_GAP_TASK}',
    dag = dag,
)

pull_card_frequency_task = BashOperator(
    task_id='pull_card_frequency_task',
    bash_command=f'curl -o {LOCAL_PULL_CARD_FREQUENCY_TASK} {GITHUB_PULL_CARD_FREQUENCY_TASK}',
    dag = dag,
)

pull_drop_task = BashOperator(
    task_id='pull_drop_task',
    bash_command=f'curl -o {LOCAL_PULL_DROP_TASK} {GITHUB_PULL_DROP_TASK}',
    dag = dag,
)

pull_categorical_columns_task = BashOperator(
    task_id='pull_drop_task',
    bash_command=f'curl -o {LOCAL_PULL_CATEGORICAL_COLUMN_TASK} {GITHUB_PULL_CATEGORICAL_COLUMN_TASK}',
    dag = dag,
)
'''


env = {
    'AIP_STORAGE_URI': 'gs://credit-card-fraud-detection-group5/model'
}

# Tasks for running scripts
run_preprocess_script = BashOperator(
    task_id='run_preprocess_script',
    bash_command=f'python {LOCAL_PREPROCESS_FILE_PATH}',
    env=env,
    dag=dag,
)

run_train_script = BashOperator(
    task_id='run_train_script',
    bash_command=f'python {LOCAL_TRAIN_FILE_PATH}',
    env=env,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)


date_column_task = PythonOperator(
    task_id = 'date_column_task',
    python_callable = date_transformations,
    op_args = [load_data_task.output],
    dag=dag,
)


merchant_column_task = PythonOperator(
    task_id = 'merchant_column_task',
    python_callable = merchant_transformations,
    op_args = [date_column_task.output],
    dag=dag,
)

dob_column_task = PythonOperator(
    task_id = 'dob_column_task',
    python_callable = extracting_age,
    op_args = [merchant_column_task.output],
    dag=dag,
)

distance_task = PythonOperator(
    task_id = 'distance_task',
    python_callable = distance_calculation,
    op_args = [dob_column_task.output],
    dag =dag,
)

ohe_task = PythonOperator(
    task_id = 'ohe_task',
    python_callable = gender_ohe,
    op_args = [distance_task.output],
    dag = dag,  
)

transaction_gap_task = PythonOperator(
    task_id = 'transaction_gap',
    python_callable = transaction_gap,
    op_args = [ohe_task.output],
    dag = dag, 
)

card_frequency_task = PythonOperator(
    task_id = 'ard_frequency_task',
    python_callable = card_frequency,
    op_args = [transaction_gap_task.output],
    dag = dag, 
)

drop_task = PythonOperator(
    task_id = 'drop_task',
    python_callable = drop_col,
    op_args = [card_frequency_task.output],
    dag = dag,
)

categorical_columns_task = PythonOperator(
    task_id = 'categorical_columns_task',
    python_callable = encode_categorical_col,
    op_args = [drop_task.output],
    dag = dag, 
)

normalize_task = PythonOperator(
    task_id = 'normalize_task',
    python_callable = normalize,
    op_args = [categorical_columns_task.output],
    dag = dag, 
)


#load_data_task >> date_column_task >> merchant_column_task >> dob_column_task >> distance_task >> ohe_task >> transaction_gap_task >> card_frequency_task >> drop_task >> categorical_columns_task >> write_to_file

# Setting up dependencies
pull_preprocess_script >> pull_train_script >> run_preprocess_script >> load_data_task >> date_column_task >> merchant_column_task >> dob_column_task >> distance_task >> ohe_task >> transaction_gap_task >> card_frequency_task >> drop_task >> categorical_columns_task >> normalize_task >> run_train_script
