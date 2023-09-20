from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022, 4, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'police_api_workflow',
    default_args=default_args,
    schedule_interval=timedelta(weeks=1)
)

# Define the tasks
def fetch_data():
    """
    Fetches crime data from the Police API and saves it to a CSV file.
    """
    url = 'https://data.police.uk/api/crimes-street/all-crime?poly=51.5072,-0.1276:51.5072,-0.0625:51.5226,-0.0625:51.5226,-0.1276'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df.to_csv('/Image/etl/etl_data/crime_data.csv', index=False)

def clean_data():
    """
    Cleans the crime data by dropping unnecessary columns and renaming others.
    """
    df = pd.read_csv('/Image/etl/etl_data/crime_data.csv')
    df = df.drop(['persistent_id', 'location_subtype', 'id'], axis=1)
    df = df.rename(columns={'category': 'crime_type', 'location': 'crime_location'})
    df.to_csv('/Image/etl/etl_data/cleaned_crime_data.csv', index=False)

def visualize_data():
    """
    Visualizes the types of crimes in London using a bar chart.
    """
    df = pd.read_csv('/Image/etl/etl_data/cleaned_crime_data.csv')
    counts = df['crime_type'].value_counts()
    plt.bar(counts.index, counts.values)
    plt.title('Types of Crimes in London')
    plt.xlabel('Crime Type')
    plt.ylabel('Number of Occurrences')
    plt.savefig('/Image/etl/etl_data/crime_types.png')

# Define the DAG tasks
fetch_data_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data,
    dag=dag
)

clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag
)

visualize_data_task = PythonOperator(
    task_id='visualize_data',
    python_callable=visualize_data,
    dag=dag
)

# Set task dependencies
fetch_data_task >> clean_data_task >> visualize_data_task
