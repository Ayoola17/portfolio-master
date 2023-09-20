from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import plotly.express as px
import plotly.graph_objs as go
from collections import ChainMap
# Define the DAG
default_args = {
    'owner': 'Ayoola',
    'start_date': datetime(2023, 3, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'task_timeout': timedelta(minutes=15)  # Set the task timeout to 30 minutes
}

dag = DAG(
    'police_api_workflow_main',
    default_args=default_args,
    schedule_interval=timedelta(weeks=4)
)


#extract



def fetch_data():
    
    """
    Cleans the crime data by dropping unnecessary columns and renaming others.
    """
    #define time
    now = datetime.now() # Get the current date and time
    last_month = (now - relativedelta(months=1)).strftime("%Y-%m-%d") # Get the date of last month as a string in the format "YYYY-MM-DD"
    two_months_ago = (now - relativedelta(months=2)).strftime("%Y-%m") # Get the date of 2 months ago as a string in the format "YYYY-MM-DD"
    date = two_months_ago
    
    
    locations = [
        {"name": "London", "lat": 51.5074, "lng": -0.1278, "date": date},
        {"name": "Manchester", "lat": 53.4808, "lng": -2.2426, "date": date},
        {"name": "Birmingham", "lat": 52.4862, "lng": -1.8904,"date": date} ,
        {"name": "Hatfield", "lat": 51.7613, "lng": -0.2407, "date": date},
        {"name": "St Albans", "lat": 51.7527, "lng": -0.3394, "date": date},
        {"name": "Welwyn Garden City", "lat": 51.8032, "lng": -0.2087, "date": date},
        {"name": "Watford", "lat": 51.6611, "lng": -0.3970, "date": date}, 
        {"name": "Liverpool", "lat": 53.4084, "lng": -2.9916, "date": date},
        {"name": "Leeds", "lat": 53.8008, "lng": -1.5491, "date": date},
        {"name": "Newcastle upon Tyne", "lat": 54.9783, "lng": -1.6178, "date": date},
        {"name": "Bristol", "lat": 51.4545, "lng": -2.5879, "date": date},
        {"name": "Oxford", "lat": 51.7520, "lng": -1.2577, "date": date},
        {"name": "Cambridge", "lat": 52.2053, "lng": 0.1218, "date": date},
        {"name": "Brighton", "lat": 50.8225, "lng": -0.1372, "date": date},
        {"name": "Southampton", "lat": 50.9097, "lng": -1.4044, "date": date},
        {"name": "Cardiff", "lat": 51.4816, "lng": -3.1791, "date": date},
        {"name": "Belfast", "lat": 54.5973, "lng": -5.9301, "date": date},
        {"name": "Leicester", "lat": 52.6369, "lng": -1.1398, "date": date},
        {"name": "Nottingham", "lat": 52.9548, "lng": -1.1581, "date": date},
        {"name": "Sheffield", "lat": 53.3811, "lng": -1.4701, "date": date},
            ]
    crime_endpoint = "https://data.police.uk/api/crimes-street/all-crime"
    location_endpoint = "https://data.police.uk/api/locate-neighbourhood"
    force_endpoint = "https://data.police.uk/api/forces"
    
    def get_crime_data(latitude, longitude, date):
        params = {"lat": latitude, "lng": longitude, "date": date}
        response = requests.get(crime_endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print("Get crime data failed with status code:", response.status_code)
            return None

    # Define the function to get the nearest police station for a given location
    def get_nearest_police_station(latitude, longitude):
        params = {"q": f"{latitude},{longitude}"}
        response = requests.get(location_endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print("Get nearest police station failed with status code:",
                  response.status_code)
            return None

        
    #re run cell if connection fails
    # Create an empty list to store the crime data and police station information
    crime_data = []

    # Iterate over the list of locations, and call the "get_crime_data" and
    #"get_nearest_police_station" functions for each location
    for location in locations:
        # Get the crime data for the location and date range
        crime_json = get_crime_data(location["lat"], 
                                    location["lng"], 
                                    location["date"], 
                                    )
        # Get the nearest police station for the location
        police_force = get_nearest_police_station(location["lat"], 
                                                  location["lng"]
                                                  )
        # Create a dictionary to store the crime data and police station information
        for crime in crime_json:
            data = {
                "date": crime["month"],
                "crime_id": crime["persistent_id"],
                "location": location["name"],            
                "street": crime['location']['street']['name'],
                "latitude": crime["location"]["latitude"],
                "longitude": crime["location"]["longitude"],
                "category": crime["category"],
                "outcome": crime["outcome_status"]["category"]\
                        if "outcome_status" in crime and crime["outcome_status"]\
                          is not None else "No outcome provided",
                "police_force": police_force["force"],
                "station_id": police_force["neighbourhood"]
            }
            crime_data.append(data)

    # Create a pandas DataFrame from the crime data
    crime_df = pd.DataFrame(crime_data)
    response = requests.get(force_endpoint)
    data = response.json()

    #create dictionary
    forces_dict = {}
    for force in data:
        forces_dict[force['id']] = force['name']

    #map dictionary to police_force column to create new column
    crime_df['police_force_name'] = crime_df['police_force'].map(forces_dict)
    

    
    crime_df.to_csv('/Image/etl/etl_data/crime_data.csv', index=False)


    


def clean_data():
    """
    Cleans the crime data by dropping unnecessary columns and renaming others.
    """
    
    #station endpoint
    station_endpoint = "https://data.police.uk/api/"
        
    # Define funtion that get police station name
    def police_station_name(force, force_id):
        response = requests.get(f'{station_endpoint}{force}/{force_id}')
        if response.status_code == 200:
            return response.json()["name"]
        else:
            print("Get police station name failed with status code:",
                  response.status_code)
            return None
    
    crime_df = pd.read_csv('/Image/etl/etl_data/crime_data.csv')
    #select unique combinationb of columnn police_force and station_id
    unique = crime_df[['police_force', 'station_id']].drop_duplicates()

    #convert to list
    force = unique['police_force'].tolist()
    station = unique['station_id'].tolist()

    #create empty list
    station_name = []

    #iterate through list force and station and append result dict to station name
    for i, j in zip(force,station):
      station_name.append({j : police_station_name(i, j)})

    #covert station name list to dictionary
    station_name = dict(ChainMap(*station_name))

    #map station name to station id to create new column station name
    crime_df['station_name'] = crime_df['station_id'].map(station_name)
    
    crime_df[['latitude', 'longitude']] = crime_df[['latitude', 'longitude']].apply(pd.to_numeric)
    


    crime_df.to_csv('/Image/etl/etl_data/cleaned_crime_data.csv', index=False)

    
def visualize_data():
    """
    Visualizes the types of crimes in London using a bar chart.
    """

    now = datetime.now() # Get the current date and time
    last_month = (now - relativedelta(months=1)).strftime("%Y-%m-%d") # Get the date of last month as a string in the format "YYYY-MM-DD"
    two_months_ago = (now - relativedelta(months=2)).strftime("%Y-%m") # Get the date of 2 months ago as a string in the format "YYYY-MM-DD"
    date = two_months_ago

    
    df = pd.read_csv('/Image/etl/etl_data/cleaned_crime_data.csv')

        
    #second plot data
    data = df['category'].value_counts().to_dict()
    
    grouped = df.groupby('location').agg({'crime_id': 'count', 'latitude': 'mean', 'longitude': 'mean'})
    grouped.rename(columns={'crime_id' : 'crime count'}, inplace=True)
   

    # Create a sample dataframe with latitude, longitude, and crime count columns
    df = grouped

    # Create the mapbox trace
    trace = go.Scattermapbox(
        lat=df['latitude'], # Set the latitude values
        lon=df['longitude'], # Set the longitude values
        mode='markers',
        marker=dict(
            size=df['crime count'] * 0.03, # Set the size of the bubbles based on the crime count
            color=df['crime count'], # Set the color of the bubbles based on the crime count
            colorscale='Viridis', # Set the colorscale for the color values
            showscale=True, # Show a color scale legend
            opacity=0.5 # Set the opacity of the bubbles
        ),
        text=['Crime Count: {}'.format(count) for count in df['crime count']] # Add labels for each bubble
    )

    # Create the layout for the map
    layout = go.Layout(
        mapbox=dict(
            style='open-street-map', # Set the map style
            center=dict(
                lat=df['latitude'].mean(), # Set the center of the map
                lon=df['longitude'].mean()
            ),
            zoom=4 # Set the zoom level for the map
        ),
        title=f'Crime Map for {date}',
        width=1000, # Set the width of the figure
        height=1000 # Set the height of the figure
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Save the plot as a html
    fig.write_html('/Image/template/etl/crime.html')

        
    #second plot

    labels = list(data.keys())
    values = list(data.values())

    # create the pie chart
    fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

    # update the layout
    fig1.update_layout(title='Crime Data', title_x=0.5, font=dict(family='Arial', size=14),
                      legend=dict(x=0.5, y=1.15, orientation='h', bgcolor='rgba(0,0,0,0)'),
                      annotations=[dict(text='Crime Types', x=0.5, y=0.5, font_size=20, showarrow=False)],
                      width=1000, height=1000)

    # save the plot
    fig1.write_html('/Image/template/etl/crime_plot2.html')
    
    
    

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



    