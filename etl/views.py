from django.shortcuts import render
import os
# Create your views here.
import base64
from io import BytesIO
from django.http import HttpResponse
from django.shortcuts import render
from django.utils.encoding import smart_str


def show_crime_types(request):
    
    context = {'airflow_url': 'http://localhost:8080'}
    return render(request, 'crime_types.html', context)


def crime_data(request):
    csv_data = open('./etl/etl_data/crime_data.csv', 'rb')

    # create an HTTP response with the CSV data as a file attachment
    response = HttpResponse(content=csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="raw_crime_data.csv"'

    return response


def cleaned_crime_data(request):
    csv_data = open('./etl/etl_data/cleaned_crime_data.csv', 'rb')

    # create an HTTP response with the CSV data as a file attachment
    response = HttpResponse(content=csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="cleaned_crime_data.csv"'

    return response
