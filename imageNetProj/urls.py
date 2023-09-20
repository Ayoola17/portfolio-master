"""imageNetProj URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from firstApp import views
from django.conf.urls.static import static
from django.conf import settings
from django.urls import include
from data.views import *
from etl.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$',views.index,name='homepage'),
    url('predictImage',views.predictImage,name='predictImage'),
    url('viewDataBase',views.viewDataBase,name='viewDataBase'),

    url('DogBreed',views.DogBreed,name='DogBreed'),
    url('cv', views.cv, name='cv'),
    
    #visualise_result views 
    url('visualise_result', visualise_result, name='visualise_result'),
    url('column_select', column_select, name='column_select'),
    url('clean_data', clean_dataset, name='clean_data'),
    url('cleaned_data_download', download_clean_data, name='cleaned_data_download'),
    url('simple_dashboard', simple_dashboard, name='simple_dashboard'),
    url('prediction', prediction, name='prediction'),
    url('download', download_dataframe, name='download_dataframe'),
    url('crime_types', show_crime_types, name='crime_types'),
    url('raw_crime_data', crime_data, name='raw_crime_data'),
    url('cleaned_crime_data', cleaned_crime_data, name='cleaned_crime_data')

    
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)