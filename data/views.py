from django.shortcuts import render


from django.http import HttpResponse, HttpResponseBadRequest
import pandas as pd
from .data import data
import matplotlib.pyplot as plt
import io
import urllib, base64
from django import forms



# Create your views here.

#selecting target columns
class UploadFileForm(forms.Form):
    file = forms.FileField()

def column_select(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            df = pd.read_csv(file)
            column_names = list(df.columns)
            request.session['csv_data'] = df.to_dict()
            form = SelectColumnForm(column_names)
            return render(request, 'select_column.html', {'form': form})
    else:
        form = UploadFileForm()
    return render(request, 'upload_file.html', {'form': form})

class SelectColumnForm(forms.Form):
    column_name = forms.ChoiceField()

    def __init__(self, column_names, *args, **kwargs):
        super(SelectColumnForm, self).__init__(*args, **kwargs)
        self.fields['column_name'].choices = [(c, c) for c in column_names]







#visualise views
def visualise_result(request):

    csv_data = request.session.get('csv_data')
    target_column = request.GET.get('column_name')
    request.session['column_name'] = target_column

    if csv_data and target_column:

        # run the data through the Python class to generate a visualization
        try:
            my_class = data(csv_data, target_column)
            image_data = my_class.visualise_data()
            image_data = base64.b64encode(image_data.content).decode('utf-8')

            # generate the HTML response with the embedded image
            context = {'image_data': image_data} #base64.b64encode(image_data).decode('utf-8')}
            return render(request, 'visualization.html', context)

        except Exception as e:
            return render(request, 'error.html', {'error_message':str(e)})

    

def clean_dataset(request):

    csv_data = request.session.get('csv_data')
    target_column = request.GET.get('column_name')

    if csv_data and target_column:
        try:
            my_class = data(csv_data, target_column)
            missing_values = my_class.clean_data()[0]
            clean_data = my_class.clean_data()[1]

            request.session['clean'] = my_class.clean_data()[2].to_dict()

            missing_values = missing_values.to_html()

            clean_data = clean_data.to_html()

            missing_values_visual = my_class.visual_missing_values()

            # Convert the PNG image data to base64 encoding
            missing_values_visual = base64.b64encode(missing_values_visual.content).decode('utf-8')

        
            context = {'missing_values':missing_values, 'clean_data' : clean_data,
                    'missing_values_visual':missing_values_visual }
            return render(request, 'clean_data.html', context)
        
        except Exception as e:
            return render(request, 'error.html', {'error_message':str(e)})

def simple_dashboard(request):

    csv_data = request.session.get('csv_data')
    target_column = request.GET.get('column_name')
     
    if csv_data and target_column:

        try:
            my_class = data(csv_data, target_column)
            dashboard = my_class.plot_columns()

            # Convert the PNG image data to base64 encoding
            dashboard = base64.b64encode(dashboard.content).decode('utf-8')

            context = {'dashboard': dashboard}

            return render(request, 'dashboard.html', context)
        
        except Exception as e:
            return render(request, 'error.html', {'error_message':str(e)})

def prediction(request):
    if request.method == 'POST':
        csv_data = request.session.get('csv_data')
        target_column = request.session.get('column_name') 
        # get the uploaded file from the request
        file = request.FILES.get('csv_file')
        df = pd.read_csv(file)
        test_csv = df.to_dict()
        
        if not file:
            return HttpResponse("No file uploaded")
        
        try:
            my_class = data(csv_data, target_column, test_data=test_csv)
            request.session['predicted_data'] = my_class.predictions()[1].to_dict()

            predicted_head = my_class.predictions()[0].to_html()
            visual_prediction = my_class.visual_prediction()

            # Convert the PNG image data to base64 encoding
            visual_prediction = base64.b64encode(visual_prediction.content).decode('utf-8')


            context = {#'predicted_df': predicted_df,
                    'predicted_head': predicted_head,
                    'visual_prediction' : visual_prediction,
                    }
        
            return render(request, 'prediction.html', context)
        except Exception as e:
            return render(request, 'error.html', {'error_message':str(e)})



def download_dataframe(request):
    # create a Pandas DataFrame with some data
    #df = request.session.get('predicted_data')
        # create a Pandas DataFrame with some data
    data = request.session.get('predicted_data')
    df = pd.DataFrame.from_dict(data)


    # create a CSV file from the DataFrame
    csv_data = df.to_csv(index=False)

    # create an HTTP response with the CSV data as a file attachment
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction.csv"'

    return response



def download_clean_data(request):
    data = request.session.get('clean')
    df = pd.DataFrame.from_dict(data)


    # create a CSV file from the DataFrame
    csv_data = df.to_csv(index=False)

    # create an HTTP response with the CSV data as a file attachment
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="clean.csv"'

    return response
          