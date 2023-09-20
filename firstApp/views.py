from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage
from .models import upload
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
import pandas as pd
import numpy as np


img_height, img_width=200,200


labelInfo = pd.read_csv('./models/class_names.csv')

model=load_model('./models/Dogbreed_main.h5')



def index(request):
    context={'a':1}
    return render(request,'indexN.html',context)


def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName

    img = tf.keras.utils.load_img(testimage, target_size=(200, 200))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array/255
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)

    lab = np.argsort(prediction)
    l = lab.tolist()
    pred_list = pd.DataFrame(l)
    preds = pred_list.T
    pred_a = preds.iloc[-1, 0]
    pred_b = preds.iloc[-2, 0]

    score = tf.nn.softmax(prediction[0])
    t_score = 100* np.max(score)

    if t_score > 1.5:
        predictedLabel= 'This Dog is a ', labelInfo.iloc[pred_a, 0]
    elif 1.0 < t_score < 1.5:
        predictedLabel= 'This is a hard one but I think this Dog is either a',\
        labelInfo.iloc[pred_b, 0], ' or ', labelInfo.iloc[pred_a, 0]
    else:
        predictedLabel= \
        "I Cannot recongnize this breed can you show me another Image. I am still learning"


    DogBreed =  str(predictedLabel)

    DogBreed = DogBreed.replace("'", '')
    DogBreed = DogBreed.replace(",", '')
    DogBreed = DogBreed.replace("(", '')
    DogBreed = DogBreed.replace(")", '')

    


    context={'filePathName':filePathName,'predictedLabel':DogBreed}
    return render(request,'DogBreed.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 



def DogBreed(request):
    return render(request, 'DogBreed.html')



def cv(request):
    if request.method=='POST':
        title=request.POST['title']       
        upload1=request.FILES['upload']
        object=upload.objects.create(title=title,upload=upload1)
        object.save()  
    context=upload.objects.all()
    return render(request,'cv.html',{'cvcontext':context})