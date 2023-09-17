from django.shortcuts import render
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from . models import Brain
import os

# Create your views here.
Classes=['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

def index(request):
    return render(request, "index.html")


def about(request):
    return render(request,"about.html")


def upload(request):
    if request.method=='POST':
        
        m1 = int(request.POST['alg'])
        
        File=request.FILES['brain']
        s=Brain(image=File)
        s.save()
        path1='app/static/saved/' + s.Imagename()

        print(path1)

        if m1==1:
            model=load_model('app/demo1/alexnet.h5',compile=False)
            x1=image.load_img(path1,target_size=(128,128))
            x1=image.img_to_array(x1)
            x1/=255
            
        elif m1==2:
            model=load_model('app/demo1/resnet.h5',compile=False)
            x1=image.load_img(path1,target_size=(128,128))
            x1=image.img_to_array(x1)
            x1/=255

        elif m1==3:
            model=load_model('app/demo1/ann2.h5',compile=False)
            x1=image.load_img(path1,target_size=(128,128))
            x1=image.img_to_array(x1)
            x1/=255
        
        elif m1==4:
            model=load_model('app/demo1/mobilenet.h5',compile=False)
            x1=image.load_img(path1,target_size=(224,224))
            x1=image.img_to_array(x1)
            x1/=255

        elif m1==5:
            model=load_model('app/demo1/InceptionResNetv2.h5',compile=False)
            x1=image.load_img(path1,target_size=(128,128))
            x1=image.img_to_array(x1)
            x1/=255
        
        

        x1=np.expand_dims(x1,axis=0)
        result= model.predict(x1)        
        pred = Classes[np.argmax(result)]
        print(pred)
        return render(request,"result.html",{"message":pred,"path":'/static/saved/' + s.Imagename()})

    return render (request,"upload.html")

