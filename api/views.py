import urllib
from django.shortcuts import render
import numpy as np
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
import cloudinary.uploader
import tensorflow as tf
#import matplotlib.pyplot as plt
import cv2

# Create your views here.
class UploadView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def post(request):
        file = request.data.get('picture')
        upload_data = cloudinary.uploader.upload(file)
        #print(upload_data)
        img_url = upload_data['url']


        #load models
        model = ApiConfig.model

        req = urllib.request.urlopen(img_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1) # 'Load it as it is'
        #image = cv2.imread('upload_chest.jpg') # read file 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
        image = cv2.resize(image,(150,150))
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        score = tf.nn.softmax(predictions[0])
        
        class_names = ['Benign','Malignant','Normal']
        result = class_names[np.argmax(score)]
        report = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
        print(report)
        
        return Response({
            'status': 'success',
            'result':result,
            'report':report
        }, status=201)


