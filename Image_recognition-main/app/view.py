from django.shortcuts import render
from . forms import MyForm
import pandas as pd
from django.http import JsonResponse
from django.http import HttpResponse
#from pandas import read_csv
import numpy as np
#from sklearn import tree
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score
#import joblib as joblib
#import argparse
#import os
from rest_framework.parsers import JSONParser
from .models import note_image
from .serializer import NoteSerializer_image_recognition
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core import serializers
from rest_framework import status
from django.http import JsonResponse
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import pickle
from tensorflow.keras.preprocessing import image
#import urllib.request
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense

from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing import image
import urllib.request


class Note_image_recognition_view(viewsets.ModelViewSet):
    queryset=note_image.objects.all()
    serializer_class=NoteSerializer_image_recognition

def myform(request):
     if request.method=="post":
         form=MyForm(request.post)
         if form.is_valid():
             myform=form.save(commit=False)
     else:
         form=MyForm()


@api_view(["POST"])
def imagerecognition(request):
    try:
        #import numpy as np
        #from tensorflow.keras.preprocessing import image
        picture=request.data
        #dictionary = {"Name":"Bob", "Age":18, "Occupation":"Student"}
        #picture = namedtuple("ObjectName", picture.keys())(*picture.values())
        #picture=np.array(list(picture.values()))
        #x = dict2obj(picture)
        #print("this is the type of image")
        #print(x)
        #picture=(picture['image'])
        print(picture['image'])
        picture=(picture['image'])
        print(picture)
        print(type(picture))
        #picture = picture.tostring()
        #print(np.fromstring(picture, dtype=int))
        #print(type(picture))
        #from tensorflow.compat.v1 import ConfigProto
        #from tensorflow.compat.v1 import InteractiveSession

        #config = ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        #config.gpu_options.allow_growth = True
        #session = InteractiveSession(config=config)
        # Importing the libraries
        
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
        
        training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

        # Preprocessing the Test set
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_set = test_datagen.flow_from_directory('dataset/test',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
        
        # Part 2 - Building the CNN
        # Initialising the CNN
        cnn = tf.keras.models.Sequential()

        # Step 1 - Convolution
        cnn.add(tf.keras.layers.Conv2D(filters=32,padding="same",kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]))

        # Step 2 - Pooling
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Adding a second convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Step 3 - Flattening
        cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Step 5 - Output Layer
        #cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ## For Binary Classification
        cnn.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation
                         ='linear'))
        ## for mulitclassification
        cnn.add(Dense(4, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation
                         ='softmax'))
        cnn.compile(optimizer = 'adam', loss = 'squared_hinge', metrics = ['accuracy'])
        # Part 3 - Training the CNN

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])

        # Training the CNN on the Training set and evaluating it on the Test set
        r=cnn.fit(x = training_set, validation_data = test_set, epochs = 15)
        # import matplotlib.pyplot as plt
        # plt.plot(r.history['loss'], label='train loss')
        # plt.plot(r.history['val_loss'], label='val loss')
        # plt.legend()
        # plt.show()
        # plt.savefig('LossVal_loss')

        # # plot the accuracy
        # plt.plot(r.history['accuracy'], label='train acc')
        # plt.plot(r.history['val_accuracy'], label='val acc')
        # plt.legend()
        # plt.show()
        # plt.savefig('AccVal_acc')
        # from tensorflow.keras.models import load_model

        # cnn.save('model_rcat_dog.h5')
        # from tensorflow.keras.models import load_model
 
        # load model
        # model = load_model('model_rcat_dog.h5')
        # model.summary()
        
        image_url = picture#the image on the web
        save_name = 'my_image.jpg' #local name to be saved
        urllib.request.urlretrieve(image_url, save_name)
        test_image = image.load_img("my_image.jpg", target_size = (64,64))
        #image = tf.keras.preprocessing.image.load_img(my_image.jpg)
        # test_image=picture
        # print(test_image)
        # print(type(picture))
        test_image = image.img_to_array(test_image)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        print("this is reult")
        print(result)
        print(type(result))
        print ("Sum of all array elements:",
                            result.sum())
        result=result.sum()
        #print(type(result))
        #print(result(0))
        
        def myFunction(result): #you can add variables to the function
            if result==1:
                result="The image classified is Covid positive"
                return result
            else:
                result="The image classified is Covid Negative"
                return result
        Final=myFunction(result)
        output= {           
         "data":{  
                 "details":Final
         }

         }      
        return Response(output)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)