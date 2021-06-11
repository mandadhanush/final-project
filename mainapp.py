# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 07:57:24 2021

@author: Dell
"""

import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 


import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageOps


from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

nav = st.sidebar.radio("Navigation",["Home","Results","Prediction"])

if nav== "Home":
    
    st.title("""
          AI DOCTOR 
         """
         
         
         )
    st.image("doc.jpg",width = 800)
    st.write("Eye diseases are caused by many substantial reasons for a particular human it can be because of dust, light or lack of food. Drusen is an Eye disease that is caused due to pollution and dust created by construction particles .This disease is mostly caused to the people who work in polluted areas. Due to this disease people can lose their eye site if the severity of the disease is high many works done the doctor are made automatic with the help of Artificial Intelligence. In the same way I have automated the drusen disease classification and deployed it on web application so that any user can use it .I have applied three deep learning algorithms namely VGG16, Convolution neural network, Resnet50 and compared the results of all the algorithms and deployed the suited .h5 model on to the front end share stream lit platform and deploy it on the cloud where it can be accessed by any person. In that web app a user can give the OCT image of eye retina membrane and can determine whether user has been effected by drusen disease or not effected. This solves the problem of doctor’s scarcity and able to determine the results using technology.")


    st.header("""
              Develped By
             """)
    st.image("mypic.jpg",width = 100)
    
    
   
    st.write("I am Dhanush Kumar manda pursuing M.tech Software Engineering in VIT University.The main aim of the application is to make doctors work automated.")
    st.markdown("""
                (My GitHub)(https://github.com/mandadhanush)
    """)
    st.markdown("""
                (My LinkedIn)(https://www.linkedin.com/in/mandadhanush/)
    """)
    
if nav == "Results":
    st.title("""
             RESULTS
             """)
    res = st.selectbox("Select the algorithm ",["Convolutional Neural Network","Resnet50","Visual Geometry Group16"])
    
    if res=="Resnet50":
        st.header("""Model Accuracy""")
        st.write("62.60%")
        st.header("""Confusion Matrix""")
        st.image("resnet/matrix.jpg",width = 500)
        st.header("""Accuracy Curve""")
        st.image("resnet/accuracy curve.jpg",width = 500)
        st.header("""Loss Curve""")
        st.image("resnet/loss curve.jpg",width = 500)
        
        
    if res=="Visual Geometry Group16":
        st.header("""Model Accuracy""")
        st.write("98.14%")
        st.header("""Confusion Matrix""")
        st.image("VGG16/matrix.jpg",width = 500) 
        st.header("""Accuracy Curve""")
        st.image("VGG16/accuracy curve.jpg",width = 500)
        st.header("""Loss Curve""")
        st.image("VGG16/loss curve.jpg",width = 500)
        
        
    if res=="Convolutional Neural Network":
        st.header("""Model Accuracy""")
        st.write("92.98%")
        st.header("""Confusion Matrix""")
        st.image("CNN/maatrix.jpg",width = 500) 
        st.header("""Accuracy Curve""")
        st.image("CNN/accuracy curve.jpg",width = 500)
        st.header("""Loss Curve""")
        st.image("CNN/loss curve.jpg",width = 500)
                   
               
if nav == "Prediction":
    
    # Model saved with Keras model.save()
    MODEL_PATH ='model.h5'

# Load your trained model
    model = load_model(MODEL_PATH)

    def import_and_predict(img_path, model):
    
       
        size = (224,224) 
        img = ImageOps.fit(img_path,size)
        
        image = img.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]
        
        prediction = model.predict(img_reshape)
        
        return prediction


    st.write("""
         # Drusen - Disease Prediction 
         """
         
         
         )

    st.write("This is a image classification web app to predict ")

    file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
#
    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
    
        if prediction[0][1]>0.5:
            st.write("It is a normal image!")
        elif prediction[0][1]<0.5:
            st.write("It is a drusen image!")
        else:
            st.write("give correct image!")
    
        st.text("Probability (0: drusen, 1: normal)")
        st.write(prediction)
    