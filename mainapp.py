# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 07:57:24 2021

@author: Dell
"""

import streamlit as st

import numpy as np 



import tensorflow as tf

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
              Developed By
             """)
             
    st.image("ravi.jpg",width=100)
    
    st.write("I am Ravi Teja Nallagatla pursuing M.tech Software Engineering in VIT University.My part of this project is to develop best suited algorithm among VGG16,CNN,Resnet50 and create a .h5 file which is helped to deploy in front end.")
    st.markdown("""
                (My LinkedIn)(https://www.linkedin.com/in/raviteja-nallagatla-ba9b241a0)
    """)
    
    
    st.image("mypic.jpg",width = 100)
    st.write("I am Dhanush Kumar manda pursuing M.tech Software Engineering in VIT University.My part of this project is to develop a front end using Streamlit framework and use the .h5 model in the page for predicting the disease.")
    
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
        st.image("matrixres.JPG",width = 500)
        st.header("""Accuracy Curve""")
        st.image("accuracy curveres.JPG",width = 500)
        st.header("""Loss Curve""")
        st.image("loss curveres.JPG",width = 500)
        
        
    if res=="Visual Geometry Group16":
        st.header("""Model Accuracy""")
        st.write("98.14%")
        st.header("""Confusion Matrix""")
        st.image("matrixvgg.JPG",width = 500)
        st.header("""Accuracy Curve""")
        st.image("accuracy curvevgg.JPG",width = 500)
        st.header("""Loss Curve""")
        st.image("loss curvevgg.JPG",width = 500)
        
        
    if res=="Convolutional Neural Network":
        st.header("""Model Accuracy""")
        st.write("92.98%")
        st.header("""Confusion Matrix""")
        st.image("maatrixcnn.JPG",width = 500) 
        st.header("""Accuracy Curve""")
        st.image("accuracy curvecnn.JPG",width = 500)
        st.header("""Loss Curve""")
        st.image("LOSS CURVEcnn.JPG",width = 500)
                   
               
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
        
    
        st.text("Probability (0: drusen, 1: normal)")
        st.write(prediction)
    