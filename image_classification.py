# -*- coding: utf-8 -*-
"""Image Classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LTzn7ejaN0UsrhK_LM2SMrx4XGU-TEcr
"""

# Commented out IPython magic to ensure Python compatibility.
!pip install ipython-autotime
# %load_ext autotime

# Data : Images
#1. Download manually the iages rom google
#2. Download dataset from kaggle.com
#3. Build a image web crawler
#4. Use python libraries to scrape the images (Using)

!pip install bing-image-downloader

!mkdir images

from bing_image_downloader import downloader
downloader.download("blitz defnese formation nfl", limit=30, output_dir = "images", adult_filter_off=True)

from bing_image_downloader import downloader
downloader.download("base defense nfl formation", limit=30, output_dir = "images", adult_filter_off=True)

from bing_image_downloader import downloader
downloader.download("nickel defensive formations nfl", limit=30, output_dir = "images", adult_filter_off=True)

import numpy as np

a = np.array([[1,2],[1,2]])
a.ndim

#How do i convert Matrix to Vector? - flatten()

a.flatten()

# Preprocessing
#1. resizing
#2. Flatten

import os
import matplotlib.pyplot
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt

target = []
images = []
flat_data = []

DATADIR = '/content/images'
CATEGORIES = ['base defense nfl formation', 'blitz defnese formation nfl', 'nickel defensive formations nfl']

for category in CATEGORIES:
  class_num = CATEGORIES.index(category) #Label Encoding the values
  path = os.path.join(DATADIR,category) #Create path to use all the images
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    # print(img_array.shape)
    # plt.imshow(img_array)
    # break #makes it so only the last image will be displayed
    img_resized = resize(img_array,(150,150,3)) #Normalizes the value from 0 to 1
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

len(flat_data[0])



150*150*3

target

unique,count = np.unique(target, return_counts=True)
plt.bar(CATEGORIES,count)

# Split data into Training and testing
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(flat_data,target, test_size = 0.3,
                                                   random_state = 109)

from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid = [

        {"C":[1,10,100,1000], "kernel": ["linear"]},
        {"C":[1,10,100,1000], "gamma": [0.001,0.0001],"kernel":["rbf"]},


]

svc = svm.SVC(probability=True)
clf = GridSearchCV(svc, param_grid)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_pred

y_test

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_pred, y_test)

confusion_matrix(y_pred, y_test) #evaluation metrix where all data points can be varied and check if proper output occurs or not

# Save the model using Pickle library
#makes it so that the model is stored inside it

import pickle
pickle.dump(clf,open("img_model.p", "wb"))

model = pickle.load(open("img_model.p", 'rb'))

#Testing a brand new image

#Confusion matrix =

flat_data = []

url = input("Enter")

img = imread(url)
img_resized = resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]]
print(f" PREDICTED OUTPUT: {y_out}")

!pip install streamlit

!pip install pyngrok
# from pyngrok import ngrok

#deployment:
 #1. webpage - HTML/CSS/ JS - Flask/Django
 #2. webapp - Streamlit/Dash
 #3. mobile app - kotlin/Java

# %%writefile app.py
import streaklit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import pickle
from PTL import Image


import streamlit as st

st.title("Image Classifier")
st.text("upload the image")

model = pickle.load(open("img_model.p", 'rb'))
uploaded_file = st.file_uploader("Choose an image...", type = "jpg")

if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption = "Uploaded Image")

if st.button("PREDICT"):
  st.write("Result...")
  flat_data = []
  img = np.array(img)
  img_resized = resize(img,(150,150,3))
  flat_data.append(img_resized.flatten())
  flat_data = np.array(flat_data)
  plt.imshow(img_resized)
  y_out = model.predict(flat_data)
  y_out = CATEGORIES[y_out[0]]
  st.title(f" PREDICTED OUTPUT: {y_out}")

!nohup streamlit run app.py &

url = ngrok.connect(port = '8501')
url