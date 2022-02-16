# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e37-NkbcDlm3bcBfvapa39uySy_vWDFH
"""

import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model.trainable = False  # we are not training the model. we are using the pre rained model

model = tensorflow.keras.Sequential([
    model, GlobalMaxPooling2D()
])  # adding the GlobalMaxPooling2D layer as a top layer for the model

print(model.summary())

import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle


# to determine progress of loop

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))