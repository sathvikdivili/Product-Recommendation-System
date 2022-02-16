import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model.trainable = False

model = tensorflow.keras.Sequential([
    model, GlobalMaxPooling2D()
])
st.title('Product Recommendation System')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("Choose an Image of Your Choice")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        # displaying the file
        st.image(display_image)
        # feature extraction
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)
        # st.text(features)
        # recommendation
        indices = recommend(features, feature_list)
        # display the results
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            image1 = Image.open(filenames[indices[0][0]])
            st.image(image1)
        with col2:
            image2 = Image.open(filenames[indices[0][1]])
            st.image(image2)
        with col3:
            image3 = Image.open(filenames[indices[0][2]])
            st.image(image3)
        with col4:
            image4 = Image.open(filenames[indices[0][3]])
            st.image(image4)
        with col5:
            image5 = Image.open(filenames[indices[0][4]])
            st.image(image5)
    else:
        st.header("An Error occurred while uploading a file")
